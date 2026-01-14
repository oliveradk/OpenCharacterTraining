"""
Async OpenRouter API client with semaphore-controlled concurrency.

Usage:
    from async_openrouter import AsyncOpenRouterClient, batch_complete

    client = AsyncOpenRouterClient(max_concurrency=10)

    requests = [
        {"messages": [{"role": "user", "content": "Hello"}], "model": "anthropic/claude-3-haiku"},
        {"messages": [{"role": "user", "content": "World"}], "model": "anthropic/claude-3-haiku"},
    ]

    results = await client.batch_complete(requests)
    # Or use the convenience function:
    results = await batch_complete(requests, max_concurrency=10)
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional
import aiohttp


@dataclass
class CompletionResult:
    """Result of a single completion request."""
    content: Optional[str]
    error: Optional[str]
    request_index: int
    success: bool

    @property
    def text(self) -> str:
        """Alias for content, returns empty string on error."""
        return self.content or ""


# Model name mapping from Anthropic to OpenRouter format
MODEL_MAP = {
    'claude-3-haiku-20240307': 'anthropic/claude-3-haiku',
    'claude-3-sonnet-20240229': 'anthropic/claude-3-sonnet',
    'claude-3-opus-20240229': 'anthropic/claude-3-opus',
    'claude-3-5-sonnet-20240620': 'anthropic/claude-3.5-sonnet',
    'claude-3-5-sonnet-20241022': 'anthropic/claude-3.5-sonnet',
    'claude-sonnet-4-20250514': 'anthropic/claude-sonnet-4',
    'claude-opus-4-20250514': 'anthropic/claude-opus-4',
    'claude-opus-4-5-20251101': 'anthropic/claude-opus-4.5',
}


def normalize_model_name(model: str) -> str:
    """Convert Anthropic model name to OpenRouter format if needed."""
    return MODEL_MAP.get(model, model if '/' in model else f'anthropic/{model}')


class AsyncOpenRouterClient:
    """Async OpenRouter client with concurrency control."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrency: int = 10,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        request: dict,
        request_index: int,
    ) -> CompletionResult:
        """Make a single API request with retries."""
        semaphore = self._get_semaphore()

        # Build request payload
        messages = request.get("messages", [])
        model = normalize_model_name(request.get("model", "anthropic/claude-3-haiku"))

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.get("max_tokens", 150),
            "temperature": request.get("temperature", 1.0),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/OpenCharacterTraining",
        }

        last_error = None

        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.BASE_URL,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        if response.status == 429:
                            # Rate limited - wait and retry
                            retry_after = float(response.headers.get("Retry-After", self.retry_delay * (attempt + 1)))
                            await asyncio.sleep(retry_after)
                            continue

                        response.raise_for_status()
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]

                        return CompletionResult(
                            content=content,
                            error=None,
                            request_index=request_index,
                            success=True,
                        )

                except asyncio.TimeoutError:
                    last_error = "Request timed out"
                except aiohttp.ClientError as e:
                    last_error = f"HTTP error: {e}"
                except (KeyError, IndexError) as e:
                    last_error = f"Invalid response format: {e}"
                except Exception as e:
                    last_error = f"Unexpected error: {e}"

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        return CompletionResult(
            content=None,
            error=last_error,
            request_index=request_index,
            success=False,
        )

    async def complete(
        self,
        messages: list[dict],
        model: str = "anthropic/claude-3-haiku",
        max_tokens: int = 150,
        temperature: float = 1.0,
    ) -> CompletionResult:
        """Make a single completion request."""
        request = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with aiohttp.ClientSession() as session:
            return await self._make_request(session, request, 0)

    async def batch_complete(
        self,
        requests: list[dict],
        progress_callback: Optional[callable] = None,
    ) -> list[CompletionResult]:
        """
        Execute multiple completion requests with concurrency control.

        Args:
            requests: List of request dicts, each containing:
                - messages: List of message dicts
                - model: (optional) Model name
                - max_tokens: (optional) Max tokens
                - temperature: (optional) Temperature
            progress_callback: (optional) Called with (completed, total) after each request

        Returns:
            List of CompletionResult objects in same order as input requests.
        """
        if not requests:
            return []

        completed = 0
        total = len(requests)

        async def tracked_request(session: aiohttp.ClientSession, req: dict, idx: int) -> CompletionResult:
            nonlocal completed
            result = await self._make_request(session, req, idx)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return result

        async with aiohttp.ClientSession() as session:
            tasks = [
                tracked_request(session, req, idx)
                for idx, req in enumerate(requests)
            ]
            results = await asyncio.gather(*tasks)

        return list(results)


async def batch_complete(
    requests: list[dict],
    max_concurrency: int = 10,
    api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None,
    **kwargs,
) -> list[CompletionResult]:
    """
    Convenience function for batch completions.

    Args:
        requests: List of request dicts
        max_concurrency: Maximum concurrent requests
        api_key: OpenRouter API key (uses env var if not provided)
        progress_callback: Called with (completed, total) after each request
        **kwargs: Additional args passed to AsyncOpenRouterClient

    Returns:
        List of CompletionResult objects.
    """
    client = AsyncOpenRouterClient(
        api_key=api_key,
        max_concurrency=max_concurrency,
        **kwargs,
    )
    return await client.batch_complete(requests, progress_callback=progress_callback)


# Helper to run async code from sync context
def run_batch_complete(
    requests: list[dict],
    max_concurrency: int = 10,
    **kwargs,
) -> list[CompletionResult]:
    """
    Sync wrapper for batch_complete.

    Usage:
        results = run_batch_complete(requests, max_concurrency=10)
    """
    return asyncio.run(batch_complete(requests, max_concurrency=max_concurrency, **kwargs))


if __name__ == "__main__":
    # Quick test
    async def test():
        client = AsyncOpenRouterClient(max_concurrency=5)

        requests = [
            {
                "messages": [{"role": "user", "content": f"Say the number {i}"}],
                "model": "anthropic/claude-3-haiku",
                "max_tokens": 20,
            }
            for i in range(5)
        ]

        def progress(done, total):
            print(f"Progress: {done}/{total}")

        results = await client.batch_complete(requests, progress_callback=progress)

        for r in results:
            if r.success:
                print(f"[{r.request_index}] {r.text[:50]}")
            else:
                print(f"[{r.request_index}] ERROR: {r.error}")

    asyncio.run(test())
