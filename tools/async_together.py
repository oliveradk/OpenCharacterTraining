"""
Async Together AI client with semaphore-controlled concurrency.

Adapted from audit-aware/async_together.py for OpenCharacterTraining.

Usage:
    from tools.async_together import AsyncTogetherChatClient

    client = AsyncTogetherChatClient(max_concurrency=10)

    requests = [
        {"messages": [{"role": "user", "content": "Hello"}], "model": "meta-llama/Llama-3.1-8B-Instruct"},
    ]

    results = await client.batch_complete(requests)
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import aiohttp


def load_env() -> None:
    """Load environment variables from .env file in repo root."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parents[2] / ".env"

    if env_path.exists():
        with open(env_path) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.startswith("export "):
                        key = key[7:]
                    if value and key not in os.environ:
                        os.environ[key] = value


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


class AsyncTogetherChatClient:
    """Async Together AI chat client with concurrency control."""

    BASE_URL = "https://api.together.xyz/v1/chat/completions"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrency: int = 10,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        timeout: float = 120.0,
    ):
        load_env()
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set")

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

        messages = request.get("messages", [])
        model = request.get("model")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.get("max_tokens", 512),
            "temperature": request.get("temperature", 0.7),
        }

        if "top_p" in request:
            payload["top_p"] = request["top_p"]
        if "stop" in request:
            payload["stop"] = request["stop"]

        response_format = request.get("response_format")
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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

    async def batch_complete(
        self,
        requests: list[dict],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[CompletionResult]:
        """Execute multiple chat requests with concurrency control."""
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


async def batch_chat_complete(
    requests: list[dict],
    max_concurrency: int = 10,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> list[CompletionResult]:
    """
    Convenience function for batch chat completions.

    Args:
        requests: List of request dicts
        max_concurrency: Maximum concurrent requests
        api_key: Together API key (uses env var if not provided)
        progress_callback: Called with (completed, total) after each request
        **kwargs: Additional args passed to AsyncTogetherChatClient

    Returns:
        List of CompletionResult objects.
    """
    client = AsyncTogetherChatClient(
        api_key=api_key,
        max_concurrency=max_concurrency,
        **kwargs,
    )
    return await client.batch_complete(requests, progress_callback=progress_callback)
