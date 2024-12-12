import hashlib
import time
from functools import wraps
from typing import Any, Callable
from fastapi import HTTPException, Request


def rate_limit(max_calls: int, period: int):
    def decorator(func: Callable) -> Callable:
        usage: dict[str, list[float]] = {}

        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            request: Request = kwargs.get("request")
            if not request:
                raise ValueError("Request object is required")

            # get the client's IP address
            if not request.client:
                raise ValueError("Request has no client information")
            ip_address: str = request.client.host

            # create a unique identifier for the client
            unique_id: str = hashlib.sha256((ip_address).encode()).hexdigest()
            

            # update the timestamps
            now = time.time()
            if unique_id not in usage:
                usage[unique_id] = []
            timestamps = usage[unique_id]
            timestamps[:] = [t for t in timestamps if now - t < period]

            if len(timestamps) < max_calls:
                timestamps.append(now)
                return await func(*args, **kwargs)

            # calculate the time to wait before the next request
            wait = period - (now - timestamps[0])
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {wait:.2f} seconds",
            )

        return wrapper

    return decorator
