import hmac
import os
import time
from collections import defaultdict, deque
from typing import Deque, DefaultDict

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.api_key = os.getenv('API_KEY', '').strip()
        self.exempt_paths = {'/health', '/docs', '/openapi.json', '/redoc'}
        self._failed_attempts: DefaultDict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if not self.api_key or request.url.path in self.exempt_paths:
            return await call_next(request)

        ip = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
        now = time.monotonic()
        recent_failures = self._failed_attempts[ip]

        while recent_failures and now - recent_failures[0] >= 300:
            recent_failures.popleft()

        if len(recent_failures) >= 5:
            return JSONResponse(
                status_code=429,
                content={'detail': 'Too many failed attempts. Try again in 5 minutes.'},
                headers={"Retry-After": "300"}
            )

        provided_key = request.headers.get('x-api-key', '').strip()
        if not hmac.compare_digest(provided_key, self.api_key):
            recent_failures.append(now)
            return JSONResponse(
                status_code=401,
                content={'detail': 'Invalid or missing API key.'},
            )

        return await call_next(request)


class PipelineRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int = 5, window_seconds: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: DefaultDict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path != '/pipeline/run' or request.method.upper() != 'POST':
            return await call_next(request)

        client_ip = request.client.host if request.client else 'unknown'
        now = time.monotonic()
        recent_requests = self._requests[client_ip]

        while recent_requests and now - recent_requests[0] >= self.window_seconds:
            recent_requests.popleft()

        if len(recent_requests) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={'detail': 'Rate limit exceeded. Max 5 pipeline runs per minute per IP.'},
            )

        recent_requests.append(now)
        return await call_next(request)
