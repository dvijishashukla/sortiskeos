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

    async def dispatch(self, request: Request, call_next):
        if not self.api_key or request.url.path in self.exempt_paths:
            return await call_next(request)

        provided_key = request.headers.get('x-api-key', '').strip()
        if provided_key != self.api_key:
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
