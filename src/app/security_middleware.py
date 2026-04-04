"""Production-oriented middleware: API key, rate limit, security headers."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocket

from ..config.settings import (
    API_SECRET_KEY,
    IS_PRODUCTION,
    RATE_LIMIT_PER_MINUTE,
    TRUST_PROXY_HEADERS,
)

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({"/health"})


def client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    if request.client:
        return request.client.host
    return "unknown"


def extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    xkey = request.headers.get("x-api-key")
    if xkey:
        return xkey.strip() or None
    return None


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline headers for HTTPS deployments and XSS/mime sniff mitigation."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response: Response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request_id)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if IS_PRODUCTION:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains",
            )
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require API_SECRET_KEY / API_KEY for all routes except public paths and CORS preflight."""

    async def dispatch(self, request: Request, call_next):
        if not API_SECRET_KEY:
            return await call_next(request)
        path = request.url.path
        if request.method == "OPTIONS":
            return await call_next(request)
        if path in PUBLIC_PATHS:
            return await call_next(request)
        token = extract_bearer_token(request)
        if token != API_SECRET_KEY:
            logger.warning("Unauthorized request to %s from %s", path, client_ip(request))
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "detail": "Send Authorization: Bearer <API_SECRET_KEY> or X-API-Key header.",
                },
            )
        return await call_next(request)


class _SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_sec: float = 60.0):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_sec
        hits = self._hits[key]
        hits[:] = [t for t in hits if t > cutoff]
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


_limiter: _SlidingWindowLimiter | None = None


def _get_limiter() -> _SlidingWindowLimiter | None:
    global _limiter
    if RATE_LIMIT_PER_MINUTE <= 0:
        return None
    if _limiter is None:
        _limiter = _SlidingWindowLimiter(RATE_LIMIT_PER_MINUTE, window_sec=60.0)
    return _limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-IP sliding window (HTTP only). /health is excluded."""

    async def dispatch(self, request: Request, call_next):
        limiter = _get_limiter()
        if limiter is None:
            return await call_next(request)
        path = request.url.path
        if path == "/health" or request.method == "OPTIONS":
            return await call_next(request)
        ip = client_ip(request)
        if not limiter.allow(ip):
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "detail": "Too many requests. Try again shortly."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


def websocket_api_key_authorized(websocket: WebSocket) -> bool:
    """Bearer, X-API-Key, or ?api_key= (simple clients). Must match API_SECRET_KEY when set."""
    if not API_SECRET_KEY:
        return True
    auth = websocket.headers.get("authorization") or ""
    xkey = websocket.headers.get("x-api-key") or ""
    q = websocket.query_params.get("api_key") or ""
    token = None
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    elif xkey:
        token = xkey.strip()
    elif q:
        token = q.strip()
    return token == API_SECRET_KEY
