"""
security.py – Session-token and CSRF-state helpers (standard library only).

These are intentionally small, dependency-free primitives. The FastAPI wiring
(dependencies, endpoints) lives in api.py so it can access Settings and the DB.

Author:
    Ruslan Magana Vsevolodovna
License:
    Apache-2.0
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional


def new_session_token() -> str:
    """Return a cryptographically-random, URL-safe opaque session token."""
    return secrets.token_urlsafe(32)


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
    if not authorization_header:
        return None
    parts = authorization_header.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


# --------------------------------------------------------------------------- #
# Stateless OAuth CSRF "state" — signed with the app secret so the callback can
# verify it without server-side storage (survives Space restarts).
# --------------------------------------------------------------------------- #

def make_oauth_state(secret: str) -> str:
    nonce = secrets.token_urlsafe(16)
    sig = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{nonce}.{sig}"


def verify_oauth_state(secret: str, state: Optional[str]) -> bool:
    if not state or "." not in state:
        return False
    nonce, sig = state.rsplit(".", 1)
    expected = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)
