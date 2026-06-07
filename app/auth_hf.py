"""
auth_hf.py – Hugging Face OAuth (OpenID Connect) client helpers.

Implements the three OAuth steps as small, testable functions:
    1. build_authorize_url() – where to redirect the browser
    2. exchange_code_for_token() – swap the ?code= for an access token
    3. fetch_user_info() – read the signed-in user's profile

``requests`` is imported lazily inside the functions so importing this module
(e.g. during tests, or when AUTH is disabled) pulls in no extra dependency.

Register an OAuth app / connected app at:
    https://huggingface.co/settings/applications
and (for Spaces) see https://huggingface.co/docs/hub/spaces-oauth

Author:
    Ruslan Magana Vsevolodovna
License:
    Apache-2.0
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode


class HuggingFaceAuthError(RuntimeError):
    """Raised when an OAuth step fails."""


def build_authorize_url(
    *,
    authorize_url: str,
    client_id: str,
    redirect_uri: str,
    scopes: str,
    state: str,
) -> str:
    """Build the Hugging Face authorization URL to redirect the browser to."""
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scopes,
            "state": state,
        }
    )
    return f"{authorize_url}?{query}"


def exchange_code_for_token(
    *,
    token_url: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Exchange an authorization code for an access token."""
    import requests  # lazy

    try:
        resp = requests.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:  # network error
        raise HuggingFaceAuthError(f"token request failed: {exc}") from exc

    if resp.status_code != 200:
        raise HuggingFaceAuthError(
            f"token exchange returned {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    if "access_token" not in data:
        raise HuggingFaceAuthError(f"no access_token in response: {data}")
    return data


def fetch_user_info(*, userinfo_url: str, access_token: str, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch the OpenID Connect userinfo for the signed-in user."""
    import requests  # lazy

    try:
        resp = requests.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise HuggingFaceAuthError(f"userinfo request failed: {exc}") from exc

    if resp.status_code != 200:
        raise HuggingFaceAuthError(
            f"userinfo returned {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()


def normalize_profile(userinfo: dict[str, Any]) -> dict[str, Any]:
    """Map HF/OIDC userinfo fields to our stable internal shape."""
    return {
        "hf_user_id": str(userinfo.get("sub") or userinfo.get("id") or ""),
        "hf_username": userinfo.get("preferred_username") or userinfo.get("name"),
        "hf_email": userinfo.get("email"),
        "picture": userinfo.get("picture"),
    }
