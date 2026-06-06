"""
cos_utils.py – Optional Cloud Object Storage (S3 / IBM COS) upload helper.

Used by the Celery worker to publish rendered MP4s to an S3-compatible bucket.
The whole module is import- and call-safe when boto3 or credentials are absent:
callers should guard on ``settings.COS_BUCKET`` before invoking ``upload_to_cos``.

Credentials are read from the standard AWS environment variables / config that
boto3 understands (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.), so no secrets
are stored in this repo.

Author:
    Ruslan Magana Vsevolodovna
License:
    Apache-2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger("avatar-renderer.cos")


def upload_to_cos(local_path: str, settings: Any, key: str | None = None) -> str:
    """Upload ``local_path`` to the configured bucket and return its object URL.

    Args:
        local_path: Path to the file to upload.
        settings:   A ``Settings`` instance exposing ``COS_*`` attributes.
        key:        Optional object key. Defaults to the file name.

    Returns:
        A URL string pointing at the uploaded object.

    Raises:
        RuntimeError: if boto3 is unavailable or no bucket is configured.
    """
    bucket = getattr(settings, "COS_BUCKET", None)
    if not bucket:
        raise RuntimeError("COS_BUCKET is not configured; cannot upload.")

    try:
        import boto3  # lazy import — only needed when uploads are enabled
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("boto3 is not installed; cannot upload to object storage.") from exc

    local = Path(local_path)
    if not local.is_file():
        raise FileNotFoundError(f"Cannot upload missing file: {local}")

    object_key = key or local.name

    client_kwargs: dict[str, str] = {}
    if getattr(settings, "COS_ENDPOINT_URL", None):
        client_kwargs["endpoint_url"] = settings.COS_ENDPOINT_URL
    if getattr(settings, "COS_REGION", None):
        client_kwargs["region_name"] = settings.COS_REGION

    client = boto3.client("s3", **client_kwargs)
    client.upload_file(str(local), bucket, object_key, ExtraArgs={"ContentType": "video/mp4"})

    public_base = getattr(settings, "COS_PUBLIC_BASE_URL", None)
    if public_base:
        url = f"{public_base.rstrip('/')}/{quote(object_key)}"
    elif client_kwargs.get("endpoint_url"):
        url = f"{client_kwargs['endpoint_url'].rstrip('/')}/{bucket}/{quote(object_key)}"
    else:
        url = f"https://{bucket}.s3.amazonaws.com/{quote(object_key)}"

    logger.info("Uploaded %s -> %s", local, url)
    return url
