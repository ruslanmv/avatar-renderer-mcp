"""Avatar Renderer MCP - AI-Powered Talking Head Generation.

This package provides a production-ready avatar rendering system that generates
realistic talking head videos from a single image and audio file using state-of-the-art
deep learning models (FOMM, Diff2Lip, Wav2Lip, SadTalker).

The system includes:
- FastAPI REST API for job submission and monitoring
- MCP (Model Context Protocol) STDIO server for AI agent integration
- Celery-based distributed task processing
- Automatic GPU memory management and fallback mechanisms
- Support for Kubernetes deployments with KEDA autoscaling

Author: Ruslan Magana Vsevolodovna
Website: https://ruslanmv.com
License: Apache-2.0
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Ruslan Magana Vsevolodovna"
__email__ = "contact@ruslanmv.com"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2025 Ruslan Magana Vsevolodovna"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]
