#!/usr/bin/env python3
"""
Installation Verification Script for Avatar Renderer MCP
=========================================================

This script verifies that the Avatar Renderer MCP package is correctly installed
and all dependencies are available. It performs comprehensive checks without
requiring model downloads or GPU access.

Usage:
    python scripts/verify_installation.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_header("Python Version Check")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}"

    if major == 3 and minor >= 11:
        print_success(f"Python {version_str} detected (requires >= 3.11)")
        return True
    else:
        print_error(f"Python {version_str} detected (requires >= 3.11)")
        return False


def check_core_imports() -> Tuple[bool, List[str]]:
    """Check if core dependencies can be imported."""
    print_header("Core Dependencies Check")

    dependencies = {
        "fastapi": "FastAPI web framework",
        "uvicorn": "ASGI server",
        "pydantic": "Data validation",
        "pydantic_settings": "Settings management",
        "torch": "PyTorch deep learning framework",
    }

    failed = []
    for module, description in dependencies.items():
        try:
            __import__(module)
            print_success(f"{module.ljust(20)} - {description}")
        except ImportError as e:
            print_error(f"{module.ljust(20)} - {description} (FAILED: {e})")
            failed.append(module)

    return len(failed) == 0, failed


def check_optional_imports() -> None:
    """Check optional dependencies."""
    print_header("Optional Dependencies Check")

    optional_deps = {
        "celery": "Distributed task queue",
        "kafka": "Kafka integration for progress tracking",
    }

    for module, description in optional_deps.items():
        try:
            __import__(module)
            print_success(f"{module.ljust(20)} - {description}")
        except ImportError:
            print_warning(f"{module.ljust(20)} - {description} (not installed, optional)")


def check_app_modules() -> Tuple[bool, List[str]]:
    """Check if app modules can be imported."""
    print_header("Application Modules Check")

    # Add parent directory to path to import app modules
    sys.path.insert(0, str(Path(__file__).parent.parent))

    modules = {
        "app.settings": "Settings configuration",
        "app.api": "FastAPI REST API",
        "app.pipeline": "Rendering pipeline",
        "app.mcp_server": "MCP STDIO server",
    }

    failed = []
    for module, description in modules.items():
        try:
            __import__(module)
            print_success(f"{module.ljust(20)} - {description}")
        except ImportError as e:
            print_error(f"{module.ljust(20)} - {description} (FAILED: {e})")
            failed.append(module)

    return len(failed) == 0, failed


def check_gpu_availability() -> None:
    """Check GPU availability."""
    print_header("GPU Availability Check")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print_success(f"CUDA available with {gpu_count} GPU(s)")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print_warning("CUDA not available (CPU mode will be used)")
            print("  - High-quality mode requires GPU")
            print("  - Real-time mode can work on CPU (slower)")
    except Exception as e:
        print_error(f"Failed to check GPU: {e}")


def check_file_structure() -> Tuple[bool, List[str]]:
    """Check if required files and directories exist."""
    print_header("File Structure Check")

    root = Path(__file__).parent.parent
    required_paths = {
        "app/": "Application directory",
        "app/api.py": "REST API module",
        "app/mcp_server.py": "MCP server module",
        "app/pipeline.py": "Rendering pipeline",
        "app/settings.py": "Configuration settings",
        "mcp-tool.json": "MCP tool definition",
        "docs/QUALITY_MODES.md": "Quality modes documentation",
    }

    missing = []
    for path, description in required_paths.items():
        full_path = root / path
        if full_path.exists():
            print_success(f"{path.ljust(30)} - {description}")
        else:
            print_error(f"{path.ljust(30)} - {description} (MISSING)")
            missing.append(path)

    return len(missing) == 0, missing


def check_quality_modes() -> bool:
    """Check if quality modes are properly configured."""
    print_header("Quality Modes Configuration Check")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.pipeline import render_pipeline
        import inspect

        # Check function signature
        sig = inspect.signature(render_pipeline)
        params = list(sig.parameters.keys())

        required_params = ['face_image', 'audio', 'out_path', 'quality_mode']
        for param in required_params:
            if param in params:
                print_success(f"Parameter '{param}' is available")
            else:
                print_error(f"Parameter '{param}' is MISSING")
                return False

        # Check default value for quality_mode
        if sig.parameters['quality_mode'].default == 'auto':
            print_success("Default quality_mode is 'auto'")
        else:
            print_warning(f"Default quality_mode is '{sig.parameters['quality_mode'].default}' (expected 'auto')")

        return True
    except Exception as e:
        print_error(f"Failed to verify quality modes: {e}")
        return False


def check_api_endpoints() -> bool:
    """Check if API endpoints are properly configured."""
    print_header("API Endpoints Check")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.api import app

        routes = [route.path for route in app.routes]

        expected_endpoints = [
            "/render",
            "/status/{job_id}",
            "/health/live",
            "/health/ready",
            "/avatars",
        ]

        for endpoint in expected_endpoints:
            if endpoint in routes:
                print_success(f"Endpoint {endpoint}")
            else:
                print_error(f"Endpoint {endpoint} (MISSING)")
                return False

        return True
    except Exception as e:
        print_error(f"Failed to verify API endpoints: {e}")
        return False


def main():
    """Run all verification checks."""
    print(f"\n{BLUE}{'*' * 70}{RESET}")
    print(f"{BLUE}{'Avatar Renderer MCP - Installation Verification'.center(70)}{RESET}")
    print(f"{BLUE}{'*' * 70}{RESET}")

    all_passed = True

    # Run all checks
    checks = [
        ("Python Version", check_python_version()),
        ("Core Imports", check_core_imports()[0]),
        ("App Modules", check_app_modules()[0]),
        ("File Structure", check_file_structure()[0]),
        ("Quality Modes", check_quality_modes()),
        ("API Endpoints", check_api_endpoints()),
    ]

    # Optional checks (don't fail on these)
    check_optional_imports()
    check_gpu_availability()

    # Summary
    print_header("Verification Summary")

    for check_name, passed in checks:
        if passed:
            print_success(f"{check_name.ljust(30)} PASSED")
        else:
            print_error(f"{check_name.ljust(30)} FAILED")
            all_passed = False

    print("\n" + "=" * 70 + "\n")

    if all_passed:
        print(f"{GREEN}{'✓ All checks passed! Installation is verified.'.center(70)}{RESET}")
        print(f"\n{GREEN}The Avatar Renderer MCP is ready to use!{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Download models: make download-models")
        print(f"  2. Start REST API: make run")
        print(f"  3. Start MCP server: make run-stdio")
        print(f"  4. Check health: curl http://localhost:8080/avatars")
        return 0
    else:
        print(f"{RED}{'✗ Some checks failed. Please fix the issues above.'.center(70)}{RESET}")
        print(f"\n{RED}Installation verification failed!{RESET}")
        print(f"\nTroubleshooting:")
        print(f"  1. Install dependencies: pip install -e '.[dev]'")
        print(f"  2. Check Python version: python --version")
        print(f"  3. Verify file structure is intact")
        return 1


if __name__ == "__main__":
    sys.exit(main())
