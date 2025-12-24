# Patches Directory

This directory contains patches and wrappers for external dependencies that don't have the exact interface our pipeline needs.

## Contents

### `fomm/fomm_wrapper.py`

**Purpose:** Provides a `main()` function interface for the first-order-model repository.

**Problem:** The original `first-order-model/demo.py` doesn't have a `main()` function - the code is directly in the `if __name__ == "__main__":` block. Our pipeline expects to call `fomm_demo.main(args)`.

**Solution:** This wrapper:
1. Imports all functions from the original `demo.py`
2. Provides a `main(args)` function that accepts our pipeline's args object
3. Handles both video-driven and audio-driven animation modes
4. Exports frames to the directory expected by our pipeline

**Installation:** The wrapper is automatically copied to `external_deps/first-order-model/` when you run:
```bash
make install-git-deps
```

**Manual Installation:**
```bash
cp patches/fomm/fomm_wrapper.py external_deps/first-order-model/fomm_wrapper.py
```

## Why Patches?

External dependencies (`external_deps/`) are cloned from third-party Git repositories and are excluded from our repository via `.gitignore`. This is correct - we shouldn't version control external code.

However, sometimes we need to adapt these external dependencies to work with our pipeline. Rather than forking the entire repository, we keep small patches/wrappers here and apply them during installation.

## Adding New Patches

If you need to patch another external dependency:

1. Create a subdirectory: `patches/<dependency-name>/`
2. Add your patch files there
3. Update `Makefile`'s `install-git-deps` target to copy the patch
4. Document it in this README

## Testing Patches

After installing external dependencies:

```bash
# Verify wrapper exists
ls external_deps/first-order-model/fomm_wrapper.py

# Test in Python
python -c "
import sys
sys.path.insert(0, 'external_deps/first-order-model')
from fomm_wrapper import main
print('✓ FOMM wrapper imports successfully')
"
```
