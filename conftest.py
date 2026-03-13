# conftest.py — loaded by pytest before any test module.
# sets environment variables that must exist before c-extension modules
# (torch, faiss) are imported. this is the correct place for such setup
# because pytest loads conftest.py first, before collecting test files.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")