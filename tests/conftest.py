"""Test-session setup.

torch MUST load before faiss anywhere in the process: on macOS both bundle
libomp.dylib and faiss-first ordering segfaults CPU inference at an OpenMP
barrier. conftest.py runs before any test module, making the ordering
unconditional. (Serving no longer imports faiss — see retrieve/retriever.py —
but the offline index build and some tests still do.)
"""

import torch  # noqa: F401  (must precede any faiss import)
