"""Fix the macOS duplicate-libomp crash: point faiss at torch's OpenMP runtime.

On macOS, the pip wheels of torch and faiss-cpu each bundle their own
``libomp.dylib``. Two OpenMP runtimes in one process abort with
``OMP: Error #15`` or segfault inside worker barriers (SIGSEGV in
``__kmp_suspend_64``) — whichever library starts a parallel region second.
The fix is a single shared runtime: replace faiss's bundled copy with a
symlink to torch's. Idempotent; the original is kept as ``libomp.dylib.orig``.

Run once after (re)installing requirements on macOS:
    python scripts/fix_macos_libomp.py

Linux (incl. the HF Spaces Docker image) is unaffected and the script is a no-op.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
    if platform.system() != "Darwin":
        print("not macOS — nothing to do")
        return
    site = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    faiss_omp = site / "faiss" / ".dylibs" / "libomp.dylib"
    torch_omp = site / "torch" / "lib" / "libomp.dylib"
    if not torch_omp.exists() or not faiss_omp.parent.exists():
        raise SystemExit(f"expected wheels not found under {site} — install requirements first")
    if faiss_omp.is_symlink():
        print(f"already fixed: {faiss_omp} -> {faiss_omp.readlink()}")
        return
    backup = faiss_omp.with_suffix(".dylib.orig")
    faiss_omp.rename(backup)
    faiss_omp.symlink_to(Path("..") / ".." / "torch" / "lib" / "libomp.dylib")
    print(f"fixed: {faiss_omp} now points at torch's libomp (original kept as {backup.name})")


if __name__ == "__main__":
    main()
