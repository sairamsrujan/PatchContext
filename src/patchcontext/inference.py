"""All native-model inference runs on one persistent worker thread.

Streamlit executes every script rerun on a fresh short-lived thread. Running
torch/OpenMP parallel regions from many transient threads intermittently
segfaults on x86-64 Linux (the OpenMP runtime binds thread-team state to the
creating thread; observed as ``Uncaught signal: 11`` right after an encode
completes). Funnelling every ``encode``/``predict`` call through a single
long-lived thread keeps the OpenMP runtime's world stable — and the models are
not thread-safe anyway, so serializing inference is correct regardless.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")

_WORKER_NAME = "model-inference"
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=_WORKER_NAME)


def run_inference(fn: Callable[[], T]) -> T:
    """Execute ``fn`` on the persistent inference thread and return its result.

    Calls made from the inference thread itself run directly (no deadlock on
    the single-worker executor).
    """
    if threading.current_thread().name.startswith(_WORKER_NAME):
        return fn()
    return _worker.submit(fn).result()
