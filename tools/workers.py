"""Pool nhiều tiến trình, mỗi tiến trình 1 nhân: trong `top` mỗi PID không vượt quá 100% CPU.

Tải CPU chỉ chỉnh bằng số tiến trình. Không để thư viện tự mở luồng theo số nhân của máy
(PyTorch, ONNX Runtime, BLAS, ffmpeg đều mặc định làm vậy).
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "ORT_NUM_THREADS")


def single_thread():
    """Gọi đầu mỗi tiến trình worker, trước khi import thư viện tính toán."""
    for k in THREAD_VARS:
        os.environ[k] = "1"
    try:  # BLAS đã được nạp trước (vd. start_method=fork sau khi cha import numpy): giới hạn lúc đang chạy
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):  # RuntimeError: interop threads đã được đặt
        pass


def _init(fn, args):
    single_thread()
    if fn:
        fn(*args)


def pool(n, start_method, init=None, initargs=()):
    """ProcessPoolExecutor với n tiến trình đơn luồng. `init(*initargs)` chạy một lần mỗi tiến trình (load model ở đây).
    start_method: 'spawn' (an toàn với CUDA, mặc định trong config) hoặc 'fork' (khởi động nhanh, không dùng khi có CUDA)."""
    return ProcessPoolExecutor(n, mp_context=mp.get_context(start_method), initializer=_init, initargs=(init, initargs))
