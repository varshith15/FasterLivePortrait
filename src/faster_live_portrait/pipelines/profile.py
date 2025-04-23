# ===== profiler.py =====
import time, logging, torch
from contextlib import contextmanager

logger = logging.getLogger("FLP‑profiler")
logger.setLevel(logging.INFO)

@contextmanager
def prof(label: str):
    # yield
    """Time a code block on CPU + GPU and log ms."""
    start_cpu = time.perf_counter()
    start_gpu = torch.cuda.Event(enable_timing=True)
    end_gpu   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_gpu.record()
    yield                       # ---------- run the block ----------
    end_gpu.record()
    torch.cuda.synchronize()
    cpu_ms = (time.perf_counter() - start_cpu) * 1000
    gpu_ms = start_gpu.elapsed_time(end_gpu)
    logger.info(f"[PROFILE] {label:25s} | CPU {cpu_ms:6.2f} ms | GPU {gpu_ms:6.2f} ms")
