"""Keep Windows awake while training runs (SetThreadExecutionState). Exits when the file v3\\STOP_CAFFEINE exists or after 14 hours."""
import ctypes
import time
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
stop = Path(__file__).parent / "STOP_CAFFEINE"
t0 = time.time()
while not stop.exists() and time.time() - t0 < 14 * 3600:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    time.sleep(30)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
