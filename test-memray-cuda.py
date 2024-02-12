"""
https://github.com/bloomberg/memray/issues/547#issuecomment-1939775423
"""

import time
import memray
import torch.cuda

with memray.Tracker("cudatest.bin"):
    print("Starting tracking")
    time.sleep(1)
    for _ in range(10_000):
        torch.cuda.is_available()
        torch.cuda.mem_get_info()
    print("Stopping tracking")
    time.sleep(1)
