import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule


def trace_test():
    # 1. Setup hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 2. Define a simple but heavy operation
    model = nn.Sequential(nn.Linear(8192, 8192), nn.ReLU(), nn.Linear(8192, 8192)).to(
        device
    )

    inputs = torch.randn(1024, 8192, device=device)

    # 3. Initialize Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # Wait 2 steps (ignored), Warmup 2 steps (CUPTI hooks in), Active 2 steps (Recorded)
        schedule=schedule(wait=2, warmup=2, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/test_trace"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for i in range(10):
            # Label this block so you can find it in Perfetto easily
            with record_function("## STEP_ITERATION ##"):
                output = model(inputs)
                loss = output.sum()
                loss.backward()

                # Force synchronization inside the active window to ensure
                # kernels are "flushed" into the trace.
                # torch.cuda.synchronize()

            # Move to next step in the schedule
            prof.step()
            print(f"Step {i} completed")


if __name__ == "__main__":
    trace_test()
