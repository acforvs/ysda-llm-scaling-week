import time
import triton
import triton.language as tl
import torch
import torch.nn as nn
from torch.library import triton_op, wrap_triton
from deep_gemm.jit_kernels.gemm import gemm_fp8_fp8_bf16_nt

import statistics
from execute_util import text, link, image
from triton_util import triton_tanh


def main():
    text("# Speeding up training with FP8 and Triton")
    text("I'm Vlad, and I work on speeding up large LLM training without quality loss.")

    how_to_benchmark()
    our_goal()
    gelu_cuda()
    about_triton()
    gelu_speedup()
    profiling()
    mixed_precision()
    fp8()
    speeding_up_mlp()
    resources()


def how_to_benchmark():
    text("## Speed measurement")
    text("Before speeding something up, we need to learn how to estimate how much time it takes in the first place")
    text("Let's start by checking how long a simple matrix multiplication takes:")

    perf_t = bench_matmuls("perf_counter")
    cpu_scheduling()
    cuda_events_t = bench_matmuls("cuda.events")
    triton_t = bench_matmuls("triton.testing") # @inspect perf_t, @inspect cuda_events_t, @inspect triton_t


def generate_data(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two FP32 tensors on the GPU."""
    a = torch.randn(m, n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, k, dtype=torch.float32, device="cuda")
    return a, b


def matmul(a: torch.Tensor, b: torch.Tensor):
    """A simple function for matrix multiplication."""
    return a @ b


def run_bench_perf_counter(m: int, n: int, k: int, num_iters: int = 100) -> list[float]:
    a, b = generate_data(m, n, k)
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        matmul(a, b)
        end = time.perf_counter()
        times.append((end - start) / 1e-3) # @inspect times
    return times


def run_bench_cuda_events(m: int, n: int, k: int, num_iters: int = 100) -> list[float]:
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    a, b = generate_data(m, n, k)
    for i in range(num_iters):
        start_events[i].record()
        matmul(a, b)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times # @inspect times


def run_bench_triton_testing(m: int, n: int, k: int, num_iters: int = 100, num_warmups: int = 10) -> float:
    a, b = generate_data(m, n, k)
    times = triton.testing.do_bench(lambda: matmul(a, b), warmup=num_warmups, rep=num_iters, return_mode='all')
    return times # @inspect times


def bench_matmuls(bench_mode: str):
    if bench_mode == "perf_counter":
        func = run_bench_perf_counter
    elif bench_mode == "cuda.events":
        func = run_bench_cuda_events
    elif bench_mode == "triton.testing":
        func = run_bench_triton_testing
    else:
        raise ValueError(f"Unknown bench_mode: {bench_mode}")

    # Launch the benchmark for shapes (16, 32) @ (32, 16)
    small_shapes_t = func(16, 32, 16)

    # Launch the benchmark for shapes (16384, 32768) @ (32768, 16384)
    large_shapes_t = func(16 * 1024, 32 * 1024, 16 * 1024)

    mean_small_shapes_t = statistics.mean(small_shapes_t)
    mean_large_shapes_t = statistics.mean(large_shapes_t) # @inspect mean_small_shapes_t, @inspect mean_large_shapes_t 
    return mean_small_shapes_t, mean_large_shapes_t


def cpu_scheduling():
    text("CPU code schedules GPU-kernels. Kernel is an operation that runs on a GPU.")
    text("CPU code can run forward without waiting for the GPU to finish its job.")
    image("var/profile_image.png", width=800)
    text("The benchmark that used time.time() was not actually measuring the correct thing! We estimated the CPU overhead for the kernel launch rather than its actual run time on a GPU.")


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


class SlowMLP(torch.nn.Module):
    def __init__(self, dim: int = 16384):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim * 4, dtype=torch.float32)
        self.fc2 = torch.nn.Linear(dim * 4, dim, dtype=torch.float32)
        self.gelu_kernel = gelu

    def get_stat(self, x):
        """Oftentimes the code will log/print intermediate results."""
        print(f"output_1/max: {x.detach().max().item()}")
        print(f"output_1/min: {x.detach().min().item()}")

    def up_proj(self, x):
        return self.fc1(x)

    def down_proj(self, x):
        return self.fc2(x)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.gelu_kernel(x)
        self.get_stat(x)
        x = self.down_proj(x)
        return x


def bench_mlp(model_cls, num_iters: int = 100, num_warmups: int = 10):
    from io import StringIO
    from contextlib import redirect_stdout

    buffer = StringIO()

    # The MLP runtime
    with redirect_stdout(buffer):
        model = model_cls().to("cuda")
        x = torch.randn(16384, 16384, dtype=torch.float32, device="cuda")
        mlp_time = triton.testing.do_bench(lambda: model(x), rep=num_iters, warmup=num_warmups) # @inspect mlp_time
    return mlp_time


def our_goal():
    text("## The goal for today is to speed up the MLP block")
    text("First, let's find out how long it takes to run the MLP block.")

    bench_mlp(SlowMLP)

    text("Now, let's benchmark the gelu kernel:")
    inter = torch.randn(16384, 16384 * 4, dtype=torch.float32, device="cuda")
    gelu_time = triton.testing.do_bench(lambda: gelu(inter)) # @inspect gelu_time

    text("Is this expected or not?")
    text("To understand that, we have to get to know how the accelerator works.")
    cuda_intro()

    text("### Arithmetic intensity: #FLOPs / #bytes")
    text("- If this value is high enough, the operation is compute-bound.")
    text("- Otherwise, the operation is memory-bound, and to estimate the runtime, we should estimate the time it takes to transfer the data in&out. ")
    text("- With moder GPUs, we need at least 200 FLOPs/byte to be compute-bound.")

    text("The gelu processes N elements in FP32, so we get arithmetic intensity of 9N / 8N = 1.125.")

    text(f"**We need to calculate the read-time**. There are 16384x16384x4 FP32 elements => 16384x16384x4x4 bytes on read and write. With ideal bandwidth of 3.35 TB/s, we would get the runtime of 16384x16384x4x4x2/3.35e12 ~ {(t := 16384**2 * 4**2 * 2 / 3.35e12):.5f} sec = {t * 1000:.3f}ms")
    text(f"Instead, we get {gelu_time:.3f} ms! That is {round(gelu_time / t / 1000, 2)} times slower.")


def cuda_intro():
    text("## CUDA")
    text("SIMT: Single Instruction Multiple Threads.")
    text("~ independent compute units called Streaming Multiprocessors (SMs).")
    text("Each SM has its own cache, which is 'connected' to the main GPU memory.")
    image("var/h100_highlevel.png", width=800)

    text("### Compute")
    text("H100:")
    text("  - A collection of Streaming Multiprocessor-s (SMs)")
    text("  - Tensor Core for matrix multiplication;")
    text("  - Vector arithmetic unit for arithmetic operations;")

    text("A single SM holds 4 sub-partitions. Inside each of them:")
    text("  - Warp Scheduler.")
    text("  - CUDA Cores: 32 fp32 cores, they execute one instruction per cycle.")
    text("  - For example, H100 can provide up to 990 TFLOPs/s in BF16. 990e12 / 132 (SM) / 4 (subpartitions) / 1.76e9 (Hz) ~ 1065 fp32 FLOPs/cycle => ~8x8x8 matrix multiplication.")

    image("var/light-gh100-sm.svg", width=800)

    text("Threads are grouped into thread blocks.")
    text("Grid is a collection of thread blocks.")
    image("var/thread_blocks.png", width=600)

    text("Thread blocks can be scheduled differently on different devices.")
    image("var/light-wave-scheduling.svg", width=600)

    text("Thread blocks are scheduled in waves.")

    text("**Important:** if two threads in the same Warp (collection of 32 threads) are assigned with different work, the Warp executes both instructions.")

    image("https://jax-ml.github.io/scaling-book/assets/gpu/warp-divergence.png", width=800)

    text("### Memory")
    text("Registers -> SMEM -> L2 Cache -> HBM")
    text("**Registers:** 16384 32-bit words / partition.")
    text("**SMEM (shared memory):** L1 cache, 228KiB, 31TB/s cache per SM;")
    text("**L2 Cache:** all SMs share 50MB L2 cache. We don't know the exact bandwidth, but it's ~12TB/s.")
    text("**HBM:** the main GPU memory, 80GB for H100, it is reported to have a bandwidth of 3.35TB/s.")
    image("var/gpu_diff.png", width=800)


def get_gelu_cuda():
    import os
    from torch.utils.cpp_extension import load_inline

    # credits to: https://stanford-cs336.github.io/spring2025/
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"
    cuda_gelu_src = """
    #include <math.h>
    #include <torch/extension.h>
    #include <c10/cuda/CUDAException.h>

    __global__ void gelu_kernel(float* in, float* out, int num_elements) {
        // Get the index into the tensor
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < num_elements) {
            out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));
        }
    }

    inline unsigned int cdiv(unsigned int a, unsigned int b) {
        return (a + b - 1) / b;
    }

    torch::Tensor gelu(torch::Tensor x) {
        TORCH_CHECK(x.device().is_cuda());
        TORCH_CHECK(x.is_contiguous());

        torch::Tensor y = torch::empty_like(x);

        int num_elements = x.numel();
        int block_size = 1024;
        int num_blocks = cdiv(num_elements, block_size);

        gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return y;
    }
    """
    os.makedirs("var/cuda_gelu", exist_ok=True)
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="var/cuda_gelu",
    )
    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu


def gelu_cuda():
    cuda_gelu_kernel = get_gelu_cuda()
    x = torch.randn(16384, 16384 * 4, dtype=torch.float32, device="cuda")
    gelu_time = triton.testing.do_bench(lambda: gelu(x)) # @inspect gelu_time
    cuda_gelu_time = triton.testing.do_bench(lambda: cuda_gelu_kernel(x)) # @inspect cuda_gelu_time


def about_triton():
    text("## Triton")
    text("OpenAI, 2021")
    link(title="https://openai.com/research/triton", url="https://openai.com/research/triton")
    triton_intro()
    triton_add()
    triton_and_torch()
    triton_and_cpu()
    triton_gelu()


def triton_intro():
    text("- Code in Python;")
    text("- Don't think on a thread-level (which is both good and bad) and don't control the SMEM etc.")

    text("What does Triton offer?", verbatim=True)
    text("                                             CUDA      Triton", verbatim=True)
    text("- Memory coalescing (transfer from DRAM)     manual    automatic", verbatim=True)
    text("- Shared memory management                   manual    automatic", verbatim=True)
    text("- Scheduling within SMs                      manual    automatic", verbatim=True)
    text("- Scheduling across SMs                      manual    manual", verbatim=True)


@triton.jit
def _add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block ID
    pid = tl.program_id(0)
    # Past blocks have already processed pid * BLOCK_SIZE elements, so we need to skip them
    block_start = pid * BLOCK_SIZE
    # Calc the read offsets. The offsets are the BLOCK_SIZE elements, starting from block_start
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # To prevent reading out-of-bounds, create a mask
    mask = offsets < n_elements
    # HBM-load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Run the computation
    output = x + y
    # Write the result back using the same offsets & mask
    tl.store(output_ptr + offsets, output, mask=mask)


# A regular function to call the Triton kernel
def add_vectors(x: torch.Tensor, y: torch.Tensor):
    # Assertions before launching the kernel
    assert x.is_contiguous() and y.is_contiguous(), "Expected x and y to be contiguous"
    assert x.device == y.device, f"Expected x and y to be on the same device, found: {x.device} != {y.device}"
    assert x.dtype == y.dtype, f"Expected x and y to have the same dtype, found: {x.dtype} != {y.dtype}"
    assert x.shape == y.shape, f"Expected x and y to have the same shape, found: {x.shape} != {y.shape}"

    output = torch.empty_like(x.view(-1))

    # Create a grid of threadblocks, with each block handling dedicated BLOCK_SIZE elements
    # We need to have round_up((number of elements) / BLOCK_SIZE) blocks to cover all the elements
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    # Launch the kernel and pass the BLOCK_SIZE=32
    _add_kernel[grid](x.view(-1), y.view(-1), output, x.numel(), BLOCK_SIZE=32)

    return output.view(x.shape)


def triton_add():
    x = torch.randn(16, 100, device="cuda")
    y = torch.randn(16, 100, device="cuda")
    quality_ok = torch.allclose(add_vectors(x, y), x + y) # @inspect quality_ok
    kernel_time = triton.testing.do_bench(lambda: add_vectors(x, y))
    torch_time = triton.testing.do_bench(lambda: x + y) # @inspect kernel_time, @inspect torch_time


def triton_and_torch():
    text("Usually, operations are auto differentiable in PyTorch. What about Triton?")
    triton_bad_backward()
    triton_impl_backward()


def triton_bad_backward():
    try:
        x = torch.randn(16, 100, device="cuda").requires_grad_(True)  # A flag indicating that the tensor requires gradient computation
        y = torch.randn(16, 100, device="cuda").requires_grad_(True)
        out = add_vectors(x, y)
        # Try to launch .backward() pass through the graph
        (out**2).mean().backward()
    except Exception as e:
        error = str(e) # @inspect error


def test_bwd(x, y, mode: str = "triton"):
    a = x.detach().requires_grad_(True)
    b = y.detach().requires_grad_(True)
    if mode == "triton":
        s = torch.ops.llm_scaling_week.add_vectors(a, b)
    if mode == "torch":
        s = a + b
    (s**2).mean().backward()
    return a.grad, b.grad


# Use the decorator. We must provide type-hints and mark which arguments will be changed (in-place) inside the function.
@triton_op("llm_scaling_week::add_vectors", mutates_args={})
def wrapped_add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.device == y.device, f"Expected x and y to be on the same device, found: {x.device} != {y.device}"
    assert x.dtype == y.dtype, f"Expected x and y to have the same dtype, found: {x.dtype} != {y.dtype}"
    assert x.shape == y.shape, f"Expected x and y to have the same shape, found: {x.shape} != {y.shape}"

    output = torch.empty_like(x.view(-1))
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    wrap_triton(_add_kernel)[grid](x.view(-1), y.view(-1), output, x.numel(), BLOCK_SIZE=32)

    return output.view(x.shape)


def triton_impl_backward():
    from torch.library import triton_op, wrap_triton

    def wrapped_add_vectors_bwd(ctx, grad_output):
        """Get dL/dOut, return dL/dX and dL/dY."""
        return grad_output, grad_output

    wrapped_add_vectors.register_autograd(wrapped_add_vectors_bwd)   # Register the autograd function

    # Check the gradient of our kernel against the original Torch function
    x = torch.randn(16, 100, device="cuda")
    y = torch.randn(16, 100, device="cuda")
    triton_grads = test_bwd(x, y, "triton")
    torch_grads = test_bwd(x, y, "torch")
    for trg, tog in zip(triton_grads, torch_grads):
        ok = torch.allclose(trg, tog, atol=1e-6) # @inspect ok

    triton_setup_context()


@triton.jit
def _sin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)


@triton_op("llm_scaling_week::mysin", mutates_args={})
def mysin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    wrap_triton(_sin_kernel)[(n_elements,)](x, out, x.numel(), BLOCK_SIZE=32)
    return out


def triton_setup_context():
    # This funcation is called by the end of the forward pass, we can store the context we need for the backward pass if we need to
    # Inside this function, there's an access to both the inputs & outputs of the forward pass
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    # This function is called by the end of the backward pass and has an access to all the context we have previously stored
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * x.cos()

    mysin.register_autograd(backward, setup_context=setup_context)


def triton_and_cpu():
    # Try to launch the addition on a CPU
    x, y = torch.randn(16, 100), torch.randn(16, 100)
    try:
        torch.ops.llm_scaling_week.add_vectors(x, y)
    except Exception as e:
        error = str(e) # @inspect error

    # Register a CPU fallback
    @wrapped_add_vectors.register_kernel("cpu")
    def _(a, b):
        return a + b

    out_device = torch.ops.llm_scaling_week.add_vectors(x, y).device # @inspect out_device


@triton.jit
def _gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Similarly to the add-kernel, read the dedicated BLOCK_SIZE elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load and UPCAST to FP32
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    # Run all the required operations at once
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    output = 0.5 * x * (1 + triton_tanh(a))
    # Downcast back to the expected output dtype
    output = output.to(output_ptr.dtype.element_ty)
    # Store the result back
    tl.store(output_ptr + offsets, output, mask=mask)


@triton_op("llm_scaling_week::gelu_triton", mutates_args={})
def gelu_triton(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x.view(-1))
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    wrap_triton(_gelu_kernel)[grid](x.view(-1), out, x.numel(), BLOCK_SIZE=32)
    return out.view(x.shape)


def triton_gelu():
    text("Now, with everything we know about Triton, let's implement the gelu kernel")

    x = torch.randn(16384, 16384 * 4, dtype=torch.float32, device="cuda")
    gelu_time = triton.testing.do_bench(lambda: torch.ops.llm_scaling_week.gelu_triton(x)) # @inspect gelu_time
    torch_time = triton.testing.do_bench(lambda: gelu(x)) # @inspect torch_time

    # Let's check that the results are identical to those of the torch function
    ok = torch.allclose(gelu(x), torch.ops.llm_scaling_week.gelu_triton(x), atol=1e-6) # @inspect ok

    text("However, it's still slow...")
    triton_autotune()


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE})
        for BLOCK_SIZE in (32, 64, 128, 256, 512, 1024)
    ],
    key=["n_elements"]
)
@triton.jit
def _gelu_kernel_autotune(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    output = 0.5 * x * (1 + triton_tanh(a))
    tl.store(output_ptr + offsets, output, mask=mask)


@triton_op("llm_scaling_week::gelu_triton_auto", mutates_args={})
def gelu_triton_auto(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x.view(-1))
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    # Do NOT pass the BLOCK_SIZE explicitly here
    wrap_triton(_gelu_kernel_autotune)[grid](x.view(-1), out, x.numel())
    return out.view(x.shape)


def triton_autotune():
    text("Let's tune the BLOCK_SIZE value automatically")

    x = torch.randn(16384, 16384 * 4, dtype=torch.float32, device="cuda")
    gelu_time = triton.testing.do_bench(lambda: torch.ops.llm_scaling_week.gelu_triton_auto(x), 1000) # @inspect gelu_time
    torch_time = triton.testing.do_bench(lambda: gelu(x), 1000) # @inspect torch_time


def gelu_speedup():
    stat = {}
    x = torch.randn(16384, 16384 * 4, dtype=torch.float32, device="cuda")
    for func, name in zip(
        (gelu, torch.ops.llm_scaling_week.gelu_triton, torch.ops.llm_scaling_week.gelu_triton_auto),
        ("baseline", "triton_gelu", "triton_gelu_autotune")
    ):
        stat[name] = triton.testing.do_bench(lambda: func(x)) # @inspect stat

    # Benchmarking the built-in function as well
    stat["torch_gelu"] = triton.testing.do_bench(lambda: torch.nn.functional.gelu(x, approximate="tanh")) # @inspect stat

    class MLPNewGelu(SlowMLP):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gelu_kernel = torch.ops.llm_scaling_week.gelu_triton_auto

    old_time = bench_mlp(SlowMLP) # @inspect old_time
    new_time = bench_mlp(MLPNewGelu) # @inspect new_time


def run_profile(func, num_warmup: int = 10):
    import os

    def trace_handler(prof):
        os.makedirs("./profile", exist_ok=True)
        prof.export_chrome_trace(f"./profile/chrome_trace_{func.__name__}.json")

    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True, # Store the CPU stack trace
        record_shapes=True, # Record the shapes
        on_trace_ready=trace_handler, # What we should do once the trace is ready
    ) as prof:
        func()

    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=80)


def profiling():
    text("## Profiling")
    text("We learnt how to measure the time it takes for an operation to run. However, we do not know where the time is spent. To see which kernels are launched when they are scheduled, we can profile the code.")

    run_profile(
        lambda: bench_mlp(SlowMLP, 1, 1)
    )

    text("We can open the stored trace in https://ui.perfetto.dev")
    image("var/mlp_profile.png", width=800)

    text("From this trace, we can see why the gelu kernel used to take so much time.")
    image("var/gelu_gpu_stream.png", width=800)

    text("We can identify another problem: CPU-bound computations.")
    image("var/cpu_bound_mlp.png", width=800)

    text("Let's see how the CPU -> GPU scheduling works")
    image("var/profile_image.png", width=800)
    text("Imagine we want to print out the result of the gelu kernel. Then we'd need to wait for the GPU result on a CPU. That's where the synchronize comes from.")
    image("var/profile_cpu_blocked_image.png", width=800)

    text("We can remove the CPU overhead:")

    class MLPNewGelu(SlowMLP):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gelu_kernel = torch.ops.llm_scaling_week.gelu_triton_auto

    class MLPNoPrint(MLPNewGelu):
        def get_stat(self, x):
            """This function can be non-blocking, but we'll just leave it empty for simplicity."""
            pass

    no_print_time = bench_mlp(MLPNoPrint) # @inspect no_print_time
    old_time = bench_mlp(MLPNewGelu) # @inspect old_time

    text("From the new trace, we can see that the gelu is now one kernel, and there's not CPU overhead now.")
    image("var/fixed_profile.png", width=800)

    text("**So the matrix multiplications are the main bottleneck now, and we need to speed them up**")


def deepseek_calc_diff(x, y):
    """https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/numeric.py#L5.
    
    Equivalent to ||x - y||^2 / (||x||^2 + ||y||^2)."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def mixed_precision():
    text("## Mixed precision")

    floats_in_memory()
    m, n, k = 16384, 16384 * 4, 16384
    a, b = generate_data(m, n, k)
    a_bf16, b_bf16 = a.to(torch.bfloat16), b.to(torch.bfloat16)

    fp32_time = triton.testing.do_bench(lambda: a @ b)
    bf16_time = triton.testing.do_bench(lambda: a_bf16 @ b_bf16) # @inspect bf16_time, @inspect fp32_time

    diff = deepseek_calc_diff(a @ b, a_bf16 @ b_bf16) # @inspect diff

    text("Spec (NVIDIA):")
    image("var/tflops_spec.png", width=600)
    text("**Divide these numbers by two:** (*) With sparsity")

    tflops = lambda ms, m, n, k: round(2 * m * n * k / (ms * 1e-3) / 1e12, 3)
    fp32_tflops = tflops(fp32_time, m, n, k)
    bf16_tflops = tflops(bf16_time, m, n, k) # @inspect fp32_tflops, @inspect bf16_tflops

    text("To be faster")
    text("  - Use more optimized libraries (torch already does this);")
    text("  - Use lower-precision, at least FP32 -> BF16.")

    old_time = bench_mlp(SlowMLP) # @inspect old_time
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        bf16_time = bench_mlp(SlowMLP) # @inspect bf16_time


def floats_in_memory():
    text("Numbers are represented as bits: 1 is reserved for the sign, the rest is divided between the exponent and the mantissa.")
    image("var/bit16_types.png", width=800)

    text("To get the final number:")
    image("var/float_representation.png", width=600)

    text("**The more the exponent is, the more the range is. The more the mantissa is, the more the precision is.**")


def fp8():
    fp8_intro()
    fp8_per_tensor()
    fp8_models()
    fp8_adoption()
    fp8_blockwise()
    fp8_linear()
    fp8_experience()


def fp8_intro():
    text("## FP8")
    text("We'd like to be even faster thanks to FP8.")

    text("There're two dtypes: e4m3 and e5m2 "),  link(title="[Paper]", url="https://arxiv.org/pdf/2209.05433")
    image("var/float8_for_dl.png", width=800)


def fp8_per_tensor():
    text("### Converting BF16 to FP8")
    text("Basically, we want to convert a range `[matrix_min, matrix_max]` to `[-448, 448]`.")
    text("For this:")
    text("  - Calculate the `absmax` for the matrix.")
    text("  - `Scale = 448. / absmax`. Multiple by scale.")
    text("  - Cast to e4m3.")
    image("var/convert_to_fp8.png", width=600)

    text("Because of the cast, we'd inevitable loose some precision, so the quantization is non-reversible in a way.")
    image("var/bf16_to_fp8_problems.png", width=600)

    text("### There're some problems with this approach")
    text("  - Not all operations (RMSNorm, Softmax etc.) are FP8-friendly, otherwise we'd loose the precision.")
    text("  - Outliers ruin the quality: tensors are huge, so there will be outliers.")


def fp8_models():
    text("## FP8 in the wild")

    text("### Llama-4")
    text("Almost no info: 'we focus on efficient model training by using FP8 precision, without sacrificing quality' "), link(title="[paper]", url="https://ai.meta.com/blog/llama-4-multimodal-intelligence/")

    text("### Nemotron-H-56B-Base")
    text("First and last 4 layers are in BF16 "),  link(title="[Paper]", url="https://arxiv.org/pdf/2504.03624")
    text("Loss is 0.1% higher, but the perf on benchmarks is the same as in BF16.")
    text("The stability must be checked with at least 1T tokens.")
    text("Math&code benchmarks are somehow 1-2% better.")
    image("var/nemotron_loss.png", width=800)

    text("### Cohere Command A")
    text("FP32 weights, cast to FP8 before running the computation"),  link(title="[Paper]", url="https://arxiv.org/pdf/2504.00698")
    text("Softmax, layernorm, embedding in FP32. Attention in BF16. The rest is in FP8.")
    text("Warmup in BF16.")

    text("**DeepSeek V3**, leave it for later.")


def fp8_adoption():
    text("FP8 is getting more popular:")
    image("var/fp8_adoption.png", width=800)

    text("Less memory, more speed.")


def fp8_blockwise():
    text("### DeepSeek v3: block-wise quantization")
    image("var/dsv3_weights.png", width=400), image("var/dsv3_act.png", width=400)

    text("To prevent the outliers from ruining the quality, we quantize the weights 128x128, and activations 1x128:")
    text("  - There is not a single `amax` now, but it is calculated per block.")
    text("  - Scale is a materix now.")
    text("  - And we need to be able to efficiently multiply `(in_e4m3, in_scale) @ (w_e4m3, w_scale)`")

    text("#### DeepGEMM")
    text("DeepGEMM is open-sourced during the open-source week"),  link(title="[github]", url="https://github.com/deepseek-ai/DeepGEMM")
    image("var/dsv3_deepgemm.png", width=600)

    text("DeepSeek recipe")
    image("var/dsv3_flow.png", width=800)


_MAX_E4M3_VAL = torch.finfo(torch.float8_e4m3fn).max
_MAX_FP32_VAL = torch.finfo(torch.float32).max

@triton.jit
def compute_scale_from_amax(
    amax: tl.tensor,
    _MAX_E4M3_VAL: tl.constexpr,
    _MAX_FP32_VAL: tl.constexpr,
) -> tl.tensor:
    scale = tl.where(amax == 0, 1.0, _MAX_E4M3_VAL / amax)

    # 0 11111111 00000000000000000000000 = +INF in FP32
    is_inf = tl.cast(scale, tl.int32, bitcast=True) == 0x7F800000
    # Insted of using +INF, use the max FP32 value
    scale = tl.where(is_inf, _MAX_FP32_VAL, scale)

    # 1 11111111 00000000000000000000000 - use log-scales
    scale_bits = tl.cast(scale, tl.uint32, bitcast=True)
    scale = tl.cast(scale_bits & 0xFF800000, tl.float32, bitcast=True)

    return scale


@triton.jit
def _quant_block_to_e4m3_kernel(
    src,
    mask,
    QUANT_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    _MAX_E4M3_VAL: tl.constexpr,
    _MAX_FP32_VAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE % QUANT_BLOCK_SIZE == 0)
    QUANT_BLOCKS_PER_BLOCK: tl.constexpr = BLOCK_SIZE // QUANT_BLOCK_SIZE

    # src is a small block, we can cast it to fp32
    src = src.to(tl.float32)
    src = tl.where(mask, src, 0.0)

    # [QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE]
    src = tl.reshape(src, (QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE))

    # Calc the absmax: one per QUANT_BLOCK_SIZE
    # The final shape is [QUANT_BLOCKS_PER_BLOCK]
    amax = tl.max(tl.abs(src), axis=1)

    # In this terminology, scale is something we have to multiply by to get FP8
    scale = compute_scale_from_amax(amax, _MAX_E4M3_VAL, _MAX_FP32_VAL)
    scale_inv = 1.0 / scale

    # src is of shape [QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE], multiply by scale which is [QUANT_BLOCKS_PER_BLOCK]
    # The final shape is [QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE]
    # Each block is multiplied by its own scale
    dst = src * tl.expand_dims(scale, 1)

    # Go back to the original [BLOCK_SIZE]
    dst = tl.reshape(dst, (BLOCK_SIZE,))
    # Cast to e4m3
    dst = dst.to(tl.float8e4nv)

    return dst, scale_inv


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE})
        for BLOCK_SIZE in (128, 256, 512, 1024, 2048, 4096, 8192)
    ],
    key=["M", "N"]
)
@triton.jit
def _quant_activation_to_e4m3_kernel(
    src_ptr,
    dst_ptr,
    scale_dst_ptr,
    M: int,
    N: int,
    QUANT_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    _MAX_E4M3_VAL: tl.constexpr,
    _MAX_FP32_VAL: tl.constexpr,
):
    # Check that we want to quantize a multiple of 128 elements
    tl.static_assert(BLOCK_SIZE % QUANT_BLOCK_SIZE == 0)
    # How many 128-element-long blocks we process
    QUANT_BLOCKS_PER_BLOCK: tl.constexpr = BLOCK_SIZE // QUANT_BLOCK_SIZE

    total_elements = M * N
    tl.device_assert(total_elements % QUANT_BLOCK_SIZE == 0)
    # A single scale per block of QUANT_BLOCK_SIZE elements
    scale_size = total_elements // QUANT_BLOCK_SIZE

    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    scale_offs = pid * QUANT_BLOCKS_PER_BLOCK + tl.arange(0, QUANT_BLOCKS_PER_BLOCK)
    scale_mask = scale_offs < scale_size

    # Read BLOCK_SIZE elements 
    src = tl.load(src_ptr + offs, mask=mask)

    # Get the E4M3 dst, FP32 scale_inv. To get back to bf16, we'd need to multiply the FP8 value by scale_inv.
    dst, scale_inv = _quant_block_to_e4m3_kernel(
        src=src,
        mask=mask,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        _MAX_E4M3_VAL=_MAX_E4M3_VAL,
        _MAX_FP32_VAL=_MAX_FP32_VAL,
    )

    tl.store(dst_ptr + offs, dst, mask=mask)
    tl.store(scale_dst_ptr + scale_offs, scale_inv, mask=scale_mask)


@triton_op("llm_scaling_week::quant_activation", mutates_args=("dst", "scale_dst"))
def quant_activation_to_e4m3(
    src: torch.Tensor,
    dst: torch.Tensor,
    scale_dst: torch.Tensor,
    quant_block_size: int = 128,
) -> None:
    assert src.is_contiguous()
    assert dst.is_contiguous()
    assert scale_dst.is_contiguous()

    assert src.size(-1) % quant_block_size == 0
    assert scale_dst.numel() == src.numel() // quant_block_size
    assert dst.size() == src.size()

    assert dst.dtype == torch.float8_e4m3fn
    assert scale_dst.dtype == torch.float32

    if src.dim() == 1:
        M, N = 1, src.shape[0]
    elif src.dim() == 2:
        M, N = src.shape
    else:
        raise ValueError("Unsupported tensor shape")

    grid = lambda meta: (triton.cdiv(src.numel(), meta["BLOCK_SIZE"]),)
    wrap_triton(_quant_activation_to_e4m3_kernel)[grid](
        src,
        dst,
        scale_dst,
        M=M,
        N=N,
        QUANT_BLOCK_SIZE=quant_block_size,
        _MAX_E4M3_VAL=_MAX_E4M3_VAL,
        _MAX_FP32_VAL=_MAX_FP32_VAL,
    )


@triton.jit
def _quant_weight_to_e4m3_kernel(
    src_ptr,
    dst_ptr,
    scale_ptr,
    M: int,
    N: int,
    BLOCK_SIZE: tl.constexpr,
    _MAX_E4M3_VAL: tl.constexpr,
    _MAX_FP32_VAL: tl.constexpr,
):
    # MxN, read BLOCK_SIZExBLOCK_SIZE blocks
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)

    # Rows to read: each program reads BLOCK_SIZE consecutive rows
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 
    # Columns to read in every row
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # BLOCK_SIZE x BLOCK_SIZE block
    offs = offs_m[:, None] * N + offs_n[None, :]
    # Prevent out-of-bounds
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    src = tl.load(src_ptr + offs, mask=mask).to(tl.float32)

    # absmax for this blocks is just one number
    amax = tl.max(tl.abs(src))
    scale = compute_scale_from_amax(amax, _MAX_E4M3_VAL, _MAX_FP32_VAL)
    scale_inv = 1.0 / scale

    # Scale the block and cast it to E4M3
    dst = (src * scale).to(dst_ptr.dtype.element_ty)

    tl.store(dst_ptr + offs, dst, mask=mask)
    tl.store(scale_ptr + pid_m * n + pid_n, scale_inv)


@triton_op("llm_scaling_week::quant_weight", mutates_args=("dst", "scale_dst"))
def quant_weight_to_e4m3(
    src: torch.Tensor,
    dst: torch.Tensor,
    scale_dst: torch.Tensor,
    block_size: int = 128,
) -> None:
    assert src.is_contiguous()
    assert dst.is_contiguous()
    assert scale_dst.is_contiguous()

    assert len(src.shape) == 2
    M, N = src.size()
    assert src.size() == dst.size()
    assert scale_dst.size() == ((M + block_size - 1) // block_size, (N + block_size - 1) // block_size)

    assert dst.dtype == torch.float8_e4m3fn
    assert scale_dst.dtype == torch.float32

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    wrap_triton(_quant_weight_to_e4m3_kernel)[grid](
        src, dst, scale_dst, M=M, N=N, BLOCK_SIZE=block_size, _MAX_E4M3_VAL=_MAX_E4M3_VAL, _MAX_FP32_VAL=_MAX_FP32_VAL
    )


def alloc_and_quant_weight(w_bf16, quant_block_size: int = 128):
    n, m = w_bf16.shape
    assert n % quant_block_size == 0 and m % quant_block_size == 0
    w_scale_shape = (n // quant_block_size, m // quant_block_size)
    w_e4m3 = torch.empty_like(w_bf16, dtype=torch.float8_e4m3fn)
    w_scale = torch.empty(*w_scale_shape, device=w_bf16.device, dtype=torch.float32)
    torch.ops.llm_scaling_week.quant_weight(w_bf16, w_e4m3, w_scale, block_size=128)
    return w_e4m3, w_scale


def alloc_and_quant_activation(x_bf16, quant_block_size: int = 128):
    n, m = x_bf16.shape
    assert m % quant_block_size == 0
    x_scale_shape = (n, m // quant_block_size)
    x_e4me = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(*x_scale_shape, device=x_bf16.device, dtype=torch.float32)
    torch.ops.llm_scaling_week.quant_activation(x_bf16, x_e4me, x_scale, quant_block_size=quant_block_size)
    return x_e4me, x_scale


def fp8_linear():
    text("### Writing the block-wise FP8 from scratch")
    text("First, start by implementing the quantization kernels")
    text("For weights, we will read blocks of 128x128, compute the scale for the block and write the result.")
    text("For activations, we will read blocks of 1x128, compute the scale for the block and write the result.")

    x_bf16 = torch.randn((16384, 16384), dtype=torch.bfloat16, device="cuda")
    x_e4m3, x_scale = alloc_and_quant_activation(x_bf16)

    w_bf16 = torch.randn((16384, 16384 * 4), dtype=torch.bfloat16, device="cuda")
    w_e4m3, w_scale = alloc_and_quant_weight(w_bf16)

    fp8_matmul()


class Fp8FwdBf16Linear(torch.autograd.Function):
    def forward(ctx, x, w):
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        gemm_fp8_fp8_bf16_nt(alloc_and_quant_activation(x), alloc_and_quant_weight(w), out)
        ctx.save_for_backward(x, w) # Can be stored in FP8, store the original values here for simplicity
        return out

    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        x_grad = w_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = grad_output @ w
        if ctx.needs_input_grad[1]:
            w_grad = grad_output.t() @ x
        return x_grad, w_grad


def fp8_matmul():
    m, n, k = 16384, 16384, 16384 * 4
    x_bf16 = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")

    gemm_fp8_fp8_bf16_nt(
        alloc_and_quant_activation(x_bf16),
        alloc_and_quant_weight(w_bf16),
        out,
    )
    diff = deepseek_calc_diff(out, x_bf16 @ w_bf16.T) # @inspect diff

    def run_fp8(inp, linear):
        w = linear.weight.data.detach().clone().requires_grad_(True)
        x = inp.detach().clone().requires_grad_(True)
        out = Fp8FwdBf16Linear.apply(x, w)
        (out**2).mean().backward()
        return out, x.grad, w.grad

    def run_bf16(inp, linear):
        out = linear(inp)
        (out**2).mean().backward()
        return out, inp.grad, linear.weight.grad

    linear = torch.nn.Linear(2048, 1024, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    x = torch.randn((1024, 2048), dtype=torch.bfloat16, device="cuda").requires_grad_(True)

    diffs = []
    for t_fp8, t_bf16 in zip(run_fp8(x, linear), run_bf16(x, linear)):
        diff = deepseek_calc_diff(t_fp8, t_bf16) 
        diffs.append(diff) # @inspect diffs


def fp8_experience():
    text("### Our experience with FP8")

    text("Compare BF16 vs. FP8")
    image("var/bf16_profile_sacrifice.png", width=800)
    image("var/fp8_profile_sacrifice.png", width=800)

    text("From the trace above, we can see that the iteration time increases with FP8")
    text("Moreover, FP8 GEMMs take as much time as in BF16.")

    text("Because of the overlap with the communication stream, there's a 'fight' for SMs vs. NCCL, which makes the E2E iteration slower.")
    text("**To fix:**, we just need to sacrifice 16 SMs for NCCL, and run FP8 GEMM on the remaining ones.")
    image("var/fp8_profile_sacrifice_fix.png", width=800)


def set_mlp_weights(mlp, fc1_weight, fc2_weight):
    mlp.fc1.weight.data = fc1_weight
    mlp.fc2.weight.data = fc2_weight


def run_mlp(mlp, x, fc1_weight, fc2_weight):
    new_x, new_fc1, new_fc2 = (
        x.clone().requires_grad_(True),
        fc1_weight.clone().requires_grad_(True),
        fc2_weight.clone().requires_grad_(True)
    )
    set_mlp_weights(
        mlp,
        new_fc1,
        new_fc2,
    )
    return mlp(new_x)


def speeding_up_mlp():
    text("Use all we learnt to speed up MLP:")
    text("  - Remove the print-statement to get rid of the CPU-syncs.")
    text("  - Use the fused gellu-kernel.")
    text("  - Use FP8-linear layers.")
    text("Check against FP32 and BF16.")

    class MLPFP8(SlowMLP):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gelu_kernel = torch.ops.llm_scaling_week.gelu_triton_auto

        def up_proj(self, x):
            return Fp8FwdBf16Linear.apply(x, self.fc1.weight.data)

        def down_proj(self, x):
            return Fp8FwdBf16Linear.apply(x, self.fc2.weight.data)

        def get_stat(self, x):
            pass

    fp32_time = bench_mlp(SlowMLP) # @inspect fp32_time
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        bf16_time = bench_mlp(SlowMLP) # @inspect bf16_time
    fp8_time = bench_mlp(MLPFP8) # @inspect fp8_time

    text("Check the quality!")

    with torch.device("cuda"):
        x = torch.randn(1024, 16384, dtype=torch.float32) * 1000 + 5
        mlp_fp8 = MLPFP8(16384)
        mlp_fp32 = SlowMLP(16384)

        fc1_weight = torch.randn_like(mlp_fp32.fc1.weight) * 100 - 30
        fc2_weight = torch.randn_like(mlp_fp32.fc2.weight) * 100 + 1000

        fp8_res = run_mlp(mlp_fp8, x.to(torch.bfloat16), fc1_weight.to(torch.bfloat16), fc2_weight.to(torch.bfloat16))
        fp32_res = run_mlp(mlp_fp32, x, fc1_weight, fc2_weight)
        diff = deepseek_calc_diff(fp8_res, fp32_res) # @inspect diff


def resources():
    text("Efficient DL "), link(title="[GitHub]", url="https://github.com/mryab/efficient-dl-systems/tree/main/week03_fast_pipelines")
    text("CS336, Stanford "), link(title="[Website]", url="https://stanford-cs336.github.io/spring2025/")

    text("CUDA MODE Lecture 1: how to profile CUDA kernels in PyTorch "), link(title="[Video]", url="https://www.youtube.com/watch?v=LuhJEEJQgUM")
    text("CUDA MODE Lecture 2: Chapters 1-3 of PPMP book "), link(title="[Video]", url="https://www.youtube.com/watch?v=NQ-0D5Ti2dc")
    text("CUDA MODE Lecture 3: Getting started with CUDA for Python Programmers "), link(title="[Video]", url="https://www.youtube.com/watch?v=4sgKnKbR-WE")
    text("CUDA MODE Lecture 4: Compute and memory basics "), link(title="[Video]", url="https://www.youtube.com/watch?v=lTmYrKwjSOU")
    text("CUDA MODE Lecture 8: CUDA performance checklist "), link(title="[Video]", url="https://www.youtube.com/watch?v=SGhfUhlowB4")

    text("Transformer Engine "), link(title="[Docs]", url="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html")
    text("Torch Mixed Precision "), link(title="[Docs]", url="https://docs.pytorch.org/docs/stable/amp.html")

    text("Triton Puzzles "), link(title="[GitHub]", url="https://github.com/srush/Triton-Puzzles")
    text("How to Scale Your Model "), link(title="[Blog]", url="https://jax-ml.github.io/scaling-book/gpus/")
    text("The Ultra-Scale Playbook "), link(title="[Blog]", url="https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=kernels")
