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
    text("# Ускоряем обучения за счет Triton и FP8")
    text("Привет! Меня зовут Влад, я занимаюсь ускорением больших LLM-обучений без потери качества.")

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
    text("## Замеры скорости")
    text("Чтобы что-то ускорять, надо научиться делать замеры")
    text("Давайте посчитаем, сколько занимает матричное умножение:")

    perf_t = bench_matmuls("perf_counter")
    cpu_scheduling()
    cuda_events_t = bench_matmuls("cuda.events")
    triton_t = bench_matmuls("triton.testing") # @inspect perf_t, @inspect cuda_events_t, @inspect triton_t


def generate_data(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Генерируем два FP32 тензора на GPU."""
    a = torch.randn(m, n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, k, dtype=torch.float32, device="cuda")
    return a, b


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Простая функция для матричного умножения."""
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

    # Запускаем замер на (16, 32) @ (32, 16)
    small_shapes_t = func(16, 32, 16)

    # Запускаем замер на (16384, 32768) @ (32768, 16384)
    large_shapes_t = func(16 * 1024, 32 * 1024, 16 * 1024)

    mean_small_shapes_t = statistics.mean(small_shapes_t)
    mean_large_shapes_t = statistics.mean(large_shapes_t) # @inspect mean_small_shapes_t, @inspect mean_large_shapes_t 
    return mean_small_shapes_t, mean_large_shapes_t


def cpu_scheduling():
    text("CPU-код 'шедулит' кернелы на GPU. Кернел - это операция, которая выполняется на GPU.")
    text("CPU-код может бежать вперед и ставить кернелы в очередь быстро, не дожидаясь GPU.")
    image("var/profile_image.png", width=800)
    text("Получается, наши замеры через time.time() на самом деле измеряют время (CPU-оверхед) на постановку кернела в очередь вместо его времени работы на GPU.")


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


class SlowMLP(torch.nn.Module):
    def __init__(self, dim: int = 16384):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim * 4, dtype=torch.float32)
        self.fc2 = torch.nn.Linear(dim * 4, dim, dtype=torch.float32)
        self.gelu_kernel = gelu

    def get_stat(self, x):
        """Часто код может логировать промежуточные результаты/активации."""
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

    # Время всего MLP-блока
    with redirect_stdout(buffer):
        model = model_cls().to("cuda")
        x = torch.randn(16384, 16384, dtype=torch.float32, device="cuda")
        mlp_time = triton.testing.do_bench(lambda: model(x), rep=num_iters, warmup=num_warmups) # @inspect mlp_time
    return mlp_time


def our_goal():
    text("## Наша цель на сегодня - ускорить MLP")
    text("Узнаем, сколько времени работает базовый вариант MLP:")

    # примерно про MLP
    bench_mlp(SlowMLP)

    text("Дополнительно давайте посмотрим, сколько занимает gelu:")
    inter = torch.randn(16384, 16384 * 4, dtype=torch.float32, device="cuda")
    gelu_time = triton.testing.do_bench(lambda: gelu(inter)) # @inspect gelu_time

    text("А это хорошо или плохо?")
    text("Надо погрузиться в то, как устроен ускоритель, чтобы ответить на этот вопрос.")
    cuda_intro()

    text("### Arithmetic intensity: #FLOPs / #bytes")
    text("- Если это значение большое, то операция compute-bound, так как на каждый байт приходится много вычислений;")
    text("- Иначе операция memory-bound, можем оценивать время на чтение-запись;")
    text("- На современных картах надо примерно 200 FLOPs/byte, чтобы стать compute-bound.")

    text("В случае gelu на N элементов в FP32 получится примерно 9N / 8N = 1.125.")

    text(f"**Посчитаем, сколько тратится на чтение.** У нас 16384x16384x4 элементов в FP32. То есть 16384x16384x4x4 байт на чтение и столько же на запись. При идеальной скорости в 3.35 TB/s, мы бы получили работу кернела в 16384x16384x4x4x2/3.35e12 ~ {(t := 16384**2 * 4**2 * 2 / 3.35e12):.5f} sec = {t * 1000:.3f}ms")
    
    text(f"А у нас {gelu_time:.3f} ms! То есть в {round(gelu_time / t / 1000, 2)} раз больше.")


def cuda_intro():
    text("## CUDA")
    text("### Модель вычислений")
    text("SIMT: Single Instruction Multiple Threads.")
    text("Условно независимые вычислительные ядра - SMки.")
    text("К каждому ядру приделан кеш, он 'подключен' к памяти GPU.")
    image("var/h100_highlevel.png", width=800)

    text("### Compute")
    text("H100:")
    text("  - Множество ± независимых Streaming Multiprocessor-ов (SM)")
    text("  - Tensor Core для матричного умножения;")
    text("  - Vector arithmetic unit для арифметических операций;")

    text("Внутри SMки 4 одинаковых партишна. В каждом:")
    text("  - Warp Scheduler;")
    text("  - CUDA Cores: 32 fp32 кора, все исполняют одну инструкцию за такт;")
    text("  - Tensor Core: матричное умножение, большинство FLOPs-ов приходится на него;")
    text("  - Например, в bf16 H100 обеспечивает ±990 TFLOPs/s. То есть 990e12 / 132 (SM) / 4 (subpartitions) / 1.76e9 (Hz) ~ 1065 bf16 FLOPs/cycle. То есть ~матричное умножение 8x8x8.")

    image("var/light-gh100-sm.svg", width=800)

    text("Треды объединяются в блоки.")
    text("Grid: коллекция тред блоков.")
    image("var/thread_blocks.png", width=600)

    text("Thread block-и могут быть зашедулены по-разному на разных девайсах.")
    image("var/light-wave-scheduling.svg", width=600)

    text("Тредблоки шедулятся на SMки волнами")

    text("**Важно помнить:** если два треда в одном Warp-е (коллекция из 32 тредов) делают разную работу, Warp выполняет обе инструкции.")
    image("https://jax-ml.github.io/scaling-book/assets/gpu/warp-divergence.png", width=800)

    text("### Memory")
    text("Registers -> SMEM -> L2 Cache -> HBM")
    text("**Registers:** 16384 32-битных слов в каждом партишне;")
    text("**SMEM:** L1 cache, каждая SMка содержит свой 228KiB, 31TB/s кеш (shared memory);")
    text("**L2 Cache:** все SMки шерят 50MB L2. Badwidth мы не знаем, но примерно 12TB/s.")
    text("**HBM:** основная память карточки, 80GB в H100, репортят 3.35TB/s.")
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
    text("- Пишем кернел на Python;")
    text("- Не надо думать на уровне тредов (хорошо и плохо), контролировать SMEM и т.д.;")

    text("What does Triton offer?", verbatim=True)
    text("                                             CUDA      Triton", verbatim=True)
    text("- Memory coalescing (transfer from DRAM)     manual    automatic", verbatim=True)
    text("- Shared memory management                   manual    automatic", verbatim=True)
    text("- Scheduling within SMs                      manual    automatic", verbatim=True)
    text("- Scheduling across SMs                      manual    manual", verbatim=True)


@triton.jit
def _add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Определим, в каком блоке мы находимся
    pid = tl.program_id(0)
    # Прошлые блоки прочитали pid * BLOCK_SIZE элементов, отступим столько от x_ptr
    block_start = pid * BLOCK_SIZE
    # Зададим оффсеты, по которым будем читать - это BLOCK_SIZE элементов, начиня с block_start
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Нужна маска, чтобы не прочитать лишнего
    mask = offsets < n_elements
    # Загружаем из HBM
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Выполняем операцию
    output = x + y
    # Загружаем в out_ptr по тем же оффсетам и с той же маской
    tl.store(output_ptr + offsets, output, mask=mask)


# Обычная функция, вызывает Triton-кернел внутри себя
def add_vectors(x: torch.Tensor, y: torch.Tensor):
    # Добавим проверки перед запуском кернела
    assert x.is_contiguous() and y.is_contiguous(), "Expected x and y to be contiguous"
    assert x.device == y.device, f"Expected x and y to be on the same device, found: {x.device} != {y.device}"
    assert x.dtype == y.dtype, f"Expected x and y to have the same dtype, found: {x.dtype} != {y.dtype}"
    assert x.shape == y.shape, f"Expected x and y to have the same shape, found: {x.shape} != {y.shape}"

    output = torch.empty_like(x.view(-1))

    # Создадим grid тредблоков - каждый блок будет отвечать за обработку BLOCK_SIZE элементов.
    # Всего блоков надо взять столько, чтобы хватило покрыть все элементы в тензоре = x.numel()
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    # Явно зададим BLOCK_SIZE
    _add_kernel[grid](x.view(-1), y.view(-1), output, x.numel(), BLOCK_SIZE=32)

    return output.view(x.shape)


def triton_add():
    x = torch.randn(16, 100, device="cuda")
    y = torch.randn(16, 100, device="cuda")
    quality_ok = torch.allclose(add_vectors(x, y), x + y) # @inspect quality_ok
    kernel_time = triton.testing.do_bench(lambda: add_vectors(x, y))
    torch_time = triton.testing.do_bench(lambda: x + y) # @inspect kernel_time, @inspect torch_time


def triton_and_torch():
    text("В Torch-е обычно операция автодифференцируемы, что происходит с triton?")
    triton_bad_backward()
    triton_impl_backward()


def triton_bad_backward():
    try:
        x = torch.randn(16, 100, device="cuda").requires_grad_(True)  # Проставим флаг о том, что по тензору должен течь градиент
        y = torch.randn(16, 100, device="cuda").requires_grad_(True)
        out = add_vectors(x, y)
        # Попробуем запустить .backward() проход по графу
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


# Добавляем декоратор-обертку. Нужно указать схему (тайпинги) и какие аргументы могут измениться внутри функции.
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
        """Функция получается dL/dOut, возвращает градиент dL/dX и dL/dY."""
        return grad_output, grad_output

    wrapped_add_vectors.register_autograd(wrapped_add_vectors_bwd)   # Регистрируем backward-функцию для autograd

    # Проверим, что градиент у нашего Triton-кернела совпадает с градиентов от встроенной Torch-функции
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
    # Функция будет вызываться в конце forward, можно сохранить в контекст для бекварда все, что нужно
    # Доступ есть как ко входам в forward, так и к выходам из него
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    # Функция вызывается при backward-проходе, имеет доступ к контексту ctx, который был сохранен
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * x.cos()

    mysin.register_autograd(backward, setup_context=setup_context)


def triton_and_cpu():
    # Попробуем запустить сложение на CPU
    x, y = torch.randn(16, 100), torch.randn(16, 100)
    try:
        torch.ops.llm_scaling_week.add_vectors(x, y)
    except Exception as e:
        error = str(e) # @inspect error

    # Зарегистрируем фоллбек на CPU
    @wrapped_add_vectors.register_kernel("cpu")
    def _(a, b):
        return a + b

    out_device = torch.ops.llm_scaling_week.add_vectors(x, y).device # @inspect out_device


@triton.jit
def _gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Делаем все по аналогии с add-кернелом: каждая программа читает свои BLOCK_SIZE элементов
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Загружаем и делаем upcast к FP32
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    # Проделаем все вычисления разом
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    output = 0.5 * x * (1 + triton_tanh(a))
    # Делаем downcast к выходному типу
    output = output.to(output_ptr.dtype.element_ty)
    # Сохраним результат
    tl.store(output_ptr + offsets, output, mask=mask)


@triton_op("llm_scaling_week::gelu_triton", mutates_args={})
def gelu_triton(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x.view(-1))
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    wrap_triton(_gelu_kernel)[grid](x.view(-1), out, x.numel(), BLOCK_SIZE=32)
    return out.view(x.shape)


def triton_gelu():
    text("Используя наши знания про Triton, давайте напишем Gelu")

    x = torch.randn(16384, 16384 * 4, dtype=torch.float32, device="cuda")
    gelu_time = triton.testing.do_bench(lambda: torch.ops.llm_scaling_week.gelu_triton(x)) # @inspect gelu_time
    torch_time = triton.testing.do_bench(lambda: gelu(x)) # @inspect torch_time

    # Проверим, что результаты совпадают с torch gelu
    ok = torch.allclose(gelu(x), torch.ops.llm_scaling_week.gelu_triton(x), atol=1e-6) # @inspect ok

    text("Почему-то все еще медленно...")
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
    # Больше не передаем сюда BLOCK_SIZE
    wrap_triton(_gelu_kernel_autotune)[grid](x.view(-1), out, x.numel())
    return out.view(x.shape)


def triton_autotune():
    text("Подберем BLOCK_SIZE автоматически.")

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

    # Давайте заодно запустим бенчмарк встроенного в torch кернела
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
        prof.export_chrome_trace(f"./profile/chrome_trace_{func}.json")

    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True, # Записывать ли CPU-stack вызова
        record_shapes=True, # Записывать ли размеры тензоров
        on_trace_ready=trace_handler, # Что делать при готовности трейса
    ) as prof:
        func()

    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=80)


def profiling():
    text("## Профилирование")
    text("Мы научились понимать, сколько времени уходит на операцию. Но пока не поняли, на что это время тратится. Чтобы увидеть, из каких кернелов состоит код, и когда они запускаются, можно снять профиль.")

    run_profile(
        lambda: bench_mlp(SlowMLP, 1, 1)
    )

    text("Если открыть сохраненный профиль в https://ui.perfetto.dev, можно увидеть:")
    image("var/mlp_profile.png", width=800)

    text("Из профиля выше понятно, почему gelu занимал так много времени:")
    image("var/gelu_gpu_stream.png", width=800)

    text("Также можно идентифицировать еще одну проблему: CPU-bound вычисления.")
    image("var/cpu_bound_mlp.png", width=800)

    text("Вспомним про то, как выглядит шедулинг CPU -> GPU:")
    image("var/profile_image.png", width=800)
    text("Если мы захотим сделать print результата запуска кернела, то нам на CPU надо дождаться готовности результата на GPU. Отсюда происходит synchronize:")
    image("var/profile_cpu_blocked_image.png", width=800)

    text("Можно убрать лишний CPU-оверхед:")

    class MLPNewGelu(SlowMLP):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gelu_kernel = torch.ops.llm_scaling_week.gelu_triton_auto

    class MLPNoPrint(MLPNewGelu):
        def get_stat(self, x):
            """Можно сделать эту функцию не блокирующей, но для простоты мы ее оставим пустой."""
            pass

    no_print_time = bench_mlp(MLPNoPrint) # @inspect no_print_time
    old_time = bench_mlp(MLPNewGelu) # @inspect old_time

    text("На новом профиле видно, что gelu превратился в один кернел, а CPU-оверхед пропал:")
    image("var/fixed_profile.png", width=800)

    text("**Теперь главным боттлнеком выступают матричные умножения, и нужно ускорять их**")


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

    text("Посмотрим в спецификацию от Nvidia")
    image("var/tflops_spec.png", width=600)
    text("**Эти числа нужно делить на два:** (*) With sparsity")

    tflops = lambda ms, m, n, k: round(2 * m * n * k / (ms * 1e-3) / 1e12, 3)
    fp32_tflops = tflops(fp32_time, m, n, k)
    bf16_tflops = tflops(bf16_time, m, n, k) # @inspect fp32_tflops, @inspect bf16_tflops

    text("Как стать быстрее?")
    text("  - Использовать оптимизированные библиотеки (torch уже это делает);")
    text("  - Переходить к типам пониженной точности, как минимум fp32 -> bf16.")

    old_time = bench_mlp(SlowMLP) # @inspect old_time
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        bf16_time = bench_mlp(SlowMLP) # @inspect bf16_time


def floats_in_memory():
    text("Числа представлены битами: 1 отведен под знак, остальные поделены между экспонентой и мантиссой.")
    image("var/bit16_types.png", width=800)

    text("Итоговое число можно получить вот так:")
    image("var/float_representation.png", width=600)

    text("**Больше экспонента - больше диапазон. Больше мантисса - больше точности.**")


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
    text("Было бы круто стать еще быстрее за счет использования FP8.")

    text("Два типа: e4m3 и e5m2 "),  link(title="[Paper]", url="https://arxiv.org/pdf/2209.05433")
    image("var/float8_for_dl.png", width=800)


def fp8_per_tensor():
    text("### Как переводить bf16 в fp8")
    text("По сути мы хотим перевести диапазон `[maxtrix_min, matrix_max]` в `[-448, 448]`.")
    text("Для этого:")
    text("  - Считаем `absmax` по всей матрице;")
    text("  - Делим на `absmax`, домножаем на 448;")
    text("  - Кастим к e4m3.")
    image("var/convert_to_fp8.png", width=600)

    text("От такого каста неизбежно появляется потери точности, поэтому наша операция не является обратимой:")
    image("var/bf16_to_fp8_problems.png", width=600)

    text("### У такого подхода есть проблемы")
    text("Трудности:")
    text("  - Не все операции (RMSNorm, Softmax etc.) можно делать в FP8;")
    text("  - Выбросы снижают точность квантизации: тензоры большие, выбросы точно найдутся.")


def fp8_models():
    text("## Кто использовал FP8")

    text("### Llama-4")
    text("Почти нет информации, но 'we focus on efficient model training by using FP8 precision, without sacrificing quality' "),  link(title="[paper]", url="https://ai.meta.com/blog/llama-4-multimodal-intelligence/")

    text("### Nemotron-H-56B-Base")
    text("Первые и последние 4 слоя в BF16 "),  link(title="[Paper]", url="https://arxiv.org/pdf/2504.03624")
    text("Loss выше примерно на 0.1%, но бенчмарки такие же, как у BF16;")
    text("Стабильность надо проверять на больших моделях на большой дистанции;")
    text("Математика и код лучше на 1-2%.")
    image("var/nemotron_loss.png", width=800)

    text("### Cohere Command A")
    text("Веса в FP32, перед вычислениями делают каст к FP8 "),  link(title="[Paper]", url="https://arxiv.org/pdf/2504.00698")
    text("Softmax, layernorm, embedding в FP32. Attention в BF16. Остальное в FP8.")
    text("Warmup в BF16.")

    text("**DeepSeek V3**, поговорим позже.")


def fp8_adoption():
    text("FP8 как формат становится популярнее:")
    image("var/fp8_adoption.png", width=800)

    text("Меньше памяти, больше скорости.")


def fp8_blockwise():
    text("### DeepSeek v3 с блочной квантизацией")
    image("var/dsv3_weights.png", width=400), image("var/dsv3_act.png", width=400)

    text("Чтобы бороться с выбросами, давайте квантовать веса 128x128, а активации 1x128:")
    text("  - `amax` считаем не по всей матрице, а внутри блока;")
    text("  - Теперь scale - тоже матрица;")
    text("  - Надо научиться эффективно умножать `(in_e4m3, in_scale) @ (w_e4m3, w_scale)`")

    text("#### DeepGEMM")
    text("Во время open source week выложили DeepGEMM "),  link(title="[github]", url="https://github.com/deepseek-ai/DeepGEMM")
    image("var/dsv3_deepgemm.png", width=600)

    text("Рецепт DeepSeek")
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

    # 0 11111111 00000000000000000000000 = +INF в FP32
    is_inf = tl.cast(scale, tl.int32, bitcast=True) == 0x7F800000
    # берем максимально возможное FP32 число вместо +INF
    scale = tl.where(is_inf, _MAX_FP32_VAL, scale)

    # 1 11111111 00000000000000000000000 - используем скейлы=степени двойки
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

    # src - маленький вычитанный блок, можем его скастовать к fp32
    src = src.to(tl.float32)
    src = tl.where(mask, src, 0.0)

    # [QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE]
    src = tl.reshape(src, (QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE))

    # Считаем absmax: по одному на каждый QUANT_BLOCK_SIZE
    # Итоговая размерность - [QUANT_BLOCKS_PER_BLOCK]
    amax = tl.max(tl.abs(src), axis=1)

    # В нашей терминологии scale - то, на что нужно домножить, чтобы получить FP8
    scale = compute_scale_from_amax(amax, _MAX_E4M3_VAL, _MAX_FP32_VAL)
    scale_inv = 1.0 / scale

    # src размера [QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE] умножаем на scale [QUANT_BLOCKS_PER_BLOCK]
    # Итоговая размерность - [QUANT_BLOCKS_PER_BLOCK, QUANT_BLOCK_SIZE]
    # Каждый блок умножается на свой scale
    dst = src * tl.expand_dims(scale, 1)

    # Возвращаем к исходному [BLOCK_SIZE]
    dst = tl.reshape(dst, (BLOCK_SIZE,))
    # Кастим к e4m3
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
    # Проверим, что мы хотим считать кратное 128 количество элементов
    tl.static_assert(BLOCK_SIZE % QUANT_BLOCK_SIZE == 0)
    # Сколько блоков по 128 элементов мы считаем
    QUANT_BLOCKS_PER_BLOCK: tl.constexpr = BLOCK_SIZE // QUANT_BLOCK_SIZE

    total_elements = M * N
    # Проверим, что тензор в теории можно квантизировать
    tl.device_assert(total_elements % QUANT_BLOCK_SIZE == 0)
    # Один scale на один блок размером в QUANT_BLOCK_SIZE элементов
    scale_size = total_elements // QUANT_BLOCK_SIZE

    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    scale_offs = pid * QUANT_BLOCKS_PER_BLOCK + tl.arange(0, QUANT_BLOCKS_PER_BLOCK)
    scale_mask = scale_offs < scale_size

    # Прочитали BLOCK_SIZE элементов 
    src = tl.load(src_ptr + offs, mask=mask)

    # Получили dst в e4m3 и scale_inv в fp32. Чтобы вернуться обратно к bf16, надо будет домножить fp8 на scale_inv.
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
    # Матрица MxN, читаем блоками по BLOCK_SIZE x BLOCK_SIZE
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)

    # Какие строчки прочитать: каждая программа читаем BLOCK_SIZE подряд идущих строк
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Какие столбцы прочитать в каждой строчке
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Блок BLOCK_SIZE x BLOCK_SIZE
    offs = offs_m[:, None] * N + offs_n[None, :]
    # Чтобы не выйти за границы матрицы
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    src = tl.load(src_ptr + offs, mask=mask).to(tl.float32)

    # Считаем absmax для блока - одно число
    amax = tl.max(tl.abs(src))
    scale = compute_scale_from_amax(amax, _MAX_E4M3_VAL, _MAX_FP32_VAL)
    scale_inv = 1.0 / scale

    # Скейлим блок на scale и кастим к e4m3
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
    text("### Как написать блочный FP8-рецепт своими руками")
    text("Сначала напишем квантизацию.")
    text("Для квантизации весов будем читать блоки 128x128, считать для блока скейл и записывать результат.")
    text("Для квантизации активаций будем читать несколько блоков 1x128 за один раз.")

    x_bf16 = torch.randn((16384, 16384), dtype=torch.bfloat16, device="cuda")
    x_e4m3, x_scale = alloc_and_quant_activation(x_bf16)

    w_bf16 = torch.randn((16384, 16384 * 4), dtype=torch.bfloat16, device="cuda")
    w_e4m3, w_scale = alloc_and_quant_weight(w_bf16)

    fp8_matmul()


class Fp8FwdBf16Linear(torch.autograd.Function):
    def forward(ctx, x, w):
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        gemm_fp8_fp8_bf16_nt(alloc_and_quant_activation(x), alloc_and_quant_weight(w), out)
        ctx.save_for_backward(x, w) # Потенциально можно сохранить в FP8
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
    text("### С чем мы столкнулись при использовании FP8")

    text("Посмотрим на профиль с BF16 умножениями и FP8 умножениями:")
    image("var/bf16_profile_sacrifice.png", width=800)
    image("var/fp8_profile_sacrifice.png", width=800)

    text("Из профиля видно, что время итерации с FP8 на самом деле растет!")
    text("Более того, геммы в FP8 занимают столько же, сколько в BF16.")

    text("Выяснилось, что из-за перекрытия NCCL-коммуникаций с вычислениями происходит борьба за SMки, из-за чего все замедляется.")
    text("**Решение:** делать sacrifice - пожертвовать 16 SMок, которые нужны NCCL-ю, запускать FP8 GEMM на оставшихся.")
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
    text("Используем все полученные знания для ускорения MLP:")
    text("  - Уберем print-ы для того, чтобы избавиться от лишней CPU-синхронизации;")
    text("  - Воспользуемся пофьюженным gelu-кернелом;")
    text("  - Используем FP8-линейные слои;")
    text("Сравним это с FP32 и BF16 вариантами.")

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

    text("Нужно не забыть проверить качество!")

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
