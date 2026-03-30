"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", add_python=None)
    .apt_install("git", "build-essential", "cmake", "wget")
    .run_commands(
        # Install CUDA 12.8 toolkit for sm_100 support
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-8 || apt-get install -y cuda-nvcc-12-8",
        # Make nvcc 12.8 default
        "ln -sf /usr/local/cuda-12.8/bin/nvcc /usr/local/bin/nvcc || true",
    )
    .pip_install("flashinfer-bench", "triton", "numpy")
    .add_local_file("/tmp/cutlass_headers.tar.gz", "/tmp/cutlass_headers.tar.gz", copy=True)
    .run_commands(
        # Install CUTLASS 4.4.2 headers
        "cd /usr/local/include && tar xzf /tmp/cutlass_headers.tar.gz --strip-components=1 include/ && "
        "tar xzf /tmp/cutlass_headers.tar.gz --strip-components=3 tools/util/include/ && "
        "echo 'CUTLASS 4.4.2 installed'",
    )
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    import os
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"  # Enable sm_100a for TMA/UMMA support
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()
            log_text = result.get("log")
            if log_text:
                log_text = log_text.strip()
                if len(log_text) > 4000:
                    log_text = log_text[:4000] + "\n...[log truncated]"
                print(log_text)


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
