import argparse
import re
import subprocess
import time
import csv
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass
from typing import Callable, Dict, Any

# =============================================================================
# Backend Engines
# =============================================================================


class QiskitEngine:
    def warmup(self, benchmark_id):
        self.run(benchmark_id, 2)

    def run(self, benchmark_id, n):
        if benchmark_id == "qft":
            return self._run_qft(n)
        raise ValueError()

    def _run_qft(self, n):
        from qiskit import transpile
        from qiskit.synthesis import synth_qft_full
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        qc = synth_qft_full(num_qubits=n, do_swaps=False, inverse=False)
        qc.save_statevector()

        tqc = transpile(qc, backend, optimization_level=0)

        start = time.perf_counter()
        job = backend.run(tqc)
        job.result()
        return time.perf_counter() - start


class CirqEngine:
    def warmup(self, benchmark_id):
        self.run(benchmark_id, 2)

    def run(self, benchmark_id, n):
        if benchmark_id == "qft":
            return self._run_qft(n)
        raise ValueError()

    def _run_qft(self, n):
        import cirq

        qubits = [cirq.LineQubit(i) for i in range(n)]
        circuit = cirq.Circuit()

        for j in range(n):
            circuit.append(cirq.H(qubits[j]))
            for k in range(2, n - j + 1):
                angle = np.pi / (2 ** (k - 1))
                circuit.append(cirq.CZ(qubits[j + k - 1], qubits[j]) ** (angle / np.pi))

        sim = cirq.Simulator()

        start = time.perf_counter()
        sim.simulate(circuit)
        return time.perf_counter() - start


class PennyLaneEngine:
    def __init__(self, device):
        self.device = device

    def warmup(self, benchmark_id):
        self.run(benchmark_id, 2)

    def run(self, benchmark_id, n):
        if benchmark_id == "qft":
            return self._run_qft(n)
        raise ValueError()

    def _run_qft(self, n):
        import pennylane as qml

        dev = qml.device(self.device, wires=n)

        @qml.qnode(dev)
        def circuit():
            for j in range(n):
                qml.Hadamard(wires=j)
                for k in range(2, n - j + 1):
                    qml.ControlledPhaseShift(
                        np.pi / (2 ** (k - 1)),
                        wires=[j, j + k - 1],
                    )
            return qml.state()

        start = time.perf_counter()
        circuit()
        return time.perf_counter() - start


class PyQPPEngine:
    def warmup(self, benchmark_id):
        self.run(benchmark_id, 2)

    def run(self, benchmark_id, n):
        if benchmark_id == "qft":
            return self._run_qft(n)
        raise ValueError()

    def _run_qft(self, n):
        import pyqpp

        qubits = [0] * n
        state = pyqpp.as_mutable(pyqpp.mket(qubits))
        gt = pyqpp.gates

        start = time.perf_counter()

        for i in range(n):
            pyqpp.apply_inplace(state, gt.H, [i])
            for j in range(2, n - i + 1):
                Rj = pyqpp.as_mutable([1, pyqpp.omega(2**j)])
                pyqpp.applyCTRL_diag_inplace(state, Rj, [i + j - 1], [i])

        return time.perf_counter() - start


class QPPEngine:
    # Map benchmark_id -> (executable_path, [extra_arguments])
    EXECUTABLES = {
        "qft": ("./cpp/build/qft_bench", ["--no-swaps"]),
        # "grover": ("./cpp/build/grover_bench", ["--target-state", "101"]),
        # "qpe": ("./cpp/build/qpe_bench", []), # Example with no extra args
    }

    def __init__(self, reps=1):
        self.reps = reps

    def warmup(self, benchmark_id):
        executable, extra_args = self._get_executable_info(benchmark_id)

        # Construct command: e.g., ["./cpp/build/qft_bench", "2", "--no-swaps"]
        cmd = [executable, "2"] + extra_args

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    def run(self, benchmark_id, n):
        executable, extra_args = self._get_executable_info(benchmark_id)

        samples = [self._run_once(executable, extra_args, n) for _ in range(self.reps)]
        return float(np.median(samples))

    def _get_executable_info(self, benchmark_id):
        try:
            return self.EXECUTABLES[benchmark_id]
        except KeyError:
            raise ValueError(f"Unsupported benchmark: {benchmark_id}")

    def _run_once(self, executable, extra_args, n):
        cmd = [executable, str(n)] + extra_args

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        match = re.search(
            r">>\s*Took:\s*([0-9.eE+-]+)\s*seconds",
            result.stdout,
        )

        if match is None:
            raise RuntimeError(f"Could not parse runtime from output:\n{result.stdout}")

        return float(match.group(1))


# =============================================================================
# SINGLE SOURCE OF TRUTH: BACKENDS
# =============================================================================


@dataclass(frozen=True)
class BackendSpec:
    name: str
    label: str
    marker: str
    factory: Callable[[], Any]


BACKENDS: Dict[str, BackendSpec] = {
    "qiskit": BackendSpec("qiskit", "Qiskit Aer", "o-", QiskitEngine),
    "cirq": BackendSpec("cirq", "Cirq", "^-", CirqEngine),
    "pennylane_default": BackendSpec(
        "pennylane_default",
        "PennyLane default.qubit",
        "s-",
        lambda: PennyLaneEngine("default.qubit"),
    ),
    "pennylane_lightning": BackendSpec(
        "pennylane_lightning",
        "PennyLane lightning.qubit",
        "d-",
        lambda: PennyLaneEngine("lightning.qubit"),
    ),
    "pyqpp": BackendSpec(
        "pyqpp",
        "qpp (Python bindings)",
        "*-",
        PyQPPEngine,
    ),
    "qpp": BackendSpec(
        "qpp",
        "qpp (C++ native)",
        "x-",
        QPPEngine,
    ),
}


# =============================================================================
# BENCHMARK
# =============================================================================


class QFTBenchmark:
    id = "qft"
    name = "QFT Simulation Benchmark"
    filename = "qft_benchmark.png"
    default_backends = ["pyqpp", "qiskit", "pennylane_lightning"]


BENCHMARKS = {
    "qft": QFTBenchmark,
}


# =============================================================================
# CSV & PLOTTING
# =============================================================================


def write_csv_wide(filename, qubits, times):
    header = ["qubits"] + sorted(times.keys())

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, n in enumerate(qubits):
            row = [n]
            for backend in sorted(times.keys()):
                row.append(times[backend][i])
            writer.writerow(row)


def plot(benchmark, qubits, times, scale="log"):
    plt.figure(figsize=(10, 6))

    for backend in sorted(times.keys()):
        spec = BACKENDS[backend]

        plt.plot(
            qubits,
            times[backend],
            spec.marker,
            label=spec.label,
        )

    plt.yscale(scale)
    plt.xlabel("Number of Qubits")
    plt.ylabel("Runtime (seconds)")
    plt.title(benchmark.name)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(benchmark.filename)


# =============================================================================
# ISOLATED EXECUTION WORKER
# =============================================================================


def _run_backend_batch(backend_name, benchmark_id, qubits):
    """
    Executes the entire sweep for a single backend inside ONE isolated process.
    Prints to the console in real-time as each qubit simulation finishes.
    """
    engine = BACKENDS[backend_name].factory()

    # Warmup once per process
    engine.warmup(benchmark_id)

    # Run all qubit sizes
    times = []
    for n in qubits:
        t = engine.run(benchmark_id, n)
        print(f"  Qubits = {n:02d} | Time: {t:.6f}s", flush=True)
        times.append(t)

    return times


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark", choices=BENCHMARKS.keys(), default="qft", help="Benchmark suite"
    )
    parser.add_argument(
        "--min-qubits", type=int, default=2, help="Minimum number of qubits"
    )
    parser.add_argument(
        "--max-qubits", type=int, default=20, help="Maximum number of qubits"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to run (use --list-backends to see available options, or 'all' to run all backends)",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit",
    )
    parser.add_argument(
        "--scale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scale type for runtime plot",
    )

    args = parser.parse_args()

    benchmark = BENCHMARKS[args.benchmark]()

    if args.list_backends:
        print("\nAvailable backends:")
        for k in sorted(BACKENDS):
            print(f"  {k:<25} {BACKENDS[k].label}")
        exit(0)

    # Resolve backends
    if args.backends is None:
        selected = benchmark.default_backends
    elif "all" in args.backends:
        if len(args.backends) > 1:
            raise ValueError("'all' cannot be combined with other backends")

        selected = list(BACKENDS.keys())
    else:
        selected = args.backends

    invalid = set(selected) - BACKENDS.keys()
    if invalid:
        raise ValueError(f"Invalid backends: {invalid}")

    # Setup
    qubits = list(range(args.min_qubits, args.max_qubits + 1))
    times = {b: [] for b in selected}

    print(f"\nRunning benchmark for {args.min_qubits} -> {args.max_qubits} qubits")

    # Run one isolated process PER BACKEND
    for b in selected:
        print(f"\n--- Running {b} ---")

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            # Submit the entire list of qubits to the worker
            batch_times = executor.submit(
                _run_backend_batch, b, benchmark.id, qubits
            ).result()

        times[b] = batch_times

    # Plot
    plot(benchmark, qubits, times, scale=args.scale)

    # CSV
    csv_file = benchmark.filename.replace(".png", ".csv")
    write_csv_wide(csv_file, qubits, times)

    print(f"\nSaved CSV → {csv_file}")
    print(f"Saved plot → {benchmark.filename}")
