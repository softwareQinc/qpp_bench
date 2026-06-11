// Quantum Fourier transform benchmark

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <qpp/qpp.hpp>

#define CHECKS false
#define KETS true

#if KETS
#define FUN(x) mket((x))
#else
#define FUN(x) mprj((x))
#endif

int main(int argc, char** argv) {
    using namespace qpp;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <nq> [--no-swaps]\n";
        exit(EXIT_FAILURE);
    }

    bool perform_swaps = true;

    for (int i = 2; i < argc; ++i) {
        if (std::string{argv[i]} == "--no-swaps") {
            perform_swaps = false;
        }
    }

    std::vector<idx> qubits(std::atoi(argv[1]), 1);
    auto state = FUN(qubits);
    auto state_0 = FUN(qubits);

    idx n = qubits.size();
    [[maybe_unused]] auto D = static_cast<idx>(std::llround(std::pow(2, n)));

    Timer t;
    cmat Rj(2, 1);
    for (idx i = 0; i < n; ++i) {
        apply_inplace(state, gt.H, {i});
        for (idx j = 2; j <= n - i; ++j) {
            auto pow_j = static_cast<idx>(std::llround(std::pow(2, j)));
            Rj << 1, omega(pow_j);
            applyCTRL_diag_inplace(state, Rj, {i + j - 1}, {i});
        }
    }

    if (perform_swaps) {
        for (idx i = 0; i < n / 2; ++i) {
            apply_inplace(state, gt.SWAP, {i, static_cast<idx>(n - i - 1)});
        }
    }

    std::cout << "\n>> Took: " << t.toc() << " seconds\n";

#if CHECKS
    if (perform_swaps && n < 14) {
        auto norm_diff =
#if KETS
            norm(state - gt.Fd(D) * state_0);
#else
            norm(state - gt.Fd(D) * state_0 * gt.Fd(D).adjoint());
#endif
        std::cout << ">> Norm difference: " << norm_diff << '\n';
    }
#endif
}
