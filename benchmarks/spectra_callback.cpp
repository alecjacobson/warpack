// Reference-only bridge: run Spectra with the CPU SuperLU operator used by ARPACK.
// No code in this file is used by warpack's numerical implementation.
#include <Spectra/SymEigsSolver.h>
#include <Eigen/Core>
#include <exception>
using Callback = void (*)(const double*, double*);
struct Operator {
    using Scalar = double;
    int n;
    Callback callback;
    int rows() const { return n; }
    int cols() const { return n; }
    void perform_op(const double* x, double* y) const { callback(x, y); }
};
extern "C" int spectra_inverse(int n, int k, int ncv, const double* initial,
                               Callback callback, double* values, double* vectors) {
    try {
        Operator op{n, callback};
        Spectra::SymEigsSolver<Operator> solver(op, k, ncv);
        solver.init(initial);
        int converged = solver.compute(Spectra::SortRule::LargestAlge, 10000, 1e-12);
        if (converged != k) return converged;
        Eigen::Map<Eigen::VectorXd>(values, k) = solver.eigenvalues();
        Eigen::Map<Eigen::MatrixXd>(vectors, n, k) = solver.eigenvectors();
        return converged;
    } catch (const std::exception&) { return -1; }
}
