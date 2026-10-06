// Native Spectra + Eigen sparse LDLT reference; no Python matvec callbacks.
#include <Eigen/SparseCholesky>
#include <Spectra/SymEigsSolver.h>
#include <memory>
using Sparse = Eigen::SparseMatrix<double>;
struct Context {
    Sparse matrix;
    Eigen::SimplicialLDLT<Sparse> factor;
    int calls = 0;
};
struct Inverse {
    using Scalar = double;
    Context& ctx;
    int rows() const { return ctx.matrix.rows(); }
    int cols() const { return ctx.matrix.cols(); }
    void perform_op(const double* x, double* y) const {
        ++ctx.calls;
        Eigen::Map<Eigen::VectorXd>(y, rows()) =
            ctx.factor.solve(Eigen::Map<const Eigen::VectorXd>(x, rows()));
    }
};
extern "C" void* spectra_factor(int n, int nnz, const int* outer,
                                const int* inner, const double* values) {
    try {
        auto ctx = std::make_unique<Context>();
        ctx->matrix = Eigen::Map<const Sparse>(n, n, nnz, outer, inner, values);
        ctx->factor.compute(ctx->matrix);
        if (ctx->factor.info() != Eigen::Success) return nullptr;
        return ctx.release();
    } catch (...) { return nullptr; }
}
extern "C" int spectra_solve(void* ptr, int k, int ncv, double tol,
                             const double* initial, double* values,
                             double* vectors, int* calls) {
    try {
        auto& ctx = *static_cast<Context*>(ptr);
        ctx.calls = 0;
        Inverse op{ctx};
        Spectra::SymEigsSolver<Inverse> solver(op, k, ncv);
        solver.init(initial);
        int count = solver.compute(Spectra::SortRule::LargestAlge, 10000, tol);
        *calls = ctx.calls;
        if (count == k) {
            Eigen::Map<Eigen::VectorXd>(values, k) = solver.eigenvalues();
            Eigen::Map<Eigen::MatrixXd>(vectors, ctx.matrix.rows(), k) = solver.eigenvectors();
        }
        return count;
    } catch (...) { return -1; }
}
extern "C" void spectra_free(void* ptr) { delete static_cast<Context*>(ptr); }
