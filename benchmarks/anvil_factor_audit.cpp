// Separate the phases of the Eigen factorization used in spectra_native.cpp.
#include <Eigen/SparseCholesky>
#ifdef USE_PARDISO
#include <Eigen/PardisoSupport>
#endif
#ifdef AUDIT_SPECTRA
#include <Spectra/SymEigsSolver.h>
#endif
#include <chrono>
#include <ctime>
#include <cstdio>

using Sparse = Eigen::SparseMatrix<double>;
using Clock = std::chrono::steady_clock;
static int refinement_steps = -1;
extern "C" void audit_configure(int steps) { refinement_steps = steps; }
struct Context {
    Sparse matrix;
#ifdef USE_PARDISO
    Eigen::PardisoLLT<Sparse> factor;
#else
    Eigen::SimplicialLDLT<Sparse> factor;
#endif
};
extern "C" void* factor_audit(int n, int nnz, const int* outer,
                              const int* inner, const double* values, double* stats) {
    auto* ctx = new Context;
#ifdef USE_PARDISO
    if (refinement_steps >= 0) ctx->factor.pardisoParameterArray()[7] = refinement_steps;
#endif
    auto start = Clock::now();
    ctx->matrix = Eigen::Map<const Sparse>(n, n, nnz, outer, inner, values);
    stats[0] = std::chrono::duration<double>(Clock::now() - start).count();
    start = Clock::now();
    ctx->factor.analyzePattern(ctx->matrix);
    stats[1] = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("Eigen copy %.6f s, symbolic %.6f s; starting numeric\n", stats[0], stats[1]);
    std::fflush(stdout);
    start = Clock::now();
    auto cpu = std::clock();
    ctx->factor.factorize(ctx->matrix);
    stats[2] = std::chrono::duration<double>(Clock::now() - start).count();
    stats[3] = double(std::clock() - cpu) / CLOCKS_PER_SEC;
#ifdef USE_PARDISO
    stats[4] = ctx->factor.pardisoParameterArray()[17];
    stats[5] = ctx->factor.pardisoParameterArray()[18];
#else
    stats[4] = ctx->factor.matrixL().nestedExpression().nonZeros();
    stats[5] = ctx->factor.vectorD().minCoeff();
#endif
    if (ctx->factor.info() != Eigen::Success) { delete ctx; return nullptr; }
    return ctx;
}
extern "C" void audit_solve(void* ptr, const double* rhs, double* out) {
    auto& ctx = *static_cast<Context*>(ptr);
    Eigen::Map<Eigen::VectorXd>(out, ctx.matrix.rows()) =
        ctx.factor.solve(Eigen::Map<const Eigen::VectorXd>(rhs, ctx.matrix.rows()));
}
extern "C" void audit_free(void* ptr) { delete static_cast<Context*>(ptr); }

#ifdef AUDIT_SPECTRA
struct Inverse {
    using Scalar = double;
    Context& ctx;
    int& calls;
    int rows() const { return ctx.matrix.rows(); }
    int cols() const { return ctx.matrix.cols(); }
    void perform_op(const double* x, double* y) const {
        ++calls;
        Eigen::Map<Eigen::VectorXd>(y, rows()) =
            ctx.factor.solve(Eigen::Map<const Eigen::VectorXd>(x, rows()));
    }
};
extern "C" int audit_spectra(void* ptr, double tol, const double* initial,
                             double* values, double* vectors, int* calls) {
    auto& ctx = *static_cast<Context*>(ptr);
    *calls = 0;
    Inverse op{ctx, *calls};
    Spectra::SymEigsSolver<Inverse> solver(op, 26, 64);
    solver.init(initial);
    int count = solver.compute(Spectra::SortRule::LargestAlge, 10000, tol);
    if (count == 26) {
        Eigen::Map<Eigen::VectorXd>(values, 26) = solver.eigenvalues();
        Eigen::Map<Eigen::MatrixXd>(vectors, ctx.matrix.rows(), 26) = solver.eigenvectors();
    }
    return count;
}
#endif
