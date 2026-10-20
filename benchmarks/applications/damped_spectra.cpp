// Native sparse complex shift-invert reference. No Python matvec callbacks.
#include <Eigen/SparseLU>
#include <Spectra/GenEigsSolver.h>
#include <complex>
#include <memory>
using Complex = std::complex<double>;
using Sparse = Eigen::SparseMatrix<Complex>;
using Vec = Eigen::VectorXcd;
struct Inverse {
    using Scalar = Complex;
    Sparse a;
    Eigen::SparseLU<Sparse> factor;
    Complex shift;
    int rows() const { return a.rows(); }
    int cols() const { return a.cols(); }
    void perform_op(const Complex* x, Complex* y) const {
        Eigen::Map<Vec>(y, rows()) = factor.solve(Eigen::Map<const Vec>(x, rows()));
    }
};
extern "C" void* damped_factor(int n, int nnz, const int* offsets, const int* columns,
                              const Complex* values, double sr, double si) {
    try {
        auto op = std::make_unique<Inverse>();
        op->a = Eigen::Map<const Sparse>(n,n,nnz,offsets,columns,values);
        op->shift = Complex(sr,si);
        Sparse identity(n,n); identity.setIdentity();
        op->factor.compute(op->a - op->shift*identity);
        if(op->factor.info() != Eigen::Success) return nullptr;
        return op.release();
    } catch (...) { return nullptr; }
}
extern "C" int damped_solve(void* ptr, int k, int ncv, const Complex* initial,
                           double tol, int budget, Complex* values, Complex* vectors,
                           int* statistics) {
    try {
        auto& op = *static_cast<Inverse*>(ptr);
        Spectra::GenEigsSolver<Inverse> solver(op,k,ncv);
        solver.init(initial);
        const int count = solver.compute(Spectra::SortRule::LargestMagn,budget,tol);
        statistics[0] = count;
        statistics[1] = solver.num_iterations();
        statistics[2] = solver.num_operations();
        statistics[3] = int(solver.info());
        const auto w = solver.eigenvalues();
        if(w.size()!=k || count!=k) return 1;
        Eigen::Map<Vec>(values,k) = (1.0/w.array() + op.shift).matrix();
        Eigen::Map<Eigen::MatrixXcd>(vectors,op.rows(),k) = solver.eigenvectors();
        return 0;
    } catch (...) { return -1; }
}
extern "C" void damped_free(void* ptr) { delete static_cast<Inverse*>(ptr); }
