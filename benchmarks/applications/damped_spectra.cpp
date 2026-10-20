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
    Sparse a, b, shifted;
    int refinement_steps;
    Eigen::SparseLU<Sparse> factor;
    Complex shift;
    int rows() const { return a.rows(); }
    int cols() const { return a.cols(); }
    void perform_op(const Complex* x, Complex* y) const {
        Vec rhs = Eigen::Map<const Vec>(x, rows());
        if(b.rows()) rhs = b*rhs;
        Vec solution = factor.solve(rhs);
        for(int i=0;i<refinement_steps;i++) {
            const Vec residual = rhs - shifted*solution;
            solution += factor.solve(residual);
        }
        Eigen::Map<Vec>(y, rows()) = solution;
    }
};
extern "C" void* damped_factor(int n, int nnz, const int* offsets, const int* columns,
                              const Complex* values, double sr, double si, int refinements) {
    try {
        auto op = std::make_unique<Inverse>();
        op->a = Eigen::Map<const Sparse>(n,n,nnz,offsets,columns,values);
        op->shift = Complex(sr,si);
        Sparse identity(n,n); identity.setIdentity();
        op->shifted = op->a - op->shift*identity;
        op->refinement_steps = refinements;
        op->factor.compute(op->shifted);
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

extern "C" int damped_polish(void* ptr, int k, Complex* vectors) {
    try {
        auto& op = *static_cast<Inverse*>(ptr);
        Eigen::Map<Eigen::MatrixXcd> input(vectors,op.rows(),k);
        Eigen::MatrixXcd result(op.rows(),k);
        for(int j=0;j<k;j++) op.perform_op(input.col(j).data(),result.col(j).data());
        input = result;
        return op.factor.info()==Eigen::Success ? 0 : -1;
    } catch (...) { return -1; }
}

extern "C" int damped_abi_version() { return 2; }

// Additional pencil entry point; the original standardized ABI is unchanged.
extern "C" int damped_pencil_abi_version() { return 1; }
extern "C" void* damped_pencil_factor(
    int n, int annz, const int* ao, const int* ac, const Complex* av,
    int bnnz, const int* bo, const int* bc, const Complex* bv,
    double sr, double si, int refinements) {
    try {
        auto op = std::make_unique<Inverse>();
        op->a = Eigen::Map<const Sparse>(n,n,annz,ao,ac,av);
        op->b = Eigen::Map<const Sparse>(n,n,bnnz,bo,bc,bv);
        op->shift = Complex(sr,si);
        op->shifted = op->a - op->shift*op->b;
        op->refinement_steps = refinements;
        op->factor.compute(op->shifted);
        if(op->factor.info()!=Eigen::Success) return nullptr;
        return op.release();
    } catch(...) { return nullptr; }
}
