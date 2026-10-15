// Native Spectra on the symmetric, mass-scaled physical KKT inverse.
#include <Eigen/SparseLU>
#include <Spectra/SymEigsSolver.h>
#include <memory>
using Sparse = Eigen::SparseMatrix<double>;
using Vec = Eigen::VectorXd;
struct Skinning {
    using Scalar = double;
    int n, size;
    Vec scale;
    Eigen::SparseLU<Sparse> factor;
    int rows() const { return n; }
    int cols() const { return n; }
    void perform_op(const double* x, double* y) const {
        Vec rhs = Vec::Zero(size);
        rhs.head(n) = scale.array() * Eigen::Map<const Vec>(x, n).array();
        Vec result = factor.solve(rhs);
        Eigen::Map<Vec>(y, n) = scale.array() * result.head(n).array();
    }
};
extern "C" void* skinning_factor(int n, int size, int nnz, const int* offsets,
                                 const int* columns, const double* values, const double* mass) {
    try {
        auto c = std::make_unique<Skinning>();
        c->n = n; c->size = size;
        c->scale = Eigen::Map<const Vec>(mass, n).array().sqrt();
        Sparse a = Eigen::Map<const Sparse>(size, size, nnz, offsets, columns, values);
        c->factor.compute(a);
        if(c->factor.info() != Eigen::Success) return nullptr;
        return c.release();
    } catch(...) { return nullptr; }
}
extern "C" int skinning_solve(void* ptr, int k, int ncv, const double* initial,
                              double shift, double* values, double* vectors) {
    try {
        auto& c = *static_cast<Skinning*>(ptr);
        Spectra::SymEigsSolver<Skinning> solver(c, k, ncv);
        solver.init(initial);
        int count = solver.compute(Spectra::SortRule::LargestMagn, 10000, 1e-11);
        if(count == k) {
            Eigen::Map<Vec>(values, k) = (1 / solver.eigenvalues().array() + shift).matrix();
            Eigen::Map<Eigen::MatrixXd>(vectors, c.n, k) = c.scale.cwiseInverse().asDiagonal() * solver.eigenvectors();
        }
        return count;
    } catch(...) { return -1; }
}
extern "C" void skinning_free(void* ptr) { delete static_cast<Skinning*>(ptr); }
