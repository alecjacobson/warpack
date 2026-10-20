// Native complex Spectra Hermitian engine with an explicit B-metric adapter.
// HermEigsBase is an internal engine, not a public HermGEigsSolver convenience API.
#include <Eigen/SparseCholesky>
#include <Spectra/HermEigsBase.h>
#include <complex>
#include <memory>
using Complex = std::complex<double>;
using Sparse = Eigen::SparseMatrix<Complex>;
using Vec = Eigen::VectorXcd;
struct Mass {
    using Scalar = Complex;
    Sparse matrix;
    int rows() const { return matrix.rows(); }
    int cols() const { return matrix.cols(); }
    void perform_op(const Complex* x, Complex* y) const {
        Eigen::Map<Vec>(y, rows()) = matrix * Eigen::Map<const Vec>(x, rows());
    }
};
struct Connection {
    using Scalar = Complex;
    Mass mass;
    double shift;
    Eigen::SimplicialLDLT<Sparse> factor;
    int rows() const { return mass.rows(); }
    int cols() const { return mass.cols(); }
    void perform_op(const Complex* x, Complex* y) const {
        Vec rhs = mass.matrix * Eigen::Map<const Vec>(x, rows());
        Eigen::Map<Vec>(y, rows()) = factor.solve(rhs);
    }
};
extern "C" void* connection_factor(int n, int nnz, const int* offsets,
    const int* columns, const Complex* values, int mnz, const int* moffsets,
    const int* mcolumns, const Complex* mvalues, double shift) {
    try {
        auto c = std::make_unique<Connection>();
        c->shift = shift;
        c->mass.matrix = Eigen::Map<const Sparse>(n,n,mnz,moffsets,mcolumns,mvalues);
        Sparse a = Eigen::Map<const Sparse>(n,n,nnz,offsets,columns,values);
        c->factor.compute(a - shift * c->mass.matrix);
        if(c->factor.info() != Eigen::Success || c->factor.vectorD().real().minCoeff() <= 0)
            return nullptr;
        return c.release();
    } catch(...) { return nullptr; }
}
extern "C" int connection_solve(void* ptr, int k, int ncv, const Complex* initial,
                                double* values, Complex* vectors) {
    try {
        auto& c = *static_cast<Connection*>(ptr);
        Spectra::HermEigsBase<Connection, Mass> solver(c, c.mass, k, ncv);
        solver.init(initial);
        int count = solver.compute(Spectra::SortRule::LargestMagn, 10000, 1e-11);
        if(count == k) {
            Eigen::Map<Eigen::VectorXd>(values,k) = (1/solver.eigenvalues().array()+c.shift).matrix();
            Eigen::Map<Eigen::MatrixXcd>(vectors,c.rows(),k) = solver.eigenvectors();
        }
        return count;
    } catch(...) { return -1; }
}
extern "C" void connection_free(void* ptr) { delete static_cast<Connection*>(ptr); }
