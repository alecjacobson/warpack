// Accuracy-only native Spectra boundary; no Python matvec callbacks.
#include <Eigen/Core>
#include <Spectra/GenEigsSolver.h>
#include <complex>

extern "C" int general_eigs(int n, int k, int ncv, const double* a,
                            const double* initial, double tol, int budget,
                            std::complex<double>* w, std::complex<double>* v,
                            int* stats) {
    try {
        using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::MatrixXd matrix = Eigen::Map<const RowMatrix>(a, n, n);
        Spectra::DenseGenMatProd<double> op(matrix);
        Spectra::GenEigsSolver<Spectra::DenseGenMatProd<double>> solver(op, k, ncv);
        solver.init(initial);
        stats[0] = solver.compute(Spectra::SortRule::LargestMagn, budget, tol);
        stats[1] = solver.num_iterations();
        stats[2] = solver.num_operations();
        stats[3] = int(solver.info());
        const auto values = solver.eigenvalues();
        const auto vectors = solver.eigenvectors();
        stats[4] = values.size();
        if (values.size() > k + 2) return -2;
        Eigen::Map<Eigen::VectorXcd>(w, values.size()) = values;
        Eigen::Map<Eigen::MatrixXcd>(v, n, values.size()) = vectors;
        return 0;
    } catch (...) { return -1; }
}
