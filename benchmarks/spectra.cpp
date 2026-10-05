#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){
 if(argc<4 || argc>5) return 2;
 Eigen::SparseMatrix<double> A;
 if(!Eigen::loadMarket(A,argv[1])) return 3;
 int k=std::stoi(argv[2]); bool inverse=std::string(argv[3])=="SA";
 int ncv=std::min(int(A.rows()),std::max(2*k+1,48));
 Eigen::VectorXd initial;
 if(argc==5){ initial.resize(A.rows()); std::ifstream f(argv[4],std::ios::binary); f.read(reinterpret_cast<char*>(initial.data()),initial.size()*sizeof(double)); if(!f) return 5; }
 auto t0=Clock::now(); Eigen::VectorXd w; Eigen::MatrixXd v; int nc=0; double setup=0;
 if(inverse){
  Spectra::SparseSymShiftSolve<double> op(A);
  Spectra::SymEigsShiftSolver<decltype(op)> sol(op,k,ncv,0.);
  auto t1=Clock::now(); setup=std::chrono::duration<double>(t1-t0).count();
  if(initial.size()) sol.init(initial.data()); else sol.init(); nc=sol.compute(Spectra::SortRule::LargestMagn,10000,1e-11);
  w=sol.eigenvalues();v=sol.eigenvectors();
 }else{
  Spectra::SparseSymMatProd<double> op(A);
  Spectra::SymEigsSolver<decltype(op)> sol(op,k,ncv);
  if(initial.size()) sol.init(initial.data()); else sol.init();nc=sol.compute(Spectra::SortRule::LargestMagn,10000,1e-11);
  w=sol.eigenvalues();v=sol.eigenvectors();
 }
 double elapsed=std::chrono::duration<double>(Clock::now()-t0).count();
 double residual=0,relative=0;
 for(int i=0;i<w.size();i++) { double r=(A*v.col(i)-w[i]*v.col(i)).norm(); residual=std::max(residual,r);relative=std::max(relative,r/std::max(1.,std::abs(w[i]))); }
 std::cout<<std::setprecision(16)<<"{\"method\":\"Spectra\",\"seconds\":"<<elapsed<<",\"setup_seconds\":"<<setup<<",\"residual\":"<<residual<<",\"relative_residual\":"<<relative<<",\"converged\":"<<nc<<"}\n";
 return nc==k?0:4;
}
