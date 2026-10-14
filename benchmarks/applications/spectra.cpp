// Native CPU comparisons; Eigen factors and Spectra iteration stay in C++.
#include <Eigen/SparseLU>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <memory>
using Sparse = Eigen::SparseMatrix<double>;
using Vec = Eigen::VectorXd;
struct Context { Sparse a,b; Eigen::SparseLU<Sparse> factor; int mode; double sigma; };
struct Product {
    using Scalar=double; const Sparse& a;
    int rows() const { return a.rows(); } int cols() const { return a.cols(); }
    void perform_op(const double*x,double*y) const { Eigen::Map<Vec>(y,rows())=a*Eigen::Map<const Vec>(x,rows()); }
};
struct Mass : Product {
    Context& c;
    void solve(const double*x,double*y) const { Eigen::Map<Vec>(y,rows())=c.factor.solve(Eigen::Map<const Vec>(x,rows())); }
};
struct Inverse {
    using Scalar=double; Context& c;
    int rows() const { return c.a.rows(); } int cols() const { return c.a.cols(); }
    void set_shift(double sigma) { if(sigma!=c.sigma) throw std::runtime_error("shift mismatch"); }
    void perform_op(const double*x,double*y) const { Eigen::Map<Vec>(y,rows())=c.factor.solve(Eigen::Map<const Vec>(x,rows())); }
};
extern "C" void* app_factor(int n,int na,const int*ap,const int*ai,const double*av,
                            int nb,const int*bp,const int*bi,const double*bv,int mode,double sigma) {
    try {
        auto c=std::make_unique<Context>();
        c->a=Eigen::Map<const Sparse>(n,n,na,ap,ai,av);
        c->b=Eigen::Map<const Sparse>(n,n,nb,bp,bi,bv); c->mode=mode; c->sigma=sigma;
        Sparse shifted=mode==1 ? c->b : Sparse(c->a-sigma*c->b);
        c->factor.compute(shifted);
        if(c->factor.info()!=Eigen::Success)return nullptr;
        return c.release();
    } catch(...) {return nullptr;}
}
template<class Solver> int run(Solver& s,int n,int k,const double*initial,double*w,double*v) {
    s.init(initial);int count=s.compute(Spectra::SortRule::LargestMagn,10000,1e-11);
    if(count==k){Eigen::Map<Vec>(w,k)=s.eigenvalues();Eigen::Map<Eigen::MatrixXd>(v,n,k)=s.eigenvectors();}
    return count;
}
extern "C" int app_solve(void*ptr,int k,int ncv,const double*initial,double*w,double*v){
    try {
        auto& c=*static_cast<Context*>(ptr);Inverse inv{c};Product a{c.a},b{c.b};
        if(c.mode==1){Mass mass{{c.b},c};Spectra::SymGEigsSolver<Product,Mass,Spectra::GEigsMode::RegularInverse> s(a,mass,k,ncv);return run(s,c.a.rows(),k,initial,w,v);}
        if(c.mode==2){Spectra::SymGEigsShiftSolver<Inverse,Product,Spectra::GEigsMode::Buckling> s(inv,a,k,ncv,c.sigma);return run(s,c.a.rows(),k,initial,w,v);}
        Spectra::SymGEigsShiftSolver<Inverse,Product,Spectra::GEigsMode::ShiftInvert> s(inv,b,k,ncv,c.sigma);return run(s,c.a.rows(),k,initial,w,v);
    }catch(...){return -1;}
}
extern "C" void app_free(void*ptr){delete static_cast<Context*>(ptr);}
