"""Export exact pinned Spectra fixtures without rewriting their RNG/generators.

Only test execution bodies are replaced in a generated build-directory copy.
The original sources and their assertions also run separately through CTest.
"""

import concurrent.futures
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "build/upstream-spectra"
OUT = ROOT / "build/spectra-fixtures"
FAMILIES = [
    "SymEigs",
    "HermEigs",
    "GenEigs",
    "ComplexEigs",
    "SymEigsShift",
    "GenEigsRealShift",
    "GenEigsComplexShift",
    "SymGEigsCholesky",
    "SymGEigsRegInv",
    "SymGEigsShift",
    "SVD",
    "DavidsonSymEigs",
    "Example1",
    "Example2",
    "Example3",
    "Example4",
]
HEADER = r"""
#include <fstream>
#include <iomanip>
#include <complex>
#include <cstdlib>
#include <Eigen/Core>
#include "catch.hpp"
static int fixture_k=0, fixture_m=0, fixture_id=0;
static double fixture_sr=0, fixture_si=0;
static std::string fixture_family=FAMILY;
template<class T> void save_matrix(const T& a, const std::string& path, bool symmetric=false) {
    Eigen::MatrixXcd d=Eigen::Matrix<typename T::Scalar,Eigen::Dynamic,Eigen::Dynamic>(a).template cast<std::complex<double>>();
    if(symmetric) d=Eigen::MatrixXcd(d.selfadjointView<Eigen::Lower>());
    std::ofstream out(path,std::ios::binary);
    int64_t shape[2]={d.rows(),d.cols()}; out.write((char*)shape,sizeof(shape));
    for(int i=0;i<d.rows();++i) for(int j=0;j<d.cols();++j) {
        double pair[2]={d(i,j).real(),d(i,j).imag()}; out.write((char*)pair,sizeof(pair));
    }
}
template<class A,class B> void fixture(const A& a,const B& b,int selection,bool symmetric,
                                     bool allowed=false,const Eigen::VectorXd& v0=Eigen::VectorXd()) {
    std::string id=fixture_family+"_"+std::to_string(fixture_id++);
    std::string base=std::string(std::getenv("SPECTRA_FIXTURES"))+"/"+id;
    save_matrix(a,base+".a",symmetric);
    if(b.size()) save_matrix(b,base+".b",true);
    if(v0.size()) save_matrix(v0,base+".v0");
    std::ofstream out(std::string(std::getenv("SPECTRA_FIXTURES"))+"/"+fixture_family+".jsonl",std::ios::app);
    out<<std::setprecision(17)<<"{\"id\":\""<<id<<"\",\"family\":\""<<fixture_family
       <<"\",\"name\":\""<<Catch::getResultCapture().getCurrentTestName()
       <<"\",\"k\":"<<fixture_k<<",\"ncv\":"<<fixture_m<<",\"selection\":"<<selection
       <<",\"sigma_real\":"<<fixture_sr<<",\"sigma_imag\":"<<fixture_si
       <<",\"upstream_allows_nonconvergence\":"<<(allowed?"true":"false")<<"}\n";
}
"""


def replace_body(s, signature, body):
    m = re.search(signature, s)
    assert m, signature
    start = s.index("{", m.end())
    depth = 1
    end = start + 1
    while depth:
        depth += (s[end] == "{") - (s[end] == "}")
        end += 1
    return s[:start] + "{\n" + body + "\n}" + s[end:]


def build(family):
    original = (UP / "test" / f"{family}.cpp").read_text()
    s = original
    empty = "Eigen::MatrixXd()"
    sym = family.startswith("Sym") or family in ("HermEigs", "DavidsonSymEigs")
    if family in ("Example1", "Example2", "Example3", "Example4"):
        bodies = {
            "Example1": "fixture_k=k;fixture_m=m;auto A=construct_cycle_laplacian(n);fixture(A,"
            + empty
            + ',0,true);fixture_sr=-1e-6;fixture_family="Example1Shift";fixture(A,'
            + empty
            + ',0,true);fixture_family="Example1";fixture_sr=0;',
            "Example2": "fixture_k=1;fixture_m=3;fixture(M," + empty + ",0,true);",
            "Example3": "fixture_k=nef;fixture_m=ncv;SpMat B=C+shift*M;fixture(M,B,0,true);",
            "Example4": "fixture_k=3;fixture_m=6;fixture(A," + empty + ",3,true,false,v0);",
        }
        s = replace_body(s, r"void run_test\([^\{]+\)", bodies[family])
    elif family == "SVD":
        s = replace_body(
            s,
            r"void run_test\([^\{]+\)",
            "fixture_k=k;fixture_m=m;fixture(mat," + empty + ",0,false);",
        )
    elif family == "DavidsonSymEigs":
        s = replace_body(
            s,
            r"void run_test\([^\{]+\)",
            "fixture_k=nev;fixture_m=0;fixture(mat," + empty + ",int(selection),true);",
        )
    else:
        has_b = family.startswith("SymGEigs")
        arg = (
            "A, B"
            if has_b
            else (
                "mat, " + empty
                if family
                in (
                    "SymEigs",
                    "HermEigs",
                    "SymEigsShift",
                    "GenEigs",
                    "ComplexEigs",
                    "GenEigsRealShift",
                    "GenEigsComplexShift",
                )
                else ""
            )
        )
        body = (
            "fixture("
            + arg
            + ",int(selection),"
            + str(sym).lower()
            + (
                "," + "allow_fail"
                if "allow_fail" in re.search(r"void run_test\([^\{]+\)", s).group()
                else ""
            )
            + ");"
        )
        s = replace_body(s, r"void run_test\([^\{]+\)", body)
        # Capture original constructor parameters at their declaration site.
        pattern = r"(^\s+[^\n;]+\beigs\(op, (?:Bop, )?k, m(?:, (?:sigma|sigmar, sigmai))?\);)"

        def before(m):
            line = m.group(1)
            params = "fixture_k=k;fixture_m=m;"
            if "sigmar" in line:
                params += "fixture_sr=sigmar;fixture_si=sigmai;"
            elif ", sigma)" in line:
                params += "fixture_sr=sigma;"
            return "\n" + params + line

        s, n = re.subn(pattern, before, s, flags=re.M)
        assert n, (family, "constructor")
    src = OUT / f"{family}.cpp"
    src.write_text('#include "export.h"\n' + s)
    exe = OUT / family
    subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++11",
            "-I/usr/include/eigen3",
            "-I" + str(UP / "include"),
            "-I" + str(UP / "test"),
            '-DFAMILY="' + family + '"',
            str(src),
            str(ROOT / "build/upstream-spectra-build/test/libtests-main.a"),
            "-o",
            str(exe),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    import os

    env = dict(os.environ, SPECTRA_FIXTURES=str(OUT))
    proc = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=120)
    (OUT / f"{family}.log").write_text(proc.stdout + proc.stderr)
    return dict(
        family=family,
        source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        exit_code=proc.returncode,
    )


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "export.h").write_text(HEADER)
    for p in OUT.glob("*.jsonl"):
        p.unlink()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(build, FAMILIES))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Exported", sum(len(p.read_text().splitlines()) for p in OUT.glob("*.jsonl")), "cases")


if __name__ == "__main__":
    main()
