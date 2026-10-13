"""Instrument Spectra's original 40 datasets x 10 trials; keep native ARPACK too."""

import json
import os
import subprocess

from .prepare_spectra import HEADER, ROOT, UP

OUT = ROOT / "build/spectra-benchmark-fixtures"


def record_native(log):
    rows = []
    family = None
    for line in log.splitlines():
        fields = line.split()
        if fields in (["eigs_sym"], ["eigs_gen"]):
            family = fields[0]
        if len(fields) == 8 and fields[0].isdigit():
            rows.append(
                dict(
                    family=family,
                    n=int(fields[0]),
                    dataset=int(fields[1]),
                    arpack_ms=float(fields[2]),
                    arpack_residual=float(fields[3]),
                    spectra_ms=float(fields[5]),
                    spectra_residual=float(fields[6]),
                )
            )
    assert len(rows) == 400, f"Expected 400 native trials, got {len(rows)}"
    path = ROOT / "results/upstream_provenance.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["native_spectra_benchmark"] = dict(
        protocol="Unchanged upstream main: 10 datasets x 10 repetitions x 2 sizes x 2 eigenproblem families; exported same matrices and v0 for Warp/CuPy",
        rows=rows,
    )
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    header = HEADER.replace('#include "catch.hpp"', "").replace(
        "Catch::getResultCapture().getCurrentTestName()", 'std::string("upstream benchmark")'
    )
    (OUT / "export.h").write_text(header)
    source = (
        '''#include "export.h"
    #define eigs_sym_Cpp native_sym
    #define eigs_gen_Cpp native_gen
    #include "'''
        + str(UP / "benchmark/Cpp.cpp")
        + '''"
    #undef eigs_sym_Cpp
    #undef eigs_gen_Cpp
    #include "'''
        + str(UP / "benchmark/main.cpp")
        + """"
    void eigs_sym_Cpp(MatrixXd& A,VectorXd& v,int k,int m,double& t,double& e,int& ops) {
     static int calls=0; if(calls++%10==0) {fixture_family="benchmark_sym";fixture_k=k;fixture_m=m;fixture(A,MatrixXd(),0,true,false,v);}
     native_sym(A,v,k,m,t,e,ops);
    }
    void eigs_gen_Cpp(MatrixXd& A,VectorXd& v,int k,int m,double& t,double& e,int& ops) {
     static int calls=0; if(calls++%10==0) {fixture_family="benchmark_gen";fixture_k=k;fixture_m=m;fixture(A,MatrixXd(),0,false,false,v);}
     native_gen(A,v,k,m,t,e,ops);
    }
    """
    )
    (OUT / "main.cpp").write_text(source)
    for p in OUT.glob("*.jsonl"):
        p.unlink()
    subprocess.run(
        ["gfortran", "-O3", "-c", str(UP / "benchmark/wrapper.f"), "-o", str(OUT / "wrapper.o")],
        check=True,
    )
    subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++11",
            "-I/usr/include/eigen3",
            "-I" + str(UP / "include"),
            '-DFAMILY="benchmark"',
            str(OUT / "main.cpp"),
            str(UP / "benchmark/F77.cpp"),
            str(OUT / "wrapper.o"),
            "-l:libarpack.so.2",
            "-llapack",
            "-lblas",
            "-lgfortran",
            "-o",
            str(OUT / "benchmark"),
        ],
        check=True,
    )
    with (ROOT / "build/upstream_spectra_benchmark_native.log").open("w") as f:
        subprocess.run(
            [str(OUT / "benchmark")],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=dict(
                os.environ, SPECTRA_FIXTURES=str(OUT), OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4"
            ),
            check=True,
        )
    record_native((ROOT / "build/upstream_spectra_benchmark_native.log").read_text())
    print(
        "Exported and benchmarked",
        sum(len(p.read_text().splitlines()) for p in OUT.glob("*.jsonl")),
        "datasets",
    )


if __name__ == "__main__":
    main()
