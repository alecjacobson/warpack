# Tridiagonal QL experiment (#161)

Two benchmark-only FP64 implicit QL layouts pass all 50 tests on pinned Warp
1.18 and stock Warp 1.15. Coverage spans dimensions 1–512, indefinite,
repeated and tightly clustered spectra, scales 1e-80..1e80, strided inputs,
changed-input graph replay, nonfinite recovery and explicit iteration-budget
failure. Independent equations, full spectra and orthogonality are checked.
The mathematical decomposition is described by
[LAPACK DSTEQR](https://netlib.org/lapack/explore-html/d3/d16/group__steqr_ga889db27ed60b61d3855c5bf5a0853a68.html);
the implementation uses Warp only, with no vendor eigensolver call.

The first layout replicates scalar QL state per group of 64 eigenvector rows.
A second repeats the recurrence in every lane, using compiler-local fixed-size
vectors. Both are much slower than global Jacobi even on inputs already in
tridiagonal form. Serial recurrences remain expensive on this L40; the row
layout also incurs local storage and duplicated arithmetic. Neither is promoted.

Uncontended component medians in milliseconds, three replays plus validated
warmup. Exact input identities match within each comparison. Device reset,
extraction and output storage are included; dense reduction and eigenvector
back-transformation are **not implemented or included**. No full-FEM claim.

| Size | Global Jacobi (grouped control) | Grouped QL | Global Jacobi (row control) | Row QL |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 1.252 | 4.595 | 1.253 | 4.354 |
| 72 | 1.870 | 10.279 | 1.868 | 9.975 |
| 104 | 3.231 | 20.768 | 3.247 | 21.403 |
| 128 | 3.573 | 31.388 | 3.572 | 34.098 |
| 208 | 7.689 | 82.368 | 7.732 | 93.258 |
| 312 | 15.876 | 187.841 | 16.015 | 208.947 |
| 512 | 33.431 | 500.949 | 33.987 | 559.055 |

Every measured/warmup result passes independent acceptance. Graphs contain no
host callback or host transfer. Raw reports and occupancy are retained under
[component](component/jacobi-a1.json) and
[row-component](row-component/jacobi-a1.json). [Pinned tests](tests.log) and
[stock tests](stock-tests.log) pass. The QL iteration field counts total implicit
steps, whereas the Jacobi field counts full sweeps; they are different work units.

There is no reason to add dense reduction around either losing component.
Close this implementation experiment without a production change; #147 remains
open. This does not rule out a different parallel tridiagonal algorithm.
