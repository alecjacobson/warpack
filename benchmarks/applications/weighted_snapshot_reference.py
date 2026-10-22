"""CPU validation only: weighted SVD without squaring snapshot conditioning.

Large diagonal mass stays sparse. A coupled mass uses a bounded-size dense
Cholesky reference; this is deliberately not a production mass square root.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


def weighted_reference(snapshots, mass, k, *, rank_tol=1e-12, dense_limit=2048):
    snapshots = np.asarray(snapshots, dtype=np.float64)
    mass = sp.csr_matrix(mass, dtype=np.float64)
    if (
        snapshots.ndim != 2
        or mass.shape != (snapshots.shape[0], snapshots.shape[0])
        or not 1 <= k <= min(snapshots.shape)
        or not np.isfinite(snapshots).all()
        or not np.isfinite(mass.data).all()
        or not 0 < rank_tol < 1
    ):
        raise ValueError("finite snapshots, matching SPD mass and valid mode/rank bounds required")
    difference = mass - mass.T
    scale = abs(mass.data).max(initial=0)
    if abs(difference.data).max(initial=0) > 1e-13 * scale:
        raise ValueError("mass must be symmetric")
    diagonal = mass.diagonal()
    off_diagonal = mass - sp.diags(diagonal)
    off_diagonal.eliminate_zeros()
    if off_diagonal.nnz == 0:
        if np.any(diagonal <= 0):
            raise ValueError("mass must be positive definite")
        root = np.sqrt(diagonal)
        weighted = root[:, None] * snapshots
        method = "diagonal-mass weighted SVD"
    else:
        if mass.shape[0] > dense_limit:
            raise ValueError("coupled-mass dense reference exceeds explicit size limit")
        try:
            root = la.cholesky(mass.toarray(), lower=True)
        except la.LinAlgError as error:
            raise ValueError("mass must be positive definite") from error
        weighted = root.T @ snapshots
        method = "dense-Cholesky weighted SVD reference"
    left, singular, right = la.svd(weighted, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * rank_tol))
    basis = None
    if rank >= k:
        basis = (
            left[:, :k] / root[:, None]
            if root.ndim == 1
            else la.solve_triangular(root.T, left[:, :k], lower=False)
        )
    return dict(
        values=singular[:k] ** 2,
        singular_values=singular,
        right_vectors=right[:k].T,
        basis=basis,
        energy=float(singular @ singular),
        rank=rank,
        rank_tol=rank_tol,
        rank_policy="relative singular-value cutoff, not covariance-eigenvalue cutoff",
        method=method,
    )
