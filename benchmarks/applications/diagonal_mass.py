"""Explicit diagonal-mass standard reformulation for comparisons, not core API."""

import warp as wp
import warp.sparse as ws

from warpack import KrylovSchur, SparseOperator


@wp.kernel
def reciprocal_sqrt(diagonal: wp.array[wp.float64], scale: wp.array[wp.float64]):
    i = wp.tid()
    scale[i] = wp.float64(1.0) / wp.sqrt(diagonal[i])


@wp.kernel
def congruence_values(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scale: wp.array[wp.float64],
):
    i = wp.tid()
    for entry in range(offsets[i], offsets[i + 1]):
        values[entry] *= scale[i] * scale[columns[entry]]


@wp.kernel
def recover_vectors(
    scaled: wp.array2d[wp.float64],
    scale: wp.array[wp.float64],
    physical: wp.array2d[wp.float64],
):
    j, i = wp.tid()
    physical[j, i] = scaled[j, i] * scale[i]


class ScaledRegular:
    """Benchmark standard D^-1/2 K D^-1/2; recover original-pencil vectors.

    The caller must verify positive diagonal mass. Scaling, sparse products,
    eigensolver work and output recovery all execute in Warp. Inputs are fixed
    for this comparison; changing their values requires rebuilding this setup.
    Counts/residuals are those of the scaled standard core. The comparison
    independently validates the original pencil and requested physical space.
    """

    def __init__(self, stiffness, mass, k, *, ncv, tol):
        self.n, self.device = stiffness.shape[0], stiffness.device
        self.scale = wp.empty(self.n, dtype=wp.float64, device=self.device)
        diagonal = ws.bsr_get_diag(mass)
        wp.launch(reciprocal_sqrt, self.n, [diagonal, self.scale], device=self.device)
        scaled = ws.bsr_copy(stiffness)
        wp.launch(
            congruence_values,
            self.n,
            [scaled.offsets, scaled.columns, scaled.values, self.scale],
            device=self.device,
        )
        self.core = KrylovSchur(SparseOperator(scaled), k, ncv=ncv, tol=tol, which="LM")
        self.eigenvectors = wp.empty_like(self.core.eigenvectors)

    def __getattr__(self, name):
        return getattr(self.core, name)

    @property
    def graph(self):
        return self.core.graph

    @graph.setter
    def graph(self, graph):
        self.core.graph = graph

    def recover(self):
        wp.launch(
            recover_vectors,
            self.eigenvectors.shape,
            [self.core.eigenvectors, self.scale, self.eigenvectors],
            device=self.device,
        )

    def capture(self, max_iterations):
        return self.core.capture(max_iterations, finalizer=self.recover)
