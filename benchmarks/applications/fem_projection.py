"""Device analytic rigid spaces and host-only physical validation contracts."""

from dataclasses import replace

import numpy as np
import warp as wp

from examples.problems.elasticity import rigid_body_basis
from warpack import MassOrthogonalComplementInverse


def device_rigid_rows(case, device):
    """Known component geometry only; no reference eigenvectors or numeric readback."""
    positions = wp.array(case.vertices, dtype=wp.vec3d, device=device)
    rows = rigid_body_basis(positions, wp.ones(len(case.vertices), dtype=wp.float64, device=device))
    if case.problem.nullity == 6:
        return rows
    if case.problem.nullity != 12:
        raise ValueError("Fixture must have one or two known free components")
    # This fixture is two disjoint copies of the same beam.
    result = wp.zeros((12, case.problem.n), dtype=wp.float64, device=device)
    for component in range(2):
        for mode in range(6):
            wp.copy(result[6 * component + mode], rows[mode], dest_offset=component * rows.shape[1])
    return result


def physical_problem(case):
    p = case.problem
    first, last = p.nullity, p.k
    if first not in (6, 12):
        raise ValueError("Fixture must have known free-body rigid modes")
    return replace(
        p,
        k=last - first,
        expected=case.spectrum[first:last],
        nullity=0,
        null_basis=np.zeros((p.n, 0)),
        requested_basis=case.vectors[:, first:last],
        notes=p.notes + "; projected solve returns only the complete positive elastic block",
    )


def rigid_inverse_transform(case):
    def transform(inverse, mass):
        return MassOrthogonalComplementInverse(
            inverse, mass, device_rigid_rows(case, inverse.device), normalize=True
        )

    return transform
