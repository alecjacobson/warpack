"""Teaser styling adapted from gptoolbox (MIT; see THIRD_PARTY_NOTICES).

Source revision dd3554053237cd02d305b78db298d913a6196508:
imageprocessing/okloop.m, imageprocessing/oklab2lin.m, matrix/squease.m.
These host-side functions only prepare visualization data, not eigenproblems.
"""

import numpy as np

GPT_REVISION = "dd3554053237cd02d305b78db298d913a6196508"


def squease(t):
    t = np.clip(np.asarray(t), 0, 1)
    a = t / 4 - 5 / 108
    b = np.cbrt(a + np.sqrt(a * a + 1 / 5832))
    c = b - 1 / (18 * b) + 1 / 3
    return np.clip(c * c * (-3 * b + 1 / (6 * b) + 2) + c**3, 0, 1)


def pulse(t):
    """gptoolbox's there-and-back example, parameterized on [0, 1]."""
    t = np.asarray(t) * 2
    return np.where(t < 1, squease(t), 1 - squease(t - 1))


def okloop_jet_linear(count=256):
    theta = np.arange(count) / count * (-4 / 3 * np.pi) - np.pi / 2
    lightness = 0.75010101010101016
    a, b = 0.12755316371916220 * np.array([np.cos(theta), np.sin(theta)])
    lms = (
        np.column_stack(
            [
                lightness + 0.3963377774 * a + 0.2158037573 * b,
                lightness - 0.1055613458 * a - 0.0638541728 * b,
                lightness - 0.0894841775 * a - 1.2914855480 * b,
            ]
        )
        ** 3
    )
    return (
        lms
        @ np.array(
            [
                [4.0767245293, -3.3072168827, 0.2307590544],
                [-1.2681437731, 2.6093323231, -0.3411344290],
                [-0.0041119885, -0.7034763098, 1.7068625689],
            ]
        ).T
    )


def bounded_scales(vertices, modes, margin=0.95):
    """Largest positive modal amplitudes inside a centered 2x rest bounding box."""
    extent = np.ptp(vertices, axis=0)
    center = (vertices.min(0) + vertices.max(0)) / 2
    low, high = center - extent, center + extent
    room = np.where(modes > 0, high - vertices, vertices - low)
    limits = np.full_like(modes, np.inf)
    np.divide(room, np.abs(modes), out=limits, where=np.abs(modes) > 1e-30)
    return margin * limits.min(axis=(1, 2)), low, high
