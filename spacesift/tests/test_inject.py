import numpy as np
import pytest

from spacesift.inject import (Injection, Star, _blocked_fraction, a_over_rs, central_depth, count_transits,
                              sample_injection, t14, transit_model)


def uniform_overlap(z, k):
    """Analytic overlap area of two circles (star radius 1, planet radius k) / pi."""
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    inside = z <= 1 - k
    out[inside] = k**2
    part = (z > 1 - k) & (z < 1 + k)
    zp = z[part]
    k0 = np.arccos((k**2 + zp**2 - 1) / (2 * k * zp))
    k1 = np.arccos((1 - k**2 + zp**2) / (2 * zp))
    out[part] = (k**2 * k0 + k1 - 0.5 * np.sqrt(4 * zp**2 - (1 + zp**2 - k**2) ** 2)) / np.pi
    return out


@pytest.mark.parametrize("k", [0.01, 0.05, 0.1])
def test_uniform_disk_matches_analytic(k):
    z = np.linspace(0, 1 + k + 0.01, 400)
    got = _blocked_fraction(z, k, 0.0, 0.0, n_r=512)
    assert np.allclose(got, uniform_overlap(z, k), atol=2e-3 * k**2 + 1e-9)


def test_limb_darkened_central_depth():
    k, b, u1, u2 = 0.02, 0.3, 0.4, 0.25
    z = np.array([b])
    assert _blocked_fraction(z, k, u1, u2)[0] == pytest.approx(central_depth(k, b, u1, u2), rel=0.01)


def test_model_shape_and_duration():
    star = Star("x")
    P, k, b = 10.0, 0.02, 0.2
    a = a_over_rs(P, star)
    dur = t14(P, a, k, b)
    assert 0.1 < dur < 0.25  # an Earth-ish 10-day transit of the Sun lasts ~4-5 hours
    inj = Injection(P, 5.0, k, b, a, dur, central_depth(k, b, .4, .25), 2.2, 10, 10, 3)
    t = np.linspace(0, 30, 300_000)
    f = transit_model(t, inj, 0.4, 0.25)
    assert f.max() == 1.0
    in_tr = f < 1
    # Three transits; total time in transit equals 3 * T14 to within a cadence.
    assert in_tr.sum() * (t[1] - t[0]) == pytest.approx(3 * dur, rel=0.01)


def test_count_transits_respects_gaps():
    t = np.arange(0, 40, 0.02)
    t = t[(t < 14) | (t > 26)]  # a gap that swallows the transits at t=15 and t=25
    assert count_transits(t, 10.0, 5.0, 0.2) == 2  # only t=5 and t=35 remain


def test_sample_injection_hits_target_snr():
    rng = np.random.default_rng(0)
    t = np.arange(0, 360, 29.4 / 1440)
    f = 1 + rng.normal(0, 200e-6, t.size)
    for _ in range(20):
        inj = sample_injection(rng, t, f, Star("x"), (5, 20), (1, 30), (0, 0.9))
        assert inj.expected_snr == pytest.approx(inj.target_snr, rel=0.15)
