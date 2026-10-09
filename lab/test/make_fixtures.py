"""Reference numbers from the Python pipeline for lab/test/engine.test.mjs.

Run from the repo root:  python lab/test/make_fixtures.py
"""

import json
from pathlib import Path

import numpy as np

from spacesift.detrend import flatten
from spacesift.inject import Injection, _blocked_fraction, a_over_rs, central_depth, mean_over_central, t14

rng = np.random.default_rng(3)
cases = []
for k, b in [(0.01, 0.0), (0.03, 0.5), (0.08, 0.85)]:
    z = np.linspace(0, 1 + k + 0.01, 25)
    cases.append({"k": k, "b": b, "z": z.tolist(), "blocked": _blocked_fraction(z, k, 0.4, 0.25, n_r=512).tolist(),
                  "mean_over_central": mean_over_central(k, b, 15.0, 5.0, 0.4, 0.25)})

star = {"radius": 1.1, "mass": 0.95}
geo = []
for P, k, b in [(3.0, 0.02, 0.2), (12.0, 0.03, 0.7), (29.0, 0.015, 0.0)]:
    from spacesift.inject import Star
    a = a_over_rs(P, Star("x", radius=star["radius"], mass=star["mass"]))
    geo.append({"P": P, "k": k, "b": b, "a_rs": a, "t14": t14(P, a, k, b), "central_depth": central_depth(k, b, 0.4, 0.25)})

# A fixed light curve with a slow trend, for the biweight comparison (wotan).
t = np.arange(0, 30, 29.4244 / 1440)
t = t[(t < 12) | (t > 13)]  # a gap, so segments are split
f = 1 + 300e-6 * np.sin(2 * np.pi * t / 5.0) + rng.normal(0, 150e-6, t.size)
biweight = {spec: flatten(t, f, spec).tolist() for spec in ("biweight-0.5", "biweight-1.0")}

out = {"blocked": cases, "geometry": geo, "star": star,
       "biweight": {"time": t.tolist(), "flux": f.tolist(), "flat": biweight}}
path = Path(__file__).with_name("fixtures.json")
path.write_text(json.dumps(out))
print(f"wrote {path}")
