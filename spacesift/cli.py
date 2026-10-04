"""Command line: spacesift run | replay | select-stars."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def cmd_run(args):
    from .runner import run_experiment

    run_experiment(Path(args.config), allow_dirty=args.dirty, max_stars=args.max_stars, n_jobs=args.jobs,
                   out_root=Path(args.out) if args.out else None)


def cmd_replay(args):
    """Re-run one stored trial from its recorded parameters and plot it."""
    from .config import load_config
    from .detrend import clean, flatten
    from .inject import Injection, transit_model
    from .plots import plot_trial
    from .runner import build_stars, load_star, run_trial

    exp = Path(args.experiment)
    cfg, _ = load_config(exp / "config.yaml")
    trials = pd.read_parquet(exp / "trials.parquet")
    row = trials[(trials.star_id == args.star) & (trials.trial == args.trial)]
    if args.detrend:
        row = row[row.detrend == args.detrend]
    if row.empty:
        raise SystemExit("no such trial")
    row = row.iloc[0]
    inj = Injection(**{k[4:]: row[k] for k in row.index if k.startswith("inj_")})
    star_index = int(row.star_index)
    star = build_stars(cfg)[star_index]
    lc = load_star(cfg, star, star_index)
    status, cand = run_trial(cfg, lc, star, inj, row.detrend)
    print(f"stored status: {row.status} | replayed status: {status}")
    model = transit_model(lc.time, inj, star.u1, star.u2, lc.exptime, cfg.injection.supersample)
    flat = flatten(lc.time, lc.flux * model, row.detrend)
    out = Path(args.out or exp / "plots" / f"trial_{row.star_id}_{row.trial}_{row.detrend}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_trial(lc.time, lc.flux * model, model, flat, inj, cand, status, out)
    print(f"wrote {out}")


def cmd_select(args):
    from .data import select_kepler_stars

    df = select_kepler_stars(args.n, args.seed, Path(args.out))
    print(f"wrote {len(df)} stars to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="spacesift")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("run", help="run an experiment from a config file")
    r.add_argument("config")
    r.add_argument("--dirty", action="store_true", help="allow uncommitted changes (recorded)")
    r.add_argument("--max-stars", type=int, help="limit stars, e.g. for a timing run")
    r.add_argument("--jobs", type=int, help="parallel workers (overrides config)")
    r.add_argument("--out", help="output root (default: experiments/)")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="re-run and plot one trial")
    rp.add_argument("experiment")
    rp.add_argument("--star", required=True)
    rp.add_argument("--trial", type=int, required=True)
    rp.add_argument("--detrend")
    rp.add_argument("--out")
    rp.set_defaults(func=cmd_replay)

    s = sub.add_parser("select-stars", help="freeze a Kepler star sample to CSV")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", default="configs/stars/SS-0001.csv")
    s.set_defaults(func=cmd_select)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
