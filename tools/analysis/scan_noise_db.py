"""Scan the loss_log.db for noise-related metrics and report unusual values.

Usage:
  python tools/analysis/scan_noise_db.py --dbpath <path> --outdir <outdir>
"""
import argparse
import os
import sqlite3
import json
import math
from statistics import median

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def analyze(dbpath, outdir, sample_limit=2000):
    os.makedirs(outdir, exist_ok=True)
    con = sqlite3.connect(dbpath)
    cur = con.cursor()
    # load metric keys
    cur.execute("SELECT id,name FROM metric_keys")
    keys = cur.fetchall()

    results = []

    # prioritize obvious names
    priority_names = ['train/noise_sigma_mean', 'noise_sigma_mean', 'noise_sigma', 'noise_mean', 'train/noise_mean']

    candidates = []
    for kid, name in keys:
        lname = name.lower()
        if 'noise' in lname or 'sigma' in lname or 'snr' in lname or 'mu' in lname or 'shift' in lname:
            candidates.append((kid, name))
    # fallback: if nothing found, examine all keys
    if not candidates:
        candidates = keys

    for kid, name in candidates:
        cur.execute("SELECT value, rowid FROM metrics WHERE metric_key_id=? ORDER BY rowid DESC LIMIT ?", (kid, sample_limit))
        rows = cur.fetchall()
        vals = [safe_float(r[0]) for r in rows]
        vals = [v for v in vals if v is not None and (not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))]
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        entry = {
            'metric_id': int(kid),
            'name': name,
            'n': int(len(arr)),
            'min': float(arr.min()),
            'q10': float(np.quantile(arr, 0.1)),
            'median': float(np.median(arr)),
            'q90': float(np.quantile(arr, 0.9)),
            'max': float(arr.max()),
            'frac_negative': float((arr < 0.0).mean()),
            'frac_lt_0.01': float((arr < 0.01).mean()),
            'frac_gt_0.98': float((arr > 0.98).mean()),
        }
        # sample rows (recent)
        sample_rows = [{'rowid': int(r[1]), 'value': safe_float(r[0])} for r in rows[:20]]
        entry['recent'] = sample_rows
        results.append(entry)

    # save results
    with open(os.path.join(outdir, 'scan_noise_metrics.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    # build a csv summary
    df = pd.DataFrame([ {k:v for k,v in r.items() if k!='recent'} for r in results ])
    if not df.empty:
        df.to_csv(os.path.join(outdir, 'scan_noise_metrics_summary.csv'), index=False)

    con.close()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbpath', required=True)
    parser.add_argument('--outdir', default='output/analysis/db_noise_scan')
    args = parser.parse_args()
    res = analyze(args.dbpath, args.outdir)
    print('Done. Results saved to', os.path.abspath(args.outdir))
    # pretty print a short table
    for r in res:
        print(f"{r['name']}: n={r['n']}, min={r['min']:.6g}, median={r['median']:.6g}, frac_neg={r['frac_negative']:.2f}, frac_lt_0.01={r['frac_lt_0.01']:.2f}, frac_gt_0.98={r['frac_gt_0.98']:.2f}")
