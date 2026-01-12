"""Loss normalization analysis script

Usage:
  python tools/analysis/loss_normalization_analysis.py --dbpath <path> --outdir <outdir>

Produces CSV, PNGs, and a JSON summary with recommended normalization.
"""
import os
import argparse
import sqlite3
import json
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
except Exception as exc:
    raise RuntimeError("This script requires numpy, pandas, matplotlib and scipy. Install them in the environment.") from exc


def list_tables(conn):
    cur = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view')")
    return [r[0] for r in cur.fetchall()]


def get_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]


def try_find_columns(cols):
    cols_l = [c.lower() for c in cols]
    loss_cols = [c for c in cols if 'loss' in c.lower()]
    noise_cols = [c for c in cols if 'noise' in c.lower() or 'sigma' in c.lower()]
    return loss_cols, noise_cols


def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def run_analysis(dbpath, outdir):
    os.makedirs(outdir, exist_ok=True)
    conn = sqlite3.connect(dbpath)
    tables = list_tables(conn)
    print(f"Found tables: {tables}")

    # Handle common key-value metrics schema: 'metrics' + 'metric_keys'
    if 'metrics' in tables and 'metric_keys' in tables:
        print('Detected metrics/metric_keys schema; pivoting metrics into wide table')
        cur = conn.execute("SELECT m.step_id, k.name, m.value FROM metrics m JOIN metric_keys k ON k.id = m.metric_key_id")
        rows = cur.fetchall()
        dfm = pd.DataFrame(rows, columns=['step_id', 'key', 'value'])
        wide = dfm.pivot_table(index='step_id', columns='key', values='value', aggfunc='first').reset_index()
        cols = list(wide.columns)
        loss_cols, noise_cols = try_find_columns(cols)
        print('Wide columns:', cols)
        print(f"Loss columns candidates: {loss_cols}")
        print(f"Noise columns candidates: {noise_cols}")

        use_loss = loss_cols[0] if loss_cols else None
        candidate_noise = None
        for pref in ('noise_sigma_mean', 'noise_sigma', 'noise_mean', 'sigma'):
            if pref in [c.lower() for c in cols]:
                candidate_noise = [c for c in cols if c.lower() == pref][0]
                break
        if not candidate_noise and noise_cols:
            candidate_noise = noise_cols[0]

        if not use_loss:
            raise RuntimeError("No loss column found in pivoted metrics; please inspect the DB")
        if not candidate_noise:
            raise RuntimeError("No noise/sigma-like column found in pivoted metrics; please inspect the DB")

        df = wide[[use_loss, candidate_noise]].dropna()
        df.columns = ['loss', 'sigma']
    else:
        best_table = None
        for t in tables:
            cols = get_columns(conn, t)
            loss_cols, noise_cols = try_find_columns(cols)
            if loss_cols and noise_cols:
                best_table = t
                break
        if best_table is None:
            # fallback: choose first table
            best_table = tables[0]
            cols = get_columns(conn, best_table)
            loss_cols, noise_cols = try_find_columns(cols)

        print(f"Using table: {best_table}")
        print(f"Loss columns candidates: {loss_cols}")
        print(f"Noise columns candidates: {noise_cols}")

        use_loss = loss_cols[0] if loss_cols else None
        # prefer noise_sigma_mean or noise_sigma
        candidate_noise = None
        for pref in ('noise_sigma_mean', 'noise_sigma', 'noise_mean', 'sigma'):
            if pref in [c.lower() for c in cols]:
                candidate_noise = [c for c in cols if c.lower() == pref][0]
                break
        if not candidate_noise and noise_cols:
            candidate_noise = noise_cols[0]

        if not use_loss:
            raise RuntimeError("No loss column found in table; please inspect the DB")
        if not candidate_noise:
            raise RuntimeError("No noise/sigma-like column found in table; please inspect the DB")

        df = pd.read_sql_query(f"SELECT rowid, * FROM '{best_table}'", conn)
        df = df[[use_loss, candidate_noise]].dropna()
        df.columns = ['loss', 'sigma']
    # ensure numeric
    df = df[pd.to_numeric(df['loss'], errors='coerce').notnull()]
    df = df[pd.to_numeric(df['sigma'], errors='coerce').notnull()]
    df['loss'] = df['loss'].astype(float)
    df['sigma'] = df['sigma'].astype(float)

    # Basic stats
    stats_summary = {
        'n': int(len(df)),
        'loss_mean': float(df['loss'].mean()),
        'loss_std': float(df['loss'].std()),
        'loss_min': float(df['loss'].min()),
        'loss_max': float(df['loss'].max()),
        'sigma_mean': float(df['sigma'].mean()),
        'sigma_std': float(df['sigma'].std()),
        'sigma_min': float(df['sigma'].min()),
        'sigma_max': float(df['sigma'].max()),
    }
    print(json.dumps(stats_summary, indent=2))

    # Correlations
    pearson_orig = stats.pearsonr(df['loss'], df['sigma'])[0]
    spearman_orig = stats.spearmanr(df['loss'], df['sigma']).correlation

    candidates = {}
    eps = 1e-12
    candidates['loss_times_1_minus_sigma'] = df['loss'] * (1.0 - df['sigma'])
    candidates['loss_div_sigma'] = df['loss'] / (df['sigma'] + eps)
    candidates['loss_div_sigma_sq'] = df['loss'] / (np.square(df['sigma']) + eps)
    # fit log-log slope to estimate exponent
    mask = (df['loss'] > 0) & (df['sigma'] > 0)
    if mask.sum() >= 10:
        slope, intercept = np.polyfit(safe_log(df.loc[mask,'sigma']), safe_log(df.loc[mask,'loss']), 1)
        p_est = float(slope)
    else:
        p_est = 1.0
    candidates[f'loss_div_sigma_pow_{p_est:.3f}'] = df['loss'] / (np.power(df['sigma'], p_est) + eps)
    candidates['log_loss'] = np.log(df['loss'] + eps)

    # evaluate correlations
    corr_summary = {}
    for name, arr in candidates.items():
        arr = np.array(arr)
        # if constant or nan skip
        if np.all(np.isfinite(arr)) and arr.std() > 0:
            corr = stats.pearsonr(arr, df['sigma'])[0]
        else:
            corr = None
        corr_summary[name] = corr
    print(json.dumps({'pearson_orig': pearson_orig, 'spearman_orig': spearman_orig, 'candidate_correlations': corr_summary}, indent=2))

    # choose best (min abs correlation)
    best = None
    best_val = None
    for k,v in corr_summary.items():
        if v is None:
            continue
        a = abs(v)
        if best is None or a < best_val:
            best = k
            best_val = a

    # Save CSV of data and candidate columns
    out_csv = os.path.join(outdir, 'extracted_loss_sigma.csv')
    df.to_csv(out_csv, index=False)

    # add candidate columns to df and save
    for name, arr in candidates.items():
        df[name] = arr
    df.to_csv(os.path.join(outdir, 'loss_with_candidates.csv'), index=False)

    # Plots
    plt.figure(figsize=(6,4))
    plt.scatter(df['sigma'], df['loss'], s=2, alpha=0.6)
    plt.xlabel('sigma')
    plt.ylabel('loss')
    plt.title('Loss vs sigma')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'loss_vs_sigma.png'), dpi=150)
    plt.close()

    for name in candidates.keys():
        plt.figure(figsize=(6,4))
        plt.scatter(df['sigma'], df[name], s=2, alpha=0.6)
        plt.xlabel('sigma')
        plt.ylabel(name)
        plt.title(f'{name} vs sigma')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{name}_vs_sigma.png'), dpi=150)
        plt.close()

    summary = {
        'table_used': best_table,
        'loss_column': use_loss,
        'sigma_column': candidate_noise,
        'n_rows': int(len(df)),
        'stats': stats_summary,
        'pearson_orig': pearson_orig,
        'spearman_orig': spearman_orig,
        'candidate_correlations': corr_summary,
        'best_candidate': best,
        'best_candidate_abs_corr': best_val,
        'estimated_sigma_exponent': p_est,
    }
    with open(os.path.join(outdir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    # Save recommendation text
    rec = {
        'recommendation': None,
        'notes': []
    }
    # prefer a normalization that reduces correlation most
    if best is not None:
        rec['recommendation'] = f"Use '{best}' (estimated exponent p={p_est:.3f})."
        rec['notes'].append(f"This candidate has absolute Pearson correlation {best_val:.4f} to sigma, lower than original {abs(pearson_orig):.4f}.")
    else:
        rec['recommendation'] = "No clear improvement found; consider inverse-variance weighting Loss/(sigma^2) or log loss." 

    with open(os.path.join(outdir, 'recommendation.json'), 'w') as fh:
        json.dump(rec, fh, indent=2)

    print('\nAnalysis complete. Outputs saved to: ' + os.path.abspath(outdir))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbpath', required=True)
    parser.add_argument('--outdir', default='output/analysis/loss_norm_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    run_analysis(args.dbpath, args.outdir)
