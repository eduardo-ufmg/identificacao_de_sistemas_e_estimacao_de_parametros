#!/usr/bin/env python3
"""
Usage examples:
    python chua.py --files pcchua_dados.dat pcchua_pert.dat --summary --plot all --fft x y --embed x 3 20 --lyapunov x 5 100

Features:
 - robust loader for the PCChua .dat format (skips comments, reads numeric columns)
 - prints summary stats (mean, std, min, max) per column
 - time-series plots (save PNGs)
 - FFT / PSD plots
 - spectrograms
 - 2D phase plots (x vs y, x vs z, y vs z)
 - time-delay embedding (Takens) and plot
 - simple Rosenstein largest Lyapunov exponent estimator (rough, for diagnostics)
 - CSV export option for cleaned numeric streams
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.spatial import KDTree
from math import log
import os
import sys

# -------------------------
# IO and parsing
# -------------------------
def load_pcchua_dat(path):
    """
    Load a PCChua-style .dat file.
    Returns (df, meta) where df is a pandas.DataFrame with columns:
      ['time','x','y','z','xref','yref','zref','ux','uy','uz'] (if present)
    and meta is a dict of parsed header lines (Date, Experiment, Ts if present).
    """
    path = Path(path)
    meta = {}
    data_lines = []
    with path.open('r') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if s.startswith('#'):
                # attempt to capture simple metadata
                content = s.lstrip('#').strip()
                if ':' in content:
                    k, v = content.split(':',1)
                    meta[k.strip()] = v.strip()
                else:
                    # other header lines like "Ts = 10.00 ms"
                    if 'Ts' in content:
                        meta['Ts'] = content
                    else:
                        # keep general header text
                        meta.setdefault('header', []).append(content)
                continue
            # numeric line
            parts = s.split()
            # drop lines that obviously are not data
            if len(parts) < 2:
                continue
            data_lines.append(parts)
    if len(data_lines) == 0:
        raise ValueError(f"No data found in {path}")
    # convert to float matrix
    mat = np.array(data_lines, dtype=float)
    # guess columns:
    # first is time, next 3 are x y z, next 3 xref yref zref, next 3 ux uy uz  => total 10 columns
    ncols = mat.shape[1]
    if ncols == 10:
        cols = ['time','x','y','z','xref','yref','zref','ux','uy','uz']
    elif ncols == 4:
        cols = ['time','x','y','z']
    elif ncols == 7:
        cols = ['time','x','y','z','xref','yref','zref']
    else:
        # fallback: generic names
        cols = ['c{}'.format(i) for i in range(ncols)]
        cols[0] = 'time'
    df = pd.DataFrame(mat, columns=cols)
    return df, meta

# -------------------------
# Basic summaries and export
# -------------------------
def summary_stats(df):
    s = df.describe().transpose()
    return s[['count','mean','std','min','25%','50%','75%','max']]

def save_csv(df, outpath):
    df.to_csv(outpath, index=False)
    return outpath

# -------------------------
# Plots
# -------------------------
def plot_time_series(df, outdir, which=None):
    """
    which: list of column names to plot. If None, plot all except 'time'
    Saves PNG files in outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    t = df['time'].values
    if which is None:
        which = [c for c in df.columns if c != 'time']
    for col in which:
        plt.figure(figsize=(8,3.5))
        plt.plot(t, df[col].values)
        plt.xlabel('time (s)')
        plt.title(col)
        plt.tight_layout()
        p = Path(outdir) / f"time_{col}.png"
        plt.savefig(p, dpi=200)
        plt.close()

def plot_phase_pairs(df, outdir, pairs=None):
    os.makedirs(outdir, exist_ok=True)
    if pairs is None:
        candidates = [c for c in df.columns if c != 'time']
        pairs = []
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                pairs.append((candidates[i], candidates[j]))
    for a,b in pairs:
        plt.figure(figsize=(5,5))
        plt.plot(df[a].values, df[b].values, linewidth=0.5)
        plt.xlabel(a); plt.ylabel(b)
        plt.title(f"{a} vs {b}")
        plt.tight_layout()
        p = Path(outdir) / f"phase_{a}_vs_{b}.png"
        plt.savefig(p, dpi=200)
        plt.close()

# -------------------------
# FFT and spectrogram
# -------------------------
def compute_psd(x, fs):
    """
    Return frequencies and one-sided PSD via welch.
    """
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    return f, Pxx

def plot_fft(x, fs, outpath, title='FFT'):
    n = len(x)
    freqs = rfftfreq(n, d=1.0/fs)
    X = rfft(x * np.hanning(n))
    mag = np.abs(np.array(X)) / n
    plt.figure(figsize=(6,3.5))
    plt.semilogy(freqs, mag)
    plt.xlabel('Hz')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_spectrogram(x, fs, outpath, nperseg=256):
    f, t_spec, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg)
    plt.figure(figsize=(7,3.5))
    plt.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-20), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -------------------------
# Embedding (Takens) & plotting
# -------------------------
def takens_embedding(x, dim, delay):
    """
    Return embedded matrix of shape (N - (dim-1)*delay, dim)
    """
    N = len(x)
    M = N - (dim - 1) * delay
    if M <= 0:
        raise ValueError("Time series too short for requested embedding")
    emb = np.zeros((M, dim))
    for i in range(dim):
        emb[:, i] = x[i*delay : i*delay + M]
    return emb

def plot_embedding(emb, outpath, comps=(0,1,2)):
    plt.figure(figsize=(6,6))
    if emb.shape[1] >= 3 and len(comps) >= 3:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.plot(emb[:, comps[0]], emb[:, comps[1]], emb[:, comps[2]], lw=0.3)
        ax.set_xlabel(f"dim{comps[0]}")
        ax.set_ylabel(f"dim{comps[1]}")
        ax.set_zlabel(f"dim{comps[2]}")
    else:
        plt.plot(emb[:,0], emb[:,1], lw=0.3)
        plt.xlabel('dim0'); plt.ylabel('dim1')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -------------------------
# Rosenstein LLE estimator (simple)
# -------------------------
def estimate_lyapunov_rosenstein(x, emb_dim=6, delay=10, fs=100.0, max_t=50):
    """
    Rough largest Lyapunov exponent estimator following Rosenstein et al. 1993.
    Returns (times, avg_divergence, slope_est)
    Notes:
      - x : 1D array
      - emb_dim, delay: embedding parameters
      - fs: sampling frequency in Hz
      - max_t: maximum number of time steps to follow a neighbor (in samples)
    """
    emb = takens_embedding(x, emb_dim, delay)
    N = len(emb)
    tree = KDTree(emb)
    # nearest neighbor excluding temporally close points (Theiler window)
    theiler = int(0.5 * fs)  # half-second default exclusion
    neigh_idx = np.zeros(N, dtype=int)
    for i in range(N):
        d, idx = tree.query(emb[i], k=10)
        # find first valid neighbor with temporal separation > theiler
        found = False
        for candidate in np.atleast_1d(idx):
            if abs(candidate - i) > theiler:
                neigh_idx[i] = candidate
                found = True
                break
        if not found:
            neigh_idx[i] = -1
    # prepare divergence curves
    max_t = int(max_t)
    L = np.full((N, max_t), np.nan)
    for i in range(N):
        j = neigh_idx[i]
        if j < 0:
            continue
        # how many steps we can follow
        max_k = min(max_t, N - max(i, j))
        for k in range(max_k):
            L[i, k] = np.linalg.norm(emb[i+k] - emb[j+k])
    # average log divergence
    with np.errstate(invalid='ignore', divide='ignore'):
        logL = np.log(L)
    avg = np.nanmean(logL, axis=0)
    times = np.arange(len(avg)) / fs
    # linear fit to initial linear region to get slope (LLE)
    # pick region avoiding k=0 and NaNs
    valid = ~np.isnan(avg)
    if np.sum(valid) < 5:
        slope = np.nan
    else:
        # use first 10-30% of the series or up to 0.5*max_t
        end = max(2, int(0.2 * len(avg)))
        x_fit = times[1:end]
        y_fit = avg[1:end]
        A = np.vstack([x_fit, np.ones_like(x_fit)]).T
        slope, intercept = np.linalg.lstsq(A, y_fit, rcond=None)[0]
    return times, avg, slope

# -------------------------
# Utilities
# -------------------------
def infer_fs_from_time(t):
    # compute median dt and return sampling frequency
    dt = np.diff(t)
    med = np.median(dt)
    if med <= 0:
        raise ValueError("Non-positive time steps encountered")
    return 1.0 / med

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Process PCChua .dat files")
    parser.add_argument('--files', nargs='+', required=True, help='Input .dat files')
    parser.add_argument('--summary', action='store_true', help='Print summary stats')
    parser.add_argument('--plot', nargs='*', default=[], help='Plot tasks: all, timeseries, phase, spectrogram')
    parser.add_argument('--fft', nargs='*', help='Columns to FFT (e.g. x y). Use "all" for all numeric except time.')
    parser.add_argument('--embed', nargs=3, metavar=('COL','DIM','DELAY'), help='Do time-delay embedding for column')
    parser.add_argument('--lyapunov', nargs=3, metavar=('COL','DIM','DELAY'), help='Estimate largest Lyapunov exponent for a column')
    parser.add_argument('--outdir', default='out_chua', help='Directory to write outputs')
    parser.add_argument('--export_csv', action='store_true', help='Export cleaned numeric CSVs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for filepath in args.files:
        print(f"Loading {filepath} ...")
        df, meta = load_pcchua_dat(filepath)
        basename = Path(filepath).stem
        subdir = outdir / basename
        subdir.mkdir(parents=True, exist_ok=True)

        # export CSV if requested
        if args.export_csv:
            csvp = subdir / f"{basename}.csv"
            save_csv(df, csvp)
            print(f"Saved CSV -> {csvp}")

        # summary
        if args.summary:
            s = summary_stats(df)
            txt = subdir / f"{basename}_summary.txt"
            with txt.open('w') as f:
                f.write("Metadata:\n")
                for k,v in meta.items():
                    f.write(f"{k}: {v}\n")
                f.write("\nSummary stats:\n")
                f.write(s.to_string())
            print(f"Wrote summary to {txt}")

        # infer fs
        try:
            fs = infer_fs_from_time(df['time'].values)
        except Exception:
            fs = None

        # plotting modes
        plot_modes = set(args.plot or [])
        if 'all' in plot_modes or 'timeseries' in plot_modes:
            cols = [c for c in df.columns if c != 'time']
            plot_time_series(df, subdir, which=cols)
            print(f"Saved time-series plots for {basename}")

        if 'all' in plot_modes or 'phase' in plot_modes:
            pairs = [('x','y'), ('x','z'), ('y','z')]
            # fallback: only existing columns
            existing_pairs = [(a,b) for (a,b) in pairs if a in df.columns and b in df.columns]
            plot_phase_pairs(df, subdir, pairs=existing_pairs)
            print(f"Saved phase plots for {basename}")

        if 'all' in plot_modes or 'spectrogram' in plot_modes:
            for col in [c for c in df.columns if c != 'time']:
                if fs is None:
                    continue
                try:
                    plot_spectrogram(df[col].values, fs, subdir / f"spect_{col}.png")
                except Exception as e:
                    print("Spectrogram failed for", col, e)
            print(f"Saved spectrograms for {basename}")

        # FFT
        if args.fft:
            fft_cols = args.fft
            if 'all' in fft_cols:
                fft_cols = [c for c in df.columns if c != 'time']
            if fs is None:
                print("Warning: sampling frequency could not be inferred. FFT may be incorrect.")
            for col in fft_cols:
                if col not in df.columns:
                    print(f"Column {col} not in {basename}; skipping FFT")
                    continue
                try:
                    plot_fft(df[col].values, fs or 1.0, subdir / f"fft_{col}.png", title=f"{basename} {col} FFT")
                except Exception as e:
                    print("FFT failed for", col, e)
            print(f"Saved FFTs for {basename}")

        # embedding
        if args.embed:
            col, dim_s, delay_s = args.embed
            dim = int(dim_s); delay = int(delay_s)
            if col not in df.columns:
                print(f"Embed column {col} not found in {basename}")
            else:
                emb = takens_embedding(df[col].values, dim, delay)
                plot_path = subdir / f"embed_{col}_d{dim}_tau{delay}.png"
                plot_embedding(emb, plot_path)
                np.save(subdir / f"embed_{col}_d{dim}_tau{delay}.npy", emb)
                print(f"Wrote embedding and plot for {col}")

        # lyapunov
        if args.lyapunov:
            col, dim_s, delay_s = args.lyapunov
            dim = int(dim_s); delay = int(delay_s)
            if col not in df.columns:
                print(f"Lyapunov column {col} not found in {basename}")
            else:
                times, avg, slope = estimate_lyapunov_rosenstein(df[col].values, emb_dim=dim, delay=delay, fs=float(fs or 1.0), max_t=200)
                # save divergence curve
                np.savetxt(subdir / f"lyap_{col}_curve.txt", np.vstack([times, avg]).T)
                with (subdir / f"lyap_{col}_report.txt").open('w') as f:
                    f.write(f"Estimated LLE (slope) = {slope}\n")
                    f.write(f"Embedding dim={dim}, delay={delay}, fs={fs}\n")
                print(f"Lyapunov estimation done for {col}. slope={slope}")

    print("Done.")

if __name__ == '__main__':
    main()
