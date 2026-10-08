# Theme 3 - objectives 1 & 2: channel heads and sensitivity of the river network to Amin
# Uses the functions and the DTM settings (basin, year, resolution, clip) of singleBasin.py
#
# 1. Slope-Area analysis: 2-segment fit in log-log space (hillslope-colluvial / fluvial);
#    the break is the drainage area of the channel heads.
# 2. chi analysis for a range of Amin: network maps, chi-z plots, number of channel heads,
#    drainage density and best-fit concavity m/n for each Amin.

import itertools
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LogNorm

import singleBasin as sb

Amins_km2 = [0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0]  # <<< Amin values to test [km2]
Amin_ref_km2 = 1.5     # highlighted value (used in the other course)
dmin = 0               # remove first-order streams shorter than dmin [m] (0 = keep all)
nbins = 60             # log bins for the Slope-Area plot
sa_fit_max_km2 = 50    # S-A fit only below this area (excludes the few bins of the outlet reach)
SHOW_PLOTS = True

pathA = os.path.join(sb.pathOut, 'amin_analysis')
figA = os.path.join(pathA, 'figures')


def fit_segments(x, y, nbreak=2, min_pts=3):
    """Continuous piecewise-linear least-squares fit y(x) with `nbreak` breaks (grid search
    over the bin centres). Returns breaks, slopes of each segment, fitted function."""
    def design(xx, b):
        return np.column_stack([np.ones_like(xx), xx] + [np.maximum(xx - bb, 0) for bb in b])

    best = (np.inf, None, None)
    for b in itertools.combinations(x[min_pts:-min_pts], nbreak):
        if nbreak > 1 and np.any(np.diff([np.sum(x < bb) for bb in b]) < min_pts):
            continue
        X = design(x, b)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = np.sum((X @ coef - y) ** 2)
        if sse < best[0]:
            best = (sse, np.array(b), coef)
    _, b, coef = best
    return b, np.cumsum(coef[1:]), lambda xx: design(xx, b) @ coef


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(figA, exist_ok=True)

    print('>>> load DTM <<<')
    DEM, transform, dem_crs, res = sb.load_dem(os.path.join(sb.pathIn, sb.dem_file), sb.resample_bool,
                                               sb.res, sb.basin_shp, sb.dem_crs_default)
    DEM = sb.inpaintnans(DEM)
    nr, nc = DEM.shape
    basin_km2 = np.isfinite(DEM).sum() * res ** 2 / 1e6
    print(f'  {nr} x {nc} cells, cellsize {res:g} m, basin {basin_km2:.1f} km2')

    print('>>> flow routing <<<')
    FD_rec, FD_order = sb._flow_routing(DEM, res, sb.D8_DR, sb.D8_DC, sb.D8_DIST)
    rank = np.empty(nr * nc, np.int64)
    rank[FD_order] = np.arange(FD_order.size)
    FAcc = sb._flowacc(FD_rec, FD_order, nr * nc)
    G = sb.gradient8(DEM, res).ravel()

    # %% 1. Slope-Area analysis ---------------------------------------------------------
    print('>>> Slope-Area analysis <<<')
    A = FAcc * res ** 2
    ok = (A > 0) & np.isfinite(G) & (G > 0)
    edges = np.logspace(np.log10(A[ok].min()), np.log10(A[ok].max()), nbins + 1)
    bi = np.digitize(A[ok], edges) - 1
    df = pd.DataFrame({'b': bi, 'A': A[ok], 'S': G[ok]})
    sa = df.groupby('b').agg(A=('A', 'median'), S=('S', 'median'), S25=('S', lambda s: s.quantile(.25)),
                             S75=('S', lambda s: s.quantile(.75)), n=('S', 'size'))
    sa = sa[sa['n'] >= 20]
    fit = sa['A'] <= sa_fit_max_km2 * 1e6
    x, y = np.log10(sa.loc[fit, 'A'].values), np.log10(sa.loc[fit, 'S'].values)
    brk, slopes, f = fit_segments(x, y, nbreak=1)
    A_brk = 10 ** brk
    print(f'  channel heads (break) at A = {A_brk[0] / 1e6:.3f} km2')
    print(f'  S-A exponents per segment: ' + ', '.join(f'{s:.3f}' for s in slopes))
    print(f'  fluvial concavity from S-A: theta = {-slopes[-1]:.3f}')
    sa.to_csv(os.path.join(pathA, f'{sb.basin}_{sb.year}_slope_area_bins.csv'), sep=';')

    fig, ax = plt.subplots(figsize=(7, 5), num='Slope-Area')
    sub = np.random.default_rng(0).choice(np.flatnonzero(ok), min(ok.sum(), 50000), replace=False)
    ax.loglog(A[sub], G[sub], '.', color='0.85', ms=1, zorder=0)
    ax.fill_between(sa['A'], sa['S25'], sa['S75'], color='0.6', alpha=0.4, lw=0, label='25-75 %')
    ax.loglog(sa['A'], sa['S'], 'ko', ms=4, label='binned median')
    xx = np.linspace(x.min(), x.max(), 200)
    ax.loglog(10 ** xx, 10 ** f(xx), 'r-', lw=2, label=f'2-segment fit (A < {sa_fit_max_km2} km$^2$)')
    ax.axvline(A_brk[0], color='r', ls='--', label=f'channel heads: {A_brk[0] / 1e6:.3g} km$^2$')
    ax.axvline(Amin_ref_km2 * 1e6, color='b', ls='-.', label=f'A$_{{min}}$ = {Amin_ref_km2} km$^2$')
    xs = np.r_[x.min(), brk, x.max()]
    for k, s in enumerate(slopes):
        xm = (xs[k] + xs[k + 1]) / 2
        ax.annotate(f'{s:.2f}', (10 ** xm, 10 ** f(np.array([xm]))[0]), textcoords='offset points',
                    xytext=(0, 8), color='r', ha='center')
    ax.set_xlabel('Drainage area [m$^2$]')
    ax.set_ylabel('Slope [m/m]')
    ax.set_title(f'Slope - Area, {sb.basin} {sb.year}')
    ax.legend(fontsize=8, loc='lower left')
    fig.savefig(os.path.join(figA, f'{sb.basin}_{sb.year}_slope_area_fit.png'), dpi=200, bbox_inches='tight')

    # %% 2. Network and chi analysis for different Amin ------------------------------------
    print('>>> chi analysis for different Amin <<<')
    rows, nets, curves = [], [], []
    for amin in Amins_km2:
        Amin = amin * 1e6
        S = sb.STREAMobj(FD_rec, FAcc, round(Amin / res ** 2), (nr, nc), transform, rank)
        S = S.klargestconncomps(1)
        if dmin > 0:
            S = S.removeshortstreams(dmin)
        paths = S.paths()
        z = sb.carve_profile(S, S.getnal(DEM))
        A_m2 = S.getnal(FAcc) * res ** 2
        mn, th, cost = sb.mnoptimvar(S, z, A_m2, Amin, paths, sb.mn_range, plot=False, return_curve=True)
        chi = sb.chitransform(S, A_m2, Amin, mn)
        # linearity of the chi-z relation (all nodes)
        p = np.polyfit(chi, z, 1)
        r2 = 1 - np.sum((z - np.polyval(p, chi)) ** 2) / np.sum((z - z.mean()) ** 2)
        L_km = S.dx.sum() / 1e3
        rows.append({'Amin_km2': amin, 'n_heads': int((S.ndon == 0).sum()), 'n_nodes': S.n,
                     'length_km': L_km, 'drainage_density_km-1': L_km / basin_km2,
                     'mn': mn, 'chi_z_R2': r2, 'ks_linear_fit': p[0]})
        nets.append((S, z, chi, A_m2, mn))
        curves.append((th, cost))
        print(f'  Amin {amin:5.2f} km2: {rows[-1]["n_heads"]:4d} heads, {L_km:6.1f} km, m/n = {mn:.3f}, '
              f'R2(chi-z) = {r2:.3f}')
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(pathA, f'{sb.basin}_{sb.year}_amin_summary.csv'), sep=';', index=False)

    # network maps
    n = len(Amins_km2)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    step = 4  # coarser hillshade for display
    Zd = DEM[::step, ::step]
    hsd = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(Zd, nan=np.nanmin(Zd)),
                                                      dx=res * step, dy=res * step)
    hsd = np.ma.masked_where(np.isnan(Zd), hsd)
    extent = [transform.c, transform.c + nc * transform.a, transform.f + nr * transform.e, transform.f]
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), num='Networks')
    for ax, amin, (S, z, chi, A_m2, mn) in zip(axs.flat, Amins_km2, nets):
        ax.imshow(hsd, cmap='gray', extent=extent)
        sb.plot_network(ax, S, S.x, S.y, colors='b', linewidths=0.6)
        heads = S.ndon == 0
        ax.plot(S.x[heads], S.y[heads], 'r.', ms=2)
        ax.set_title(f'A$_{{min}}$ = {amin} km$^2$  ({heads.sum()} heads)',
                     fontweight='bold' if amin == Amin_ref_km2 else None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(figA, f'{sb.basin}_{sb.year}_networks.png'), dpi=200, bbox_inches='tight')

    # chi-z plots coloured by drainage area
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), num='chi-z',
                            constrained_layout=True)
    norm = LogNorm(min(Amins_km2) * 1e6, FAcc.max() * res ** 2)
    for ax, amin, (S, z, chi, A_m2, mn) in zip(axs.flat, Amins_km2, nets):
        lc = sb.plot_network(ax, S, chi, z, c=A_m2, cmap='viridis', norm=norm, linewidths=0.8)
        ax.set_title(f'A$_{{min}}$ = {amin} km$^2$, m/n = {mn:.2f}',
                     fontweight='bold' if amin == Amin_ref_km2 else None)
        ax.set_xlabel('$\\chi$ [m]')
        ax.set_ylabel('Elevation [m]')
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    fig.colorbar(lc, ax=axs, label='Drainage area [m$^2$]', shrink=0.6)
    fig.savefig(os.path.join(figA, f'{sb.basin}_{sb.year}_chi_z_amin.png'), dpi=200, bbox_inches='tight')

    # m/n calibration curves and summary
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), num='Amin summary')
    cmap = plt.get_cmap('plasma')
    for k, (amin, (th, cost)) in enumerate(zip(Amins_km2, curves)):
        axs[0].plot(th, cost / cost.min(), color=cmap(k / (n - 1)), label=f'{amin} km$^2$',
                    lw=2.5 if amin == Amin_ref_km2 else 1)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('m/n')
    axs[0].set_ylabel('normalised misfit')
    axs[0].legend(fontsize=7, title='A$_{min}$')
    axs[0].set_title('Concavity calibration')
    axs[1].semilogx(summary['Amin_km2'], summary['mn'], 'ko-', label='$\\chi$ (mnoptimvar)')
    axs[1].axhline(-slopes[-1], color='r', ls='--', label='S-A fluvial segment')
    axs[1].set_xlabel('A$_{min}$ [km$^2$]')
    axs[1].set_ylabel('m/n')
    axs[1].legend()
    axs[1].set_title('Concavity vs A$_{min}$')
    axs[2].loglog(summary['Amin_km2'], summary['n_heads'], 'ko-', label='channel heads')
    ax2 = axs[2].twinx()
    ax2.semilogx(summary['Amin_km2'], summary['drainage_density_km-1'], 'bs--', label='drainage density')
    ax2.set_ylabel('Drainage density [km$^{-1}$]', color='b')
    axs[2].set_xlabel('A$_{min}$ [km$^2$]')
    axs[2].set_ylabel('Number of channel heads')
    axs[2].axvline(A_brk[0] / 1e6, color='r', ls='--')
    axs[2].set_title('Network vs A$_{min}$ (red: S-A channel heads)')
    for ax in (axs[1], axs[2]):
        ax.axvline(Amin_ref_km2, color='b', ls='-.', lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(figA, f'{sb.basin}_{sb.year}_amin_summary.png'), dpi=200, bbox_inches='tight')

    print(summary.to_string(index=False, float_format='%.3f'))
    print(f'done in {time.time() - t0:.0f} s, results in {pathA}')
    if SHOW_PLOTS:
        plt.show()
