# Code for Theme 3 : River Basin Erosion Class
# Python port of singleBasin.m (original author: Ludovico Agostini, lagostin@ethz.ch, ETH-Zurich)
#
# Self-contained re-implementation of the TopoToolbox functions used in singleBasin.m
# (FLOWobj 'carve', flowacc, STREAMobj, klargestconncomps, removeshortstreams, trunk,
#  knickpointfinder, crs, mnoptimvar, chitransform, ksn, drainagebasins, gradient8, maplateral).
# The algorithms follow the same ideas but are NOT bit-identical to TopoToolbox -- see the
# notes in each function.
#
# Requirements: numpy, scipy, pandas, matplotlib, rasterio, numba, pyshp
#   python -m pip install numpy scipy pandas matplotlib rasterio numba pyshp

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, Normalize
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling, WktVersion
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_geom
from rasterio.windows import Window, from_bounds
from scipy import ndimage, optimize
from numba import njit
import shapefile  # pyshp

# %% 1. Set working directories / parameters -------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# "Project data-20261008" folder: set the environment variable RBE_DATA_DIR or edit the default
DATA_DIR = os.environ.get('RBE_DATA_DIR', os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')))

basin = 'Plessur'                                                  # <<< catchment name
year = 2023                                                        # <<< DTM year (2013, 2016, 2023)
pathIn = os.path.join(DATA_DIR, 'DEM', 'DEM', 'DEM', basin)        # <<< folder with the DTM
dem_file = f'{basin}_{year}.tif'                                   # <<< DTM file name
dem_crs_default = 'EPSG:21781'  # CH1903/LV03, used if the tif has no CRS (Plessur_2016)
basin_shp = os.path.join(DATA_DIR, 'Basin outline', 'Basin outline', basin,
                         f'{basin}_basin.shp')                     # outline to clip the DTM (None = no clip)
pathOut = os.path.join(HERE, 'output', f'{basin}_{year}')          # results (shp, csv) go here
pathFig = os.path.join(pathOut, 'figures')                         # graphs go here

resample_bool = True   # << Decide if you want to resample (2 m over a whole basin needs a lot of RAM)
res = 5.0              # New resolution to resample in [m]

Amin = 3e6             # <<< A min in surface values m2 (1e6 m2 == 1 km2) from Slope-Area analysis
dmin = 3000            # <<< threshold of trunks to be removed [m]
kp_tol = 10            # knickpoint tolerance (min dz of the knickpoint) [m]
Slope_min = 0.0001     # min slope to keep in the smoothed profile
zs_smooth = 100        # smoothing window of the corrected profile [m] (replaces crs 'K',6)
buffer = 100           # <<< lateral buffer for the confinement [m]
d_smooth = 3000        # slope smoothing for the chi plot [m]
mn_range = (0.1, 1.0)  # search range for the concavity m/n

SHOW_PLOTS = True      # False -> only save the figures as PNG

D8_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
D8_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
D8_DIST = np.sqrt(D8_DR ** 2 + D8_DC ** 2).astype(np.float64)


# =========================================================================================
# Grid functions
# =========================================================================================
def load_dem(path, resample, res, basin_shp=None, default_crs=None):
    """GRIDobj + clip to the basin outline + resample.
    Returns Z (float64, nan = nodata), transform, crs, cellsize."""
    with rasterio.open(path) as src:
        crs = src.crs if src.crs is not None else CRS.from_user_input(default_crs)
        window = None
        if basin_shp is not None:
            shp = shapefile.Reader(basin_shp)
            with open(os.path.splitext(basin_shp)[0] + '.prj') as f:
                shp_crs = CRS.from_wkt(f.read())
            geoms = [transform_geom(shp_crs, crs, s.__geo_interface__) for s in shp.shapes()]
            xs = np.concatenate([np.array(r)[:, 0] for g in geoms for p in
                                 (g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']])
                                 for r in p])
            ys = np.concatenate([np.array(r)[:, 1] for g in geoms for p in
                                 (g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']])
                                 for r in p])
            window = from_bounds(xs.min(), ys.min(), xs.max(), ys.max(), src.transform)
            window = window.round_offsets().round_lengths().intersection(
                Window(0, 0, src.width, src.height))
        Z = src.read(1, window=window).astype(np.float32)
        transform = src.window_transform(window) if window is not None else src.transform
        if src.nodata is not None:
            Z[Z == np.float32(src.nodata)] = np.nan
    Z[(Z <= 0) | (Z < -1e30) | (Z == -9999)] = np.nan  # set pixels to nan (0 = nodata in 2023)
    if basin_shp is not None:
        inside = rasterize([(g, 1) for g in geoms], out_shape=Z.shape, transform=transform,
                           fill=0, all_touched=True, dtype='uint8').astype(bool)
        Z[~inside] = np.nan
    if resample and abs(transform.a - res) > 1e-9:
        f = res / transform.a
        if abs(f - round(f)) < 1e-9:  # integer factor: nan-aware block mean
            f = int(round(f))
            nr, nc = Z.shape[0] // f, Z.shape[1] // f
            B = Z[:nr * f, :nc * f].reshape(nr, f, nc, f)
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                Z = np.nanmean(B, axis=(1, 3))
            transform = transform * transform.scale(f, f)
        else:
            nr, nc = int(round(Z.shape[0] / f)), int(round(Z.shape[1] / f))
            Zr = np.full((nr, nc), np.nan, np.float32)
            tr = transform * transform.scale(Z.shape[1] / nc, Z.shape[0] / nr)
            reproject(Z, Zr, src_transform=transform, src_crs=crs, dst_transform=tr, dst_crs=crs,
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
            Z, transform = Zr, tr
    return Z.astype(np.float64), transform, crs, float(transform.a)


def inpaintnans(Z):
    """Fill nan holes that are not connected to the DEM border (nearest valid value)."""
    nan = np.isnan(Z)
    lab, n = ndimage.label(nan)
    if n == 0:
        return Z
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = nan & ~np.isin(lab, border[border > 0])
    if holes.any():
        idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
        Z = Z.copy()
        Z[holes] = Z[idx[0][holes], idx[1][holes]]
    return Z


def gradient8(Z, cs):
    """Maximum downward gradient to the 8 neighbours [m/m] (TopoToolbox gradient8)."""
    P = np.pad(Z, 1, constant_values=np.nan)
    G = np.zeros(Z.shape, np.float32)
    nr, nc = Z.shape
    for dr, dc, dd in zip(D8_DR, D8_DC, D8_DIST):
        nb = P[1 + dr:1 + dr + nr, 1 + dc:1 + dc + nc]
        G = np.fmax(G, ((Z - nb) / (cs * dd)).astype(np.float32))
    G[np.isnan(Z)] = np.nan
    return G


# =========================================================================================
# Flow routing (FLOWobj with 'carve')
# =========================================================================================
@njit(cache=True)
def _heap_push(hk, hi, hs, key, idx):
    k = hs
    hk[k] = key
    hi[k] = idx
    while k > 0:
        p = (k - 1) >> 1
        if hk[p] <= hk[k]:
            break
        hk[p], hk[k] = hk[k], hk[p]
        hi[p], hi[k] = hi[k], hi[p]
        k = p
    return hs + 1


@njit(cache=True)
def _heap_pop(hk, hi, hs):
    idx = hi[0]
    hs -= 1
    hk[0] = hk[hs]
    hi[0] = hi[hs]
    k = 0
    while True:
        l = 2 * k + 1
        if l >= hs:
            break
        m = l
        if l + 1 < hs and hk[l + 1] < hk[l]:
            m = l + 1
        if hk[k] <= hk[m]:
            break
        hk[m], hk[k] = hk[k], hk[m]
        hi[m], hi[k] = hi[k], hi[m]
        k = m
    return idx, hs


@njit(cache=True)
def _flow_routing(Z, cs, dr, dc, dd):
    """Priority-flood from the DEM border on the *unfilled* DEM. Every cell gets a rank
    (processing order). A cell drains to its steepest-descent neighbour among the cells
    with lower rank; if none is lower (pit / flat) it drains to the cell it was reached
    from, i.e. along the least-cost path out of the depression (= carving).
    Receivers always have a lower rank -> the flow network is acyclic."""
    nr, nc = Z.shape
    n = nr * nc
    Zf = Z.ravel()
    rank = np.full(n, -1, np.int64)     # -1 unvisited, -2 queued, >=0 rank
    rec = np.full(n, -1, np.int64)      # first holds the flood parent
    order = np.empty(n, np.int64)
    hk = np.empty(n, np.float64)
    hi = np.empty(n, np.int64)
    hs = 0
    for r in range(nr):
        for c in range(nc):
            i = r * nc + c
            if np.isnan(Zf[i]):
                continue
            for k in range(8):
                rr = r + dr[k]
                cc = c + dc[k]
                if rr < 0 or rr >= nr or cc < 0 or cc >= nc or np.isnan(Zf[rr * nc + cc]):
                    hs = _heap_push(hk, hi, hs, Zf[i], i)
                    rank[i] = -2
                    break
    cnt = 0
    while hs > 0:
        i, hs = _heap_pop(hk, hi, hs)
        rank[i] = cnt
        order[cnt] = i
        cnt += 1
        r = i // nc
        c = i - r * nc
        for k in range(8):
            rr = r + dr[k]
            cc = c + dc[k]
            if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                continue
            j = rr * nc + cc
            if rank[j] != -1 or np.isnan(Zf[j]):
                continue
            rank[j] = -2
            rec[j] = i
            hs = _heap_push(hk, hi, hs, Zf[j], j)
    order = order[:cnt]
    for t in range(cnt):
        i = order[t]
        r = i // nc
        c = i - r * nc
        best = 0.0
        b = -1
        for k in range(8):
            rr = r + dr[k]
            cc = c + dc[k]
            if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                continue
            j = rr * nc + cc
            if rank[j] < 0 or rank[j] >= rank[i]:
                continue
            s = (Zf[i] - Zf[j]) / (cs * dd[k])
            if s > best:
                best = s
                b = j
        if b >= 0:
            rec[i] = b
    return rec, order


@njit(cache=True)
def _flowacc(rec, order, n):
    A = np.zeros(n, np.float64)
    for t in range(order.size):
        A[order[t]] = 1.0
    for t in range(order.size - 1, -1, -1):
        i = order[t]
        if rec[i] >= 0:
            A[rec[i]] += A[i]
    return A


@njit(cache=True)
def _label_to_nodes(rec, order, nodeof):
    """drainagebasins(FD, S.IXgrid): each cell gets the first stream node downstream."""
    lab = np.full(rec.size, -1, np.int64)
    for t in range(order.size):
        i = order[t]
        if nodeof[i] >= 0:
            lab[i] = nodeof[i]
        elif rec[i] >= 0:
            lab[i] = lab[rec[i]]
    return lab


# =========================================================================================
# Stream network (STREAMobj). Nodes are ordered topologically: receivers before donors.
# =========================================================================================
@njit(cache=True)
def _cum_down(rec, w):
    """out[i] = out[rec[i]] + w[i] (e.g. distance from outlet)."""
    out = np.zeros(rec.size)
    for i in range(rec.size):
        if rec[i] >= 0:
            out[i] = out[rec[i]] + w[i]
    return out


@njit(cache=True)
def _outlet_of(rec):
    out = np.empty(rec.size, np.int64)
    for i in range(rec.size):
        out[i] = i if rec[i] < 0 else out[rec[i]]
    return out


@njit(cache=True)
def _short_streams(rec, dx, ndon, dmin):
    keep = np.ones(rec.size, np.bool_)
    for h in range(rec.size):
        if ndon[h] != 0:
            continue
        L = 0.0
        i = h
        while rec[i] >= 0 and ndon[rec[i]] < 2:
            L += dx[i]
            i = rec[i]
        if rec[i] < 0:  # reached the outlet without confluence -> keep
            continue
        L += dx[i]
        if L < dmin:
            i = h
            while ndon[i] < 2:
                keep[i] = False
                i = rec[i]
    return keep


class Stream:
    def __init__(self, ix, rec, nr, nc, transform):
        self.ix, self.rec, self.nr, self.nc, self.transform = ix, rec, nr, nc, transform
        self.row, self.col = np.divmod(ix, nc)
        self.x = transform.c + (self.col + 0.5) * transform.a
        self.y = transform.f + (self.row + 0.5) * transform.e
        has = rec >= 0
        self.dx = np.zeros(ix.size)
        self.dx[has] = np.hypot(self.x[has] - self.x[rec[has]], self.y[has] - self.y[rec[has]])
        self.distance = _cum_down(rec, self.dx)
        self.ndon = np.bincount(rec[has], minlength=ix.size)

    @property
    def n(self):
        return self.ix.size

    def subset(self, keep):
        new = np.cumsum(keep) - 1
        rec = self.rec[keep]
        rec = np.where(rec >= 0, new[np.maximum(rec, 0)], -1)
        return Stream(self.ix[keep], rec, self.nr, self.nc, self.transform)

    def klargestconncomps(self, k=1):
        out = _outlet_of(self.rec)
        cnt = np.bincount(out, minlength=self.n)
        best = np.argsort(cnt)[::-1][:k]
        return self.subset(np.isin(out, best))

    def removeshortstreams(self, dmin):
        return self.subset(_short_streams(self.rec, self.dx, self.ndon, float(dmin)))

    def trunk(self):
        keep = np.zeros(self.n, bool)
        i = int(np.argmax(self.distance))
        while i >= 0:
            keep[i] = True
            i = self.rec[i]
        return self.subset(keep)

    def getnal(self, grid):
        return np.asarray(grid).ravel()[self.ix].astype(np.float64)

    def nodeof(self):
        m = np.full(self.nr * self.nc, -1, np.int64)
        m[self.ix] = np.arange(self.n)
        return m

    def paths(self):
        """Split the network in flow paths, longest first: [(nodes head->down, junction), ...].
        The first path is the trunk; junction is the node where the path joins (-1 = outlet)."""
        heads = np.flatnonzero(self.ndon == 0)
        heads = heads[np.argsort(self.distance[heads])[::-1]]
        owned = np.zeros(self.n, bool)
        out = []
        for h in heads:
            nodes = []
            i = h
            while i >= 0 and not owned[i]:
                owned[i] = True
                nodes.append(i)
                i = self.rec[i]
            out.append((np.array(nodes, np.int64), int(i)))
        return out

    def gradient(self, z):
        """Downstream gradient (z - z_receiver)/dx; nan at the outlet."""
        g = np.full(self.n, np.nan)
        has = self.rec >= 0
        g[has] = (z[has] - z[self.rec[has]]) / self.dx[has]
        return g

    def segments(self):
        """Segment id per node, split at confluences (as STREAMobj2mapstruct)."""
        seg = np.full(self.n, -1, np.int64)
        nid = 0
        for i in range(self.n - 1, -1, -1):  # upstream first
            if seg[i] < 0:
                seg[i] = nid
                nid += 1
            r = self.rec[i]
            if r >= 0 and self.ndon[r] == 1:
                seg[r] = seg[i]
        return seg

    def lines(self, X, Y):
        has = np.flatnonzero(self.rec >= 0)
        r = self.rec[has]
        return np.stack([np.c_[X[has], Y[has]], np.c_[X[r], Y[r]]], axis=1), has


def STREAMobj(rec_grid, A, Nmin, shape, transform, rank):
    nr, nc = shape
    ix = np.flatnonzero(A > Nmin)
    ix = ix[np.argsort(rank[ix], kind='stable')]
    nodeof = np.full(nr * nc, -1, np.int64)
    nodeof[ix] = np.arange(ix.size)
    r = rec_grid[ix]
    rec = np.where(r >= 0, nodeof[np.maximum(r, 0)], -1)
    return Stream(ix, rec, nr, nc, transform)


# =========================================================================================
# Profile tools
# =========================================================================================
@njit(cache=True)
def _lower_hull_profile(d, z):
    """Lower convex hull of z(d) (d increasing) evaluated at d -> monotone, upward concave
    profile lying below the data."""
    n = d.size
    h = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        while m >= 2:
            a = h[m - 2]
            b = h[m - 1]
            cr = (d[b] - d[a]) * (z[i] - z[a]) - (z[b] - z[a]) * (d[i] - d[a])
            if cr <= 0:
                m -= 1
            else:
                break
        h[m] = i
        m += 1
    out = np.empty(n)
    out[h[0]] = z[h[0]]
    for k in range(m - 1):
        a = h[k]
        b = h[k + 1]
        for i in range(a, b + 1):
            out[i] = z[a] + (z[b] - z[a]) * (d[i] - d[a]) / (d[b] - d[a])
    return out


def carve_profile(S, z):
    """Hydrologically correct a profile by carving: no node is higher than any upstream node."""
    zc = z.copy()
    for i in range(S.n - 1, -1, -1):
        r = S.rec[i]
        if r >= 0 and zc[i] < zc[r]:
            zc[r] = zc[i]
    return zc


def knickpointfinder(S, z, tol, paths):
    """Knickpoints after Schwanghart & Scherler (2017), heuristic version:
    each flow path is fitted with an upward-concave, monotone profile (lower convex hull of
    the carved profile). The node with the largest deviation becomes a knickpoint (the
    concavity constraint is relaxed there) and the fit is repeated until max deviation < tol.
    Returns the knickpoint-constrained profile zk and a knickpoint table."""
    zc = carve_profile(S, z)
    zk = zc.copy()
    kp, kdz = [], []
    for nodes, junc in paths:
        seq = nodes[::-1] if junc < 0 else np.r_[junc, nodes[::-1]]
        if seq.size < 3:
            continue
        d, zz = S.distance[seq], zc[seq]
        brk = [0, seq.size - 1]
        prof = _lower_hull_profile(d, zz)
        while True:
            dev = zz - prof
            k = int(np.argmax(dev))
            if dev[k] <= tol:
                break
            kp.append(seq[k])
            kdz.append(dev[k])
            j = np.searchsorted(brk, k)
            a, b = brk[j - 1], brk[j]
            brk.insert(j, k)
            prof[a:k + 1] = _lower_hull_profile(d[a:k + 1], zz[a:k + 1])
            prof[k:b + 1] = _lower_hull_profile(d[k:b + 1], zz[k:b + 1])
        own = slice(1, None) if junc >= 0 else slice(None)
        zk[seq[own]] = prof[own]
    kp = np.array(kp, np.int64)
    tab = pd.DataFrame({'x': S.x[kp], 'y': S.y[kp], 'IXgrid': S.ix[kp], 'row': S.row[kp],
                        'col': S.col[kp], 'distance': S.distance[kp], 'z': zc[kp],
                        'dz': np.array(kdz)})
    tab.insert(0, 'node', kp)
    return zk, tab


def smooth(S, v, L, paths, breaks=None):
    """Moving-average smoothing along the network with window length L [m] (nan-aware).
    Each flow path is smoothed together with its downstream continuation; the smoothing
    does not cross nodes in `breaks` (e.g. knickpoints)."""
    out = np.full(S.n, np.nan)
    isbrk = np.zeros(S.n, bool)
    if breaks is not None:
        isbrk[breaks] = True
    for nodes, junc in paths:
        ext = []
        i, Ld = junc, 0.0
        while i >= 0 and Ld < L:
            ext.append(i)
            Ld += S.dx[i]
            i = S.rec[i]
        seq = np.r_[nodes, np.array(ext, np.int64)]
        d = -S.distance[seq]                       # increasing downstream
        pieces = np.split(np.arange(seq.size), np.flatnonzero(isbrk[seq]) + 1)
        for p in pieces:
            if p.size == 0:
                continue
            vv = v[seq[p]]
            ok = np.isfinite(vv)
            cs = np.r_[0, np.cumsum(np.where(ok, vv, 0))]
            cn = np.r_[0, np.cumsum(ok)]
            lo = np.searchsorted(d[p], d[p] - L / 2, 'left')
            hi = np.searchsorted(d[p], d[p] + L / 2, 'right')
            with np.errstate(invalid='ignore', divide='ignore'):
                sm = (cs[hi] - cs[lo]) / (cn[hi] - cn[lo])
            mine = p[p < nodes.size]
            out[seq[mine]] = sm[mine - p[0]]
    return out


def crs(S, z, paths, L, mingradient, knickpoints=None):
    """Replacement for TopoToolbox crs (constrained regularized smoothing): profile raised
    over pits (upper-quantile like tau=0.95) with a minimum gradient, smoothed along the
    network with window L (not across knickpoints), then corrected again."""
    def impose(zz):
        zz = zz.copy()
        for i in range(S.n):  # downstream -> upstream
            r = S.rec[i]
            if r >= 0:
                zz[i] = max(zz[i], zz[r] + mingradient * S.dx[i])
        return zz
    zs = impose(z)
    zs = smooth(S, zs, L, paths, knickpoints)
    return impose(zs)


def chitransform(S, A_m2, a0, mn):
    return _cum_down(S.rec, (a0 / A_m2) ** mn * S.dx)


def mnoptimvar(S, z, A_m2, a0, paths, mn_range, plot=True, ngrid=200, return_curve=False):
    """Concavity m/n that minimises the (robust) variance of elevation between the flow
    paths at equal chi (= tributaries collapse onto the trunk, Perron & Royden 2013)."""
    seqs = [nodes[::-1] if j < 0 else np.r_[j, nodes[::-1]] for nodes, j in paths]
    seqs = [s for s in seqs if s.size >= 2]

    def cost(mn):
        chi = chitransform(S, A_m2, a0, mn)
        grid = np.linspace(0, chi.max(), ngrid)
        M = np.full((len(seqs), ngrid), np.nan)
        for k, s in enumerate(seqs):
            c = chi[s]
            m = (grid >= c[0]) & (grid <= c[-1])
            M[k, m] = np.interp(grid[m], c, z[s])
        ok = np.sum(np.isfinite(M), axis=0) >= 2
        M = M[:, ok]
        med = np.nanmedian(M, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(M - med), axis=0)
        return np.nanmean(mad ** 2)

    thetas = np.arange(mn_range[0], mn_range[1] + 1e-9, 0.025)
    costs = np.array([cost(t) for t in thetas])
    k = int(np.argmin(costs))
    lo, hi = thetas[max(k - 1, 0)], thetas[min(k + 1, thetas.size - 1)]
    mn = optimize.minimize_scalar(cost, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-3}).x
    if plot:
        fig, ax = plt.subplots(num='m/n optimisation')
        ax.plot(thetas, costs, 'k.-')
        ax.axvline(mn, color='r', label=f'm/n = {mn:.3f}')
        ax.set_xlabel('m/n')
        ax.set_ylabel('robust variance of z at equal $\\chi$ [m$^2$]')
        ax.legend()
        ax.set_title('Concavity calibration')
    if return_curve:
        return mn, thetas, costs
    return mn


def maplateral(S, G, buffer, cs, func=np.nanmedian):
    """Median of grid G within `buffer` [m] laterally of each stream node."""
    mask = np.ones((S.nr, S.nc), bool)
    mask.ravel()[S.ix] = False
    dist, idx = ndimage.distance_transform_edt(mask, sampling=cs, return_indices=True)
    inbuf = (dist <= buffer) & np.isfinite(G)
    near = (idx[0] * S.nc + idx[1])[inbuf]
    node = S.nodeof()[near]
    df = pd.DataFrame({'n': node, 'g': G[inbuf]})
    agg = df.groupby('n')['g'].median()
    out = np.full(S.n, np.nan)
    out[agg.index.values] = agg.values
    footprint = np.zeros((S.nr, S.nc), bool)
    footprint[inbuf] = True
    return out, footprint


# =========================================================================================
# Plot helpers
# =========================================================================================
def plot_network(ax, S, X, Y, c=None, cmap='viridis', norm=None, **kw):
    segs, has = S.lines(X, Y)
    lc = LineCollection(segs, **kw)
    if c is not None:
        lc.set_array(np.asarray(c)[has])
        lc.set_cmap(cmap)
        if norm is not None:
            lc.set_norm(norm)
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def hillshade(ax, Z, transform):
    extent = [transform.c, transform.c + Z.shape[1] * transform.a,
              transform.f + Z.shape[0] * transform.e, transform.f]
    hs = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(Z, nan=np.nanmin(Z)),
                                                     dx=transform.a, dy=-transform.e)
    hs = np.ma.masked_where(np.isnan(Z), hs)
    ax.imshow(hs, cmap='gray', extent=extent, interpolation='nearest')
    ax.set_aspect('equal')


def save(fig, name):
    fig.savefig(os.path.join(pathFig, f'{basin}_{year}_{name}.png'), dpi=200, bbox_inches='tight')


# =========================================================================================
# MAIN
# =========================================================================================
if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(pathFig, exist_ok=True)

    print('>>> load DTM <<<')
    DEM, transform, dem_crs, res = load_dem(os.path.join(pathIn, dem_file), resample_bool, res,
                                            basin_shp, dem_crs_default)
    DEM = inpaintnans(DEM)  # fill gaps, interpolate nan if close notNull cell
    nr, nc = DEM.shape
    print(f'  {nr} x {nc} cells, cellsize {res:g} m')

    # %% 2. ---------------- Extract river network and main trunk -----------------
    print('>>> Extract the river network <<<')
    FD_rec, FD_order = _flow_routing(DEM, res, D8_DR, D8_DC, D8_DIST)  # carve through pits
    rank = np.empty(nr * nc, np.int64)
    rank[FD_order] = np.arange(FD_order.size)
    FAcc = _flowacc(FD_rec, FD_order, nr * nc).reshape(nr, nc)  # [number of cells]
    Nmin = round(Amin / res ** 2)  # depends on the res
    S = STREAMobj(FD_rec, FAcc.ravel(), Nmin, (nr, nc), transform, rank)
    S2 = S.klargestconncomps(1)  # select main catchment (if DTM is made of more basins)
    S2 = S2.removeshortstreams(dmin)
    St = S2.trunk()  # select the main trunk
    paths = S2.paths()
    print(f'  {S2.n} stream nodes, {len(paths)} flow paths, trunk length {St.distance.max() / 1e3:.1f} km')

    z = S2.getnal(DEM)  # elevations at stream nodes
    A_m2 = S2.getnal(FAcc) * res ** 2

    # %% 2. ------------------- Knickpoint finder ---------------------------------
    print('>>> Knickpoint and concavity calculation <<<')
    zk, kp = knickpointfinder(S2, z, kp_tol, paths)
    print(f'  {len(kp)} knickpoints (tol = {kp_tol} m)')

    # ------------- smooth and correct interpolation errors ---------------------
    zs = crs(S2, z, paths, zs_smooth, Slope_min, knickpoints=kp['node'].values)

    # ----------------- calculate m/n ratio (concavity) ------------------------
    mn = mnoptimvar(S2, carve_profile(S2, z), A_m2, Amin, paths, mn_range, plot=True)
    print(f'concavity {mn:.4f}')

    # Calculate channel steepness
    k = S2.gradient(zs) * A_m2 ** mn

    # %% 3. Upstream basin average slope
    print('>>> Calculate the upstream catchment average slope <<<')
    G = gradient8(DEM, res)
    lab = _label_to_nodes(FD_rec, FD_order, S2.nodeof())
    ok = (lab >= 0) & np.isfinite(G.ravel())
    med = pd.Series(G.ravel()[ok]).groupby(lab[ok]).median()
    g = np.full(S2.n, np.nan)
    g[med.index.values] = med.values

    # %% 4. Confinement
    print('>>> Calculate the degree of confinement <<<')
    print(f'buffer: {buffer} m')
    s_mean, mask = maplateral(S2, G, buffer, res)
    grad = S2.gradient(z)  # gradient along stream (raw DTM)

    # %% SAVE FILES
    print('*** SAVE FILES ***')
    seg = S2.segments()
    attrs = {'ksn': k, 'FlowAcc_m2': A_m2, 'Elev_m': z, 'SlopeHill': s_mean, 'SlopeChan': grad}
    shp_path = os.path.join(pathOut, f'{basin}_{year}_ks')
    with shapefile.Writer(shp_path, shapeType=shapefile.POLYLINE) as w:
        for name in attrs:
            w.field(name, 'N', decimal=6)
        order = np.argsort(seg, kind='stable')
        bounds = np.flatnonzero(np.diff(seg[order])) + 1
        for nodes in np.split(order, bounds):
            nodes = nodes[::-1]  # upstream -> downstream
            r = S2.rec[nodes[-1]]
            pts = nodes if r < 0 else np.r_[nodes, r]
            if pts.size < 2:
                continue
            w.line([np.c_[S2.x[pts], S2.y[pts]].tolist()])
            w.record(*[float(np.nanmedian(v[nodes])) if np.isfinite(v[nodes]).any() else None
                       for v in attrs.values()])
    if dem_crs is not None:
        with open(shp_path + '.prj', 'w') as f:
            f.write(dem_crs.to_wkt(version=WktVersion.WKT1_ESRI))

    # %% Plot
    # Basin m/n concavity calibration
    c = chitransform(S2, A_m2, Amin, mn)
    kp['chi'] = c[kp['node'].values]
    kp['A_m2'] = A_m2[kp['node'].values]
    kp.drop(columns='node').to_csv(os.path.join(pathOut, f'{basin}_{year}_kp.csv'), sep=';', index=False)
    pd.DataFrame({'x': S2.x, 'y': S2.y, 'distance': S2.distance, 'chi': c, 'z': z, 'zs': zs,
                  'zk': zk, 'A_m2': A_m2, 'ksn': k, 'slope_channel': grad,
                  'slope_hill_buffer': s_mean, 'slope_basin': g}
                 ).to_csv(os.path.join(pathOut, f'{basin}_{year}_streamnodes.csv'), sep=';', index=False)
    save(plt.figure('m/n optimisation'), 'mn_optimisation')

    # Slope-Area plot (all cells of the DTM; objective 1: channel heads)
    fig, ax = plt.subplots(num='Slope-Area')
    Aall = FAcc.ravel() * res ** 2
    Gall = G.ravel()
    ok = (Aall > 0) & np.isfinite(Gall) & (Gall > 0)
    bins = np.logspace(np.log10(Aall[ok].min()), np.log10(Aall[ok].max()), 50)
    bi = np.digitize(Aall[ok], bins)
    sa = pd.DataFrame({'b': bi, 'A': Aall[ok], 'S': Gall[ok]}).groupby('b').median()
    sub = np.random.default_rng(0).choice(np.flatnonzero(ok), min(ok.sum(), 50000), replace=False)
    ax.loglog(Aall[sub], Gall[sub], '.', color='0.8', ms=1, label='cells (sample)')
    ax.loglog(sa['A'], sa['S'], 'ko-', ms=4, label='binned median')
    ax.axvline(Amin, color='r', ls='--', label='A$_{min}$')
    ax.set_xlabel('Drainage area [m$^2$]')
    ax.set_ylabel('Slope [m/m]')
    ax.legend()
    ax.set_title('Slope - Area')
    save(fig, 'slope_area')

    fig = plt.figure(3)
    ax = fig.gca()
    gs = smooth(S2, g, d_smooth, paths)  # smoothed slope
    lc = plot_network(ax, S2, c, z, c=gs, cmap='viridis', linewidths=2)
    ax.plot(kp['chi'], kp['z'], 'ko', mfc='w')
    ax.set_xlabel('$\\chi$ [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_title(f'$\\chi$ - z plot with knickpoints (m/n = {mn:.2g})')
    fig.colorbar(lc, ax=ax, label='Slope upstream catchment [m/m]')
    save(fig, 'chi_z')

    # Plot knickpoints location
    fig = plt.figure(4)
    ax = fig.gca()
    hillshade(ax, DEM, transform)
    lc = plot_network(ax, S2, S2.x, S2.y, c=k, cmap='jet', linewidths=1.5,
                      norm=Normalize(*np.nanpercentile(k, [2, 98])))
    ax.plot(kp['x'], kp['y'], 'ko', mfc='w')
    fig.colorbar(lc, ax=ax, label='k$_{sn}$')
    ax.set_title('Knickpoint distribution in the catchment')
    save(fig, 'kp_map')

    # Plot knickpoints longitudinal
    fig = plt.figure(5)
    ax = fig.gca()
    plot_network(ax, S2, S2.distance, zs, colors='b', linewidths=1)
    plot_network(ax, S2, S2.distance, zk, colors='r', linewidths=0.5)
    ax.plot(kp['distance'], kp['z'], 'ko', mfc='w')
    ax.set_xlabel('Distance upstream [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_title('Reconstructed river profile')
    save(fig, 'kp_profile')

    # Plot knickpoints height drop (planar)
    size = 20 + 200 * (kp['dz'] - kp['dz'].min()) / max(np.ptp(kp['dz']), 1e-9) if len(kp) else 20
    fig = plt.figure(6)
    ax = fig.gca()
    plot_network(ax, S2, S2.x, S2.y, colors='0.5', linewidths=0.5)
    sc = ax.scatter(kp['x'], kp['y'], s=size, c=kp['z'], cmap='viridis', edgecolors='k')
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Elevation [m]')
    if len(kp):
        h, l = sc.legend_elements('sizes', num=4, func=lambda s: (s - 20) / 200 * np.ptp(kp['dz']) + kp['dz'].min())
        ax.legend(h, l, title='Knickpoint height [m]', loc='upper left')
    ax.set_title('Knickpoint height and elevation')
    save(fig, 'kp_height_map')

    fig = plt.figure(7)
    ax = fig.gca()
    plot_network(ax, S2, S2.distance, z, colors='0.5', linewidths=0.5)
    sc = ax.scatter(kp['distance'], kp['z'], s=size, c=kp['z'], cmap='viridis', edgecolors='k')
    fig.colorbar(sc, ax=ax, label='Elevation [m]')
    if len(kp):
        ax.legend(h, l, title='Knickpoint height [m]', loc='upper left')
    ax.set_xlabel('Distance upstream [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_title('Knickpoint height and elevation - profile view')
    save(fig, 'kp_height_profile')

    print(f'done in {time.time() - t0:.0f} s, results in {pathOut}')
    if SHOW_PLOTS:
        plt.show()
