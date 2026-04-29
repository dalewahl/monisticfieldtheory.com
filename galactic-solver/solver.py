"""
MFT Galactic Rotation Curves — browser-adapted

Solves the nonlinear contraction-field BVP on a galactic scale [0.1, 80] kpc
for one of six published spiral galaxies, then fits the stellar mass-to-light
ratio Υ* and halo density scale ρ_scale jointly to the observed rotation curve.

THE PHYSICS
-----------
The MFT contraction field δ(r) satisfies the nonlinear Poisson-like equation:

    ∇²δ + (2/r)dδ/dr = (1/κ)[m²_g δ + λ_4_g δ³ + λ_6_g δ⁵ + β ρ_baryon(r)]

with boundary conditions dδ/dr(R_min) = 0 and δ(R_max) = 0.

The contraction-field energy density becomes:

    ρ_MFT(r) = ½κ(dδ/dr)² + ½m²_g δ² + ¼λ_4_g δ⁴ + (1/6)λ_6_g δ⁶

This effective halo, integrated as M_MFT(r), adds to the baryonic mass to give
the total enclosed mass and hence the rotation curve:

    v(r) = √(G·M_total(r)/r)

The silver-ratio condition λ_4_g² = 8·m²_g·λ_6_g is enforced exactly. The
gravitational coupling β ≈ 1 × 10⁻⁴ is the same value that produces the
neutrino mass hierarchy and passes the Cassini Solar-System bound.

Source: mft_galactic_nonlinear.py (P5 of the corpus).
"""

import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d


# ── Physical constants ────────────────────────────────────────────
G = 4.30091e-6       # gravitational constant in (kpc/M_sun)·(km/s)²
KAPPA = 1.0
DELTA_SR = 1.0 + np.sqrt(2.0)   # silver ratio

# Gravitational sextic — silver-ratio condition exactly enforced
M2G = 1e-6
LAM4G = -2 * M2G**2          # NOTE: NEGATIVE coefficient in the gravitational sector
LAM6G = 0.5 * M2G**3
# Verify silver-ratio: λ_4² = 8 m² λ_6
assert abs(LAM4G**2 - 8 * M2G * LAM6G) / max(abs(LAM4G**2), 1e-50) < 1e-6

# Default β (can be varied by the user)
BETA_DEFAULT = 1.016e-4

R_MIN = 0.1     # kpc
R_MAX = 80.0    # kpc
N_BVP = 300
N_FINE = 2000


# ── Galaxy database (from P5) ─────────────────────────────────────

GALAXIES = {
    'MW': {
        'full': 'Milky Way',
        'M_b': 8e9, 'a_b': 0.5,           # bulge
        'M_d': 6e10, 'R_d': 3.0,          # disk
        'M_g': 1e10, 'R_g': 7.0,          # gas
        'BH': 4e6,
        'r_obs': [5., 8, 10, 15, 20, 25, 30, 50],
        'v_obs': [234.3, 229.2, 226, 217.3, 208.8, 200.3, 185, 157.8],
        'sv': 10.0,
    },
    'M31': {
        'full': 'Andromeda (M31)',
        'M_b': 3e10, 'a_b': 1.0,
        'M_d': 8e10, 'R_d': 5.5,
        'M_g': 2e10, 'R_g': 9.0,
        'BH': 1.4e8,
        'r_obs': [5., 8, 12, 16, 20, 25, 35],
        'v_obs': [250., 255, 255, 250, 245, 235, 210],
        'sv': 10.0,
    },
    'NGC3198': {
        'full': 'NGC 3198',
        'M_b': 1e9, 'a_b': 0.5,
        'M_d': 4e10, 'R_d': 3.5,
        'M_g': 1.5e10, 'R_g': 7.0,
        'BH': 1e7,
        'r_obs': [2., 6, 10, 14, 18, 22, 26, 30],
        'v_obs': [130., 155, 160, 159, 156.5, 153, 150, 150],
        'sv': 10.0,
    },
    'NGC2403': {
        'full': 'NGC 2403',
        'M_b': 5e8, 'a_b': 0.4,
        'M_d': 2e10, 'R_d': 2.0,
        'M_g': 8e9, 'R_g': 5.0,
        'BH': 5e6,
        'r_obs': [2., 6, 10, 14, 18, 22, 26, 30],
        'v_obs': [90., 125, 130, 127.5, 122.5, 120, 120, 120],
        'sv': 10.0,
    },
    'NGC7793': {
        'full': 'NGC 7793',
        'M_b': 3e8, 'a_b': 0.4,
        'M_d': 1.5e10, 'R_d': 1.8,
        'M_g': 7e9, 'R_g': 4.0,
        'BH': 3e6,
        'r_obs': [2., 6, 10, 14, 18, 22, 26, 30],
        'v_obs': [80., 106.7, 107.5, 100, 95, 95, 95, 95],
        'sv': 10.0,
    },
    'UGC2259': {
        'full': 'UGC 2259',
        'M_b': 1e8, 'a_b': 0.3,
        'M_d': 6e9, 'R_d': 1.5,
        'M_g': 5e9, 'R_g': 4.0,
        'BH': 1e6,
        'r_obs': [2., 6, 10, 14, 18, 22, 26, 30],
        'v_obs': [60., 77, 72.5, 70, 70, 70, 70, 70],
        'sv': 10.0,
    },
}


# ── Baryonic density profiles ─────────────────────────────────────

def rho_stellar(r, g):
    """Hernquist bulge + exponential disk."""
    r = np.maximum(np.atleast_1d(np.float64(r)), 1e-6)
    rho = (g['M_b'] / (2 * np.pi)) * g['a_b'] / (r * (r + g['a_b'])**3)
    if g['M_d'] > 0:
        rho += (g['M_d'] / (8 * np.pi * g['R_d']**3)) * np.exp(-r / g['R_d'])
    return rho


def rho_gas(r, g):
    """Exponential gas disk."""
    if g['M_g'] == 0:
        return np.zeros_like(np.atleast_1d(np.float64(r)))
    r = np.maximum(np.atleast_1d(np.float64(r)), 1e-6)
    return (g['M_g'] / (8 * np.pi * g['R_g']**3)) * np.exp(-r / g['R_g'])


# ── BVP solver ────────────────────────────────────────────────────

def solve_galaxy_bvp(g, beta, r_fine):
    """
    Solve the contraction-field BVP for one galaxy.
    Returns (delta, delta', success).
    """
    rho_arr = rho_stellar(r_fine, g) + rho_gas(r_fine, g)
    rbi = interp1d(r_fine, rho_arr, fill_value='extrapolate')
    
    def ode(r, y):
        d, dp = y
        rs = np.maximum(r, 1e-3)
        return np.vstack([
            dp,
            (M2G * d + LAM4G * d**3 + LAM6G * d**5 + beta * rbi(rs)) / KAPPA - 2 / rs * dp
        ])
    
    def bc(ya, yb):
        return np.array([ya[1], yb[0]])
    
    rn = np.linspace(R_MIN, R_MAX, N_BVP)
    dg = 60000.0 * (1 - rn / R_MAX)**2
    ddg = np.gradient(dg, rn)
    
    for tol, mn in [(5e-2, 8000), (1e-1, 10000)]:
        try:
            sol = solve_bvp(ode, bc, rn, np.vstack([dg, ddg]),
                            tol=tol, max_nodes=mn, verbose=0)
            if sol.success:
                du = np.interp(r_fine, sol.x, sol.y[0])
                dpu = np.interp(r_fine, sol.x, sol.y[1])
                return du, dpu, True
        except Exception:
            pass
    
    return np.zeros(len(r_fine)), np.zeros(len(r_fine)), False


def compute_halo(delta, dphi, r_fine):
    """Compute MFT halo density and enclosed mass."""
    rho = np.maximum(
        0.5 * KAPPA * dphi**2
        + 0.5 * M2G * delta**2
        + 0.25 * LAM4G * delta**4
        + (1.0 / 6.0) * LAM6G * delta**6,
        0.0
    )
    M = 4 * np.pi * cumulative_trapezoid(rho * r_fine**2, r_fine, initial=0.0)
    return rho, M


# ── Fit Υ* and ρ_scale to rotation curve ─────────────────────────

def fit_galaxy(g, M_mft_raw, r_fine, sMenc, gMenc):
    """Find best-fit (Υ*, ρ_scale) by 2-D parameter scan."""
    ro = np.array(g['r_obs'])
    vo = np.array(g['v_obs'])
    sv = g['sv']
    
    Ms_o = np.interp(ro, r_fine, sMenc)
    Mg_o = np.interp(ro, r_fine, gMenc)
    MU_o = np.interp(ro, r_fine, M_mft_raw)
    
    # Coarse scan
    best_chi2 = 1e30
    best_ups = 1.0
    best_rs = 1.0
    for ups in np.arange(0.3, 3.01, 0.05):
        Mb = ups * Ms_o + Mg_o
        for lg in np.linspace(-10, 4, 57):
            rs = 10.0**lg
            v = np.sqrt(G * (Mb + g['BH'] + rs * MU_o) / np.maximum(ro, 1e-3))
            chi2 = float(np.sum(((v - vo) / sv)**2))
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_ups = ups
                best_rs = rs
    
    # Refined scan
    for ups in np.linspace(max(0.2, best_ups - 0.04), min(4, best_ups + 0.04), 17):
        Mb = ups * Ms_o + Mg_o
        for lg in np.linspace(np.log10(max(best_rs * 0.03, 1e-10)), np.log10(best_rs * 30), 61):
            rs = 10.0**lg
            v = np.sqrt(G * (Mb + g['BH'] + rs * MU_o) / np.maximum(ro, 1e-3))
            chi2 = float(np.sum(((v - vo) / sv)**2))
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_ups = ups
                best_rs = rs
    
    return best_chi2, float(best_ups), float(best_rs)


# ── Compute predicted rotation curve at given (Υ*, ρ_scale) ──────

def rotation_curve(r, g, ups, rho_scale, M_mft_raw, r_fine, sMenc, gMenc):
    """Compute v(r) at fitted parameters."""
    Ms = np.interp(r, r_fine, sMenc)
    Mg = np.interp(r, r_fine, gMenc)
    MU = rho_scale * np.interp(r, r_fine, M_mft_raw)
    
    Mb = ups * Ms + Mg
    M_total = Mb + g['BH'] + MU
    
    v_baryon = np.sqrt(G * (Mb + g['BH']) / np.maximum(r, 1e-3))
    v_halo = np.sqrt(G * np.maximum(MU, 0) / np.maximum(r, 1e-3))
    v_total = np.sqrt(G * M_total / np.maximum(r, 1e-3))
    
    return v_baryon, v_halo, v_total


# ── Downsample arrays for plotting ────────────────────────────────

def downsample(arr, n_target=200):
    """Reduce a 1D array to ~n_target points by uniform indexing."""
    a = np.asarray(arr)
    if len(a) <= n_target:
        return a.tolist()
    idx = np.linspace(0, len(a) - 1, n_target).astype(int)
    return a[idx].tolist()


# ── Main solve() entry point ──────────────────────────────────────

def solve(params):
    """
    Top-level solver entry point.
    
    params: {
        'galaxy': str (key from GALAXIES),
        'beta': float,
        'fit_mode': 'auto' | 'manual',
        'ups_manual': float,         # used if fit_mode=='manual'
        'rho_scale_manual': float,   # used if fit_mode=='manual'
    }
    """
    try:
        galaxy_key = params.get('galaxy', 'MW')
        if galaxy_key not in GALAXIES:
            return {'success': False,
                    'message': f'Unknown galaxy {galaxy_key}. Available: {list(GALAXIES.keys())}'}
        
        g = GALAXIES[galaxy_key]
        beta = float(params.get('beta', BETA_DEFAULT))
        fit_mode = params.get('fit_mode', 'auto')
        
        if beta <= 0 or beta > 1e-2:
            return {'success': False,
                    'message': f'β = {beta} out of physical range (0, 1e-2].'}
        
        # Set up fine grid
        r_fine = np.linspace(R_MIN, R_MAX, N_FINE)
        
        # Solve BVP
        delta, dpdelta, ok = solve_galaxy_bvp(g, beta, r_fine)
        if not ok:
            return {'success': False,
                    'message': 'BVP did not converge for this galaxy at this β.',
                    'galaxy': galaxy_key, 'beta': beta}
        
        # Compute halo
        rho_mft, M_mft = compute_halo(delta, dpdelta, r_fine)
        
        # Pre-compute baryonic enclosed masses
        sMenc = 4 * np.pi * cumulative_trapezoid(rho_stellar(r_fine, g) * r_fine**2,
                                                 r_fine, initial=0.0)
        gMenc = 4 * np.pi * cumulative_trapezoid(rho_gas(r_fine, g) * r_fine**2,
                                                 r_fine, initial=0.0)
        
        # Fit or use manual values
        if fit_mode == 'manual':
            ups = float(params.get('ups_manual', 1.0))
            rho_scale = float(params.get('rho_scale_manual', 1.0))
            ro = np.array(g['r_obs'])
            vo = np.array(g['v_obs'])
            sv = g['sv']
            Ms_o = np.interp(ro, r_fine, sMenc)
            Mg_o = np.interp(ro, r_fine, gMenc)
            MU_o = np.interp(ro, r_fine, M_mft)
            v_pred = np.sqrt(G * (ups * Ms_o + Mg_o + g['BH'] + rho_scale * MU_o)
                             / np.maximum(ro, 1e-3))
            chi2 = float(np.sum(((v_pred - vo) / sv)**2))
        else:
            chi2, ups, rho_scale = fit_galaxy(g, M_mft, r_fine, sMenc, gMenc)
        
        dof = max(len(g['r_obs']) - 2, 1)
        
        # Compute rotation curve on a fine grid for plotting
        r_plot = np.linspace(R_MIN, min(R_MAX, max(g['r_obs']) * 1.3), 250)
        v_baryon, v_halo, v_total = rotation_curve(
            r_plot, g, ups, rho_scale, M_mft, r_fine, sMenc, gMenc
        )
        
        # Residuals at observation points
        ro = np.array(g['r_obs'])
        vo = np.array(g['v_obs'])
        v_baryon_obs, v_halo_obs, v_total_obs = rotation_curve(
            ro, g, ups, rho_scale, M_mft, r_fine, sMenc, gMenc
        )
        residuals = (v_total_obs - vo).tolist()
        residuals_sigma = ((v_total_obs - vo) / g['sv']).tolist()
        
        # Pre-compute info about silver-ratio scales
        alpha = 1.0 / np.sqrt(M2G)
        delta_b_silver = float(alpha * np.sqrt(2 - np.sqrt(2)))
        delta_v_silver = float(alpha * np.sqrt(2 + np.sqrt(2)))
        
        return {
            'success': True,
            'message': 'Galaxy fit complete.',
            'galaxy_key': galaxy_key,
            'galaxy_full_name': g['full'],
            'galaxy_params': {
                'M_b': g['M_b'], 'a_b': g['a_b'],
                'M_d': g['M_d'], 'R_d': g['R_d'],
                'M_g': g['M_g'], 'R_g': g['R_g'],
                'BH': g['BH'],
            },
            'beta': beta,
            'fit_mode': fit_mode,
            'fit_results': {
                'chi2': chi2,
                'dof': dof,
                'chi2_per_dof': chi2 / dof,
                'ups': ups,
                'rho_scale': rho_scale,
                'log10_rho_scale': float(np.log10(max(rho_scale, 1e-30))),
            },
            'observed': {
                'r': g['r_obs'],
                'v': g['v_obs'],
                'sigma_v': g['sv'],
            },
            'rotation_curve': {
                'r': downsample(r_plot, 200),
                'v_baryon': downsample(v_baryon, 200),
                'v_halo': downsample(v_halo, 200),
                'v_total': downsample(v_total, 200),
            },
            'contraction_field': {
                'r': downsample(r_fine, 200),
                'delta': downsample(delta, 200),
                'delta_b_silver': delta_b_silver,
                'delta_v_silver': delta_v_silver,
                'max_delta': float(np.max(np.abs(delta))),
            },
            'residuals': {
                'r': g['r_obs'],
                'delta_v': residuals,
                'delta_v_over_sigma': residuals_sigma,
            },
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'message': f'Solver error: {type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }


# ── Galaxy info helper for the JS layer ──────────────────────────

def get_galaxy_list():
    """Return list of (key, full_name) for the dropdown."""
    return [(k, GALAXIES[k]['full']) for k in
            ['MW', 'M31', 'NGC3198', 'NGC2403', 'NGC7793', 'UGC2259']]
