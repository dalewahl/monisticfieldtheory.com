"""
MFT Hedgehog / Skyrmion BVP — browser-adapted

Solves the radial profile equation for the B=1 hedgehog Skyrmion:

    f''(r) = [sin(2f)(1 + sin²f / r²) - 2r f'(r) - sin(2f) f'(r)²] / (r² + 2 sin²f)

with boundary conditions f(0) = π, f(∞) = 0.

After convergence, computes:
  - E2 (quadratic energy)
  - E4 (quartic energy)
  - E2/E4 (virial balance — should be 1.0 for true soliton)
  - ε₀ = E2 + E4  (total energy in soliton units)
  - λ₀  (moment of inertia integral)
  - B (topological charge — should be 1)

From these, extracts the Skyrme parameters f_π and e by simultaneously
matching the nucleon mass M_N = 938.3 MeV and Δ-N splitting
M_Δ - M_N = 293.7 MeV.

The MFT prediction f_π² = δ × (MFT_TO_MEV)² ≈ (185.9 MeV)² is then
compared to the BVP-extracted f_π and to the observed value 186 MeV.

Source: mft_hedgehog_bvp_v2.py from the MFT corpus (P8).
"""

import numpy as np
from scipy.integrate import solve_bvp

try:
    _trap = np.trapezoid
except AttributeError:
    _trap = np.trapz


# ── Physical constants ────────────────────────────────────────────
M_ELECTRON = 0.511      # MeV (calibration)
E_ELECTRON_MFT = 0.00427
MFT_TO_MEV = M_ELECTRON / E_ELECTRON_MFT     # ≈ 119.67 MeV per MFT unit
M_N = 938.3             # MeV (nucleon)
M_DELTA = 1232.0        # MeV (delta)
F_PI_OBS = 186.0        # MeV (observed pion decay constant)
DELTA = 1.0 + np.sqrt(2.0)                 # silver ratio


# ── ODE and boundary conditions ───────────────────────────────────

def hedgehog_ode(r, y):
    """
    f'' = [sin(2f)(1 + sin²f/r²) - 2 r f' - sin(2f) f'²] / (r² + 2 sin²f)
    
    The full ODE includes the nonlinear term -sin(2f)·f'² that was
    missing from earlier versions.
    """
    f = y[0]
    fp = y[1]
    s = np.sin(f)
    c = np.cos(f)
    s2 = s * s
    s2f = 2.0 * s * c
    r_safe = np.maximum(r, 1e-10)
    denom = r_safe * r_safe + 2.0 * s2
    numer = s2f * (1.0 + s2 / (r_safe * r_safe)) - 2.0 * r_safe * fp - s2f * fp * fp
    return np.vstack([fp, numer / np.maximum(denom, 1e-15)])


def bc(ya, yb):
    """f(0) = π, f(rmax) = 0."""
    return np.array([ya[0] - np.pi, yb[0]])


# ── Single BVP solve at chosen parameters ─────────────────────────

def solve_bvp_at(rmin, rmax, R_h, n_mesh=300, tol=1e-9):
    """
    Solve the hedgehog BVP at one set of parameters.
    
    Initial guess uses the f(r) = 2 arctan(R_h / r) Skyrmion profile —
    the analytic answer for the pure-Skyrme limit (no MFT corrections).
    """
    # Mesh with clustering near origin
    n_inner = max(20, n_mesh // 2)
    n_outer = max(20, n_mesh // 2)
    r_inner = np.linspace(rmin, 3.0, n_inner)
    r_outer = np.linspace(3.0, rmax, n_outer)
    r_mesh = np.unique(np.concatenate([r_inner, r_outer]))
    
    f_guess = 2.0 * np.arctan(R_h / np.maximum(r_mesh, 1e-12))
    fp_guess = -2.0 * R_h / (r_mesh**2 + R_h**2)
    
    sol = solve_bvp(
        hedgehog_ode, bc, r_mesh,
        np.vstack([f_guess, fp_guess]),
        tol=tol, max_nodes=30000, verbose=0
    )
    
    return sol


# ── Compute integrals from a converged solution ───────────────────

def compute_integrals(r, f, fp):
    """
    From the radial profile (r, f(r), f'(r)), compute:
    - E2 = ∫ d³x [ ½ Tr(R_i R_i) ]    quadratic energy
    - E4 = ∫ d³x [ -¼ Tr [R_i, R_j]² ] quartic (Skyrme) energy
    - B  = baryon number (should be 1)
    - λ  = moment of inertia integral
    """
    s = np.sin(f)
    c = np.cos(f)
    s2 = s * s
    s4 = s2 * s2
    r_safe = np.maximum(r, 1e-10)
    
    # Energy integrals (in MFT units; factor of 4π already absorbed)
    E2 = 4.0 * np.pi * float(_trap(fp**2 * r**2 + 2.0 * s2, r))
    E4 = 4.0 * np.pi * float(_trap(2.0 * s2 * fp**2 + s4 / r_safe**2, r))
    
    # Topological baryon number
    # B = -(2/π) ∫ sin²f · f' dr  (analytic)
    B_int = -(2.0 / np.pi) * float(_trap(s2 * fp, r))
    B_topo = float((f[0] - f[-1]) / np.pi)  # degree of map from boundary values
    
    # Moment of inertia integral
    integrand = s2 * (r**2 + 2.0 * s2 * (fp**2 + s2 / r_safe**2))
    lam_0 = (8.0 * np.pi / 3.0) * float(_trap(integrand, r))
    
    eps_0 = E2 + E4
    virial = abs(E2 - E4) / max(E2, 1e-10)
    e2_over_e4 = E2 / max(E4, 1e-15)
    
    return {
        'E2': float(E2),
        'E4': float(E4),
        'eps_0': float(eps_0),
        'lambda_0': float(lam_0),
        'B': float(B_int),
        'B_topo': float(B_topo),
        'virial_imbalance': float(virial),
        'e2_over_e4': float(e2_over_e4),
    }


# ── Extract f_π and e from converged solution ─────────────────────

def extract_skyrme_params(integrals):
    """
    Given the converged integrals (ε₀, λ₀), solve simultaneously for
    f_π and e by matching M_N = 938.3 and ΔM = M_Δ - M_N = 293.7 MeV.
    
    The Skyrmion mass formula:
      M_N = (f_π / 4e) · ε₀ + 3/(8 Λ)
    
    where Λ is the moment of inertia:
      Λ = (1 / 6 e³ f_π) · λ₀
    
    From the rotational quantisation:
      M_N + M_Δ - 2 M_N = 3/(2 Λ)  (gives Λ)
    
    Then: M_core = M_N - 3/(8 Λ)
    M_core = (f_π / 4e) · ε₀
    
    Combined with Λ formula → 2 equations, 2 unknowns.
    """
    eps_0 = integrals['eps_0']
    lam_0 = integrals['lambda_0']
    
    if eps_0 < 1e-6 or lam_0 < 1e-6:
        return None
    
    deltaM = M_DELTA - M_N        # 293.7 MeV
    
    # Λ from Δ-N splitting: M_Δ - M_N = 3/(2 Λ)
    Lambda_tgt = 3.0 / (2.0 * deltaM)
    
    # Core mass: M_core = M_N - 3/(8 Λ)
    M_core_tgt = M_N - 3.0 / (8.0 * Lambda_tgt)
    
    if M_core_tgt <= 0:
        return None
    
    # From Λ = λ₀ / (6 e³ f_π) and M_core = (f_π / 4e) · ε₀
    # → f_π = 4 e M_core / ε₀ ... (1)
    # → Λ = λ₀ / (6 e³ × 4 e M_core / ε₀) = λ₀ ε₀ / (24 e⁴ M_core)
    # → 24 e⁴ M_core Λ = λ₀ ε₀
    # → e⁴ = λ₀ ε₀ / (24 M_core Λ)
    
    e4_val = (eps_0 * 2.0 * lam_0) / (4.0 * M_core_tgt * 3.0 * Lambda_tgt)
    if e4_val <= 0:
        return None
    
    e = e4_val ** 0.25
    f_pi = 4.0 * e * M_core_tgt / eps_0   # MeV
    
    # MFT prediction: f_π = sqrt(δ) × MFT_TO_MEV
    f_pi_mft_pred = float(np.sqrt(DELTA) * MFT_TO_MEV)
    
    # Compare e to 2δ (a candidate value from MFT)
    e_candidate = 2.0 * DELTA
    
    return {
        'f_pi_extracted_MeV': float(f_pi),
        'f_pi_observed_MeV': F_PI_OBS,
        'f_pi_mft_predicted_MeV': f_pi_mft_pred,
        'f_pi_error_vs_obs_pct': abs(f_pi - F_PI_OBS) / F_PI_OBS * 100,
        'f_pi_error_vs_mft_pct': abs(f_pi - f_pi_mft_pred) / f_pi_mft_pred * 100,
        'e_extracted': float(e),
        'e_candidate_2delta': float(e_candidate),
        'e_diff_vs_2delta_pct': abs(e - e_candidate) / e_candidate * 100,
        'Lambda_MeV_inv': float(Lambda_tgt),
        'M_core_MeV': float(M_core_tgt),
    }


# ── Plotting helpers ──────────────────────────────────────────────

def downsample_profile(r, f, fp, n_target=200, r_max_plot=15.0):
    """Reduce profile to ~n_target points clipped to plotting range."""
    mask = r <= r_max_plot
    r_p = r[mask]
    f_p = f[mask]
    fp_p = fp[mask]
    
    if len(r_p) > n_target:
        idx = np.linspace(0, len(r_p) - 1, n_target).astype(int)
        r_p = r_p[idx]
        f_p = f_p[idx]
        fp_p = fp_p[idx]
    
    return r_p.tolist(), f_p.tolist(), fp_p.tolist()


def energy_density_profile(r, f, fp, r_max_plot=8.0, n_pts=180):
    """Return (r, e2_density, e4_density) for the energy density plot."""
    s = np.sin(f)
    s2 = s * s
    s4 = s2 * s2
    r_safe = np.maximum(r, 1e-10)
    
    e2_density = 4.0 * np.pi * (fp**2 * r**2 + 2.0 * s2)
    e4_density = 4.0 * np.pi * (2.0 * s2 * fp**2 + s4 / r_safe**2)
    
    mask = r <= r_max_plot
    r_p = r[mask]
    e2 = e2_density[mask]
    e4 = e4_density[mask]
    
    if len(r_p) > n_pts:
        idx = np.linspace(0, len(r_p) - 1, n_pts).astype(int)
        r_p = r_p[idx]
        e2 = e2[idx]
        e4 = e4[idx]
    
    return r_p.tolist(), e2.tolist(), e4.tolist()


# ── Main solve() entry point ──────────────────────────────────────

def solve(params):
    """
    Top-level solver entry point.
    
    params: dict containing {
        'rmax': float (e.g., 30.0),
        'R_h': float (initial guess scale, e.g., 1.0),
        'n_mesh': int (e.g., 300),
        'rmin': float (default 1e-3),
        'tol': float (default 1e-9),
    }
    """
    try:
        rmax = float(params.get('rmax', 30.0))
        R_h = float(params.get('R_h', 1.0))
        n_mesh = int(params.get('n_mesh', 300))
        rmin = float(params.get('rmin', 1e-3))
        tol = float(params.get('tol', 1e-9))
        
        # Validation
        if rmax < 5.0 or rmax > 200.0:
            return {'success': False,
                    'message': f'rmax = {rmax} out of range [5, 200].'}
        if R_h <= 0 or R_h > 10:
            return {'success': False,
                    'message': f'R_h = {R_h} out of range (0, 10].'}
        if n_mesh < 50 or n_mesh > 2000:
            return {'success': False,
                    'message': f'n_mesh = {n_mesh} out of range [50, 2000].'}
        
        # Solve BVP
        sol = solve_bvp_at(rmin, rmax, R_h, n_mesh=n_mesh, tol=tol)
        
        if not sol.success:
            return {
                'success': False,
                'message': f'BVP did not converge: {sol.message}',
                'params_used': {
                    'rmin': rmin, 'rmax': rmax, 'R_h': R_h, 'n_mesh': n_mesh,
                },
            }
        
        # Evaluate on a fine grid
        r_fine = np.linspace(rmin, rmax, 5000)
        y_fine = sol.sol(r_fine)
        f = y_fine[0]
        fp = y_fine[1]
        
        # Verify boundary behaviour
        f_at_origin = float(f[0])
        f_at_infty = float(f[-1])
        is_monotonic = bool(np.all(np.diff(f) <= 1e-6))
        
        # Compute integrals
        integrals = compute_integrals(r_fine, f, fp)
        
        # Extract Skyrme parameters
        skyrme = extract_skyrme_params(integrals)
        
        # Plotting data
        r_plot, f_plot, fp_plot = downsample_profile(r_fine, f, fp, n_target=200, r_max_plot=15.0)
        r_e, e2_density, e4_density = energy_density_profile(r_fine, f, fp, r_max_plot=8.0)
        
        return {
            'success': True,
            'message': 'Hedgehog soliton converged.',
            'params_used': {
                'rmin': rmin, 'rmax': rmax, 'R_h': R_h, 'n_mesh': n_mesh, 'tol': tol,
            },
            'profile_check': {
                'f_at_origin': f_at_origin,
                'f_at_origin_target': float(np.pi),
                'f_at_infty': f_at_infty,
                'f_at_infty_target': 0.0,
                'is_monotonic': is_monotonic,
            },
            'integrals': integrals,
            'skyrme_params': skyrme,
            'plot_profile': {
                'r': r_plot,
                'f': f_plot,
                'fp': fp_plot,
            },
            'plot_energy_density': {
                'r': r_e,
                'e2_density': e2_density,
                'e4_density': e4_density,
            },
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'message': f'Solver error: {type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }


# ── Presets ──────────────────────────────────────────────────────

PRESETS = {
    'standard': {
        'rmax': 30.0, 'R_h': 1.0, 'n_mesh': 300,
        'description': (
            'Standard converged configuration: rmax=30, R_h=1.0, n_mesh=300. '
            'Reproduces the published B=1 hedgehog with virial balance ~0.03%.'
        ),
    },
    'high_resolution': {
        'rmax': 50.0, 'R_h': 1.0, 'n_mesh': 500,
        'description': (
            'High-resolution: rmax=50, n_mesh=500. Better long-range behaviour '
            'and tighter virial balance, at the cost of ~2-3× longer runtime.'
        ),
    },
    'compact_guess': {
        'rmax': 20.0, 'R_h': 0.5, 'n_mesh': 250,
        'description': (
            'Compact initial guess (R_h=0.5, rmax=20). Tests robustness — should '
            'still converge to the same B=1 soliton. Faster but less accurate.'
        ),
    },
    'extended_range': {
        'rmax': 80.0, 'R_h': 1.5, 'n_mesh': 400,
        'description': (
            'Extended range (rmax=80, R_h=1.5). The hedgehog tail decays algebraically; '
            'larger rmax confirms the tail does not affect the energy integrals.'
        ),
    },
}


def get_preset(name):
    return PRESETS.get(name, PRESETS['standard'])
