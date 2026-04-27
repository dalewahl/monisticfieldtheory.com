"""
MFT Q-Ball Solver — browser-adapted version

Adapted from mft_qball_lepton_masses.py for execution under Pyodide.
Provides a clean solve(params) -> dict API callable from JavaScript.
"""

import numpy as np
from scipy.optimize import brentq

# NumPy 2.0+ renamed trapz to trapezoid; handle both.
try:
    _trap = np.trapezoid
except AttributeError:
    _trap = np.trapz

# Calibration constant (MeV per MFT energy unit)
# This is hardwired: m_e = 0.511 MeV ↔ E0 ≈ 0.00427 MFT units → MFT_TO_MEV = 119.67
M_ELECTRON_MEV = 0.511

# Default grid (matches mft_qball_lepton_masses.py)
RMAX_DEFAULT = 20.0
N_DEFAULT = 200


def potential(phi, m2, lam4, lam6):
    """V(φ) = m2/2·φ² − lam4/4·φ⁴ + lam6/6·φ⁶"""
    return 0.5 * m2 * phi**2 - 0.25 * lam4 * phi**4 + (1.0 / 6.0) * lam6 * phi**6


def find_barrier_and_vacuum(m2, lam4, lam6):
    """Find φ_barrier (local max of V) and φ_vacuum (second local min of V)."""
    disc = lam4**2 - 4.0 * m2 * lam6
    if disc <= 0:
        return None, None
    phi_b = float(np.sqrt((lam4 - np.sqrt(disc)) / (2.0 * lam6)))
    phi_v = float(np.sqrt((lam4 + np.sqrt(disc)) / (2.0 * lam6)))
    return phi_b, phi_v


def shoot(A, omega2, m2, lam4, lam6, Z, a, r, h, N):
    """
    Shoot the nonlinear Q-ball soliton equation outward from r=0.
    u ~ A*r near origin (so φ_core = A at r→0).
    Returns (u_endpoint, u_array).
    """
    u = np.zeros(N)
    u[0] = 0.0
    u[1] = A * r[1]
    for i in range(1, N - 1):
        phi_i = u[i] / r[i]
        d2u = (m2 - omega2
               - lam4 * phi_i**2
               + lam6 * phi_i**4
               - Z / np.sqrt(r[i]**2 + a**2)) * u[i]
        u[i + 1] = 2.0 * u[i] - u[i - 1] + h * h * d2u
        if not np.isfinite(u[i + 1]) or abs(u[i + 1]) > 1e8:
            u[i + 1:] = 0.0
            break
    return u[-1], u


def find_solitons_at_omega2(omega2, m2, lam4, lam6, Z, a, A_max=8.0, n_pts=300,
                             rmax=RMAX_DEFAULT, n_grid=N_DEFAULT):
    """
    At fixed omega2, scan amplitude A and find all soliton solutions
    (where u_endpoint changes sign).

    Returns: list of dicts with E, Q, omega2, A, n_nodes, phi_core, u, r.
    """
    r = np.linspace(rmax / (n_grid * 100.0), rmax, n_grid)
    h = r[1] - r[0]

    A_vals = np.linspace(0.01, A_max, n_pts)
    u_ends = [shoot(A, omega2, m2, lam4, lam6, Z, a, r, h, n_grid)[0]
              for A in A_vals]
    results = []

    for i in range(len(A_vals) - 1):
        if u_ends[i] * u_ends[i + 1] < 0:
            try:
                A_s = brentq(
                    lambda A: shoot(A, omega2, m2, lam4, lam6, Z, a, r, h, n_grid)[0],
                    A_vals[i], A_vals[i + 1],
                    xtol=1e-8, maxiter=50
                )
                _, u = shoot(A_s, omega2, m2, lam4, lam6, Z, a, r, h, n_grid)
                Q = float(_trap(u**2, r))
                E = omega2 * Q
                nc = int(np.sum(np.diff(np.sign(u[:int(0.95 * n_grid)])) != 0))
                phi_core = float(u[1] / r[1])
                results.append({
                    'E': float(E),
                    'Q': Q,
                    'omega2': float(omega2),
                    'A': float(A_s),
                    'n_nodes': nc,
                    'phi_core': phi_core,
                    'u': u.tolist(),
                    'r': r.tolist(),
                })
            except Exception:
                pass

    return results


def regime_for(phi_core, phi_b):
    """Classify a soliton by where its core sits relative to the barrier."""
    if phi_b is None or phi_b == 0:
        return "—"
    if phi_core < 0.5 * phi_b:
        return "linear vacuum"
    elif phi_core < 1.1 * phi_b:
        return "near-barrier"
    else:
        return "nonlinear vacuum"


def solve(params):
    """
    Main entry point called from JavaScript.

    params: dict with keys:
        m2, lam4, lam6, Z, a, omega2

    Returns: dict with keys:
        success: bool
        message: str (status or error)
        solitons: list of soliton dicts (if any found)
        phi_barrier, phi_vacuum: potential landscape (or None)
        potential_curve: {phi: [...], V: [...]} for plotting V(φ)
    """
    try:
        m2 = float(params.get('m2', 1.0))
        lam4 = float(params.get('lam4', 2.0))
        lam6 = float(params.get('lam6', 0.5))
        Z = float(params.get('Z', 1.0))
        a = float(params.get('a', 1.0))
        omega2 = float(params.get('omega2', 0.5))

        # Potential landscape
        phi_b, phi_v = find_barrier_and_vacuum(m2, lam4, lam6)

        # Sample V(φ) for plotting
        phi_max = (phi_v * 1.4) if phi_v else 3.0
        phi_arr = np.linspace(0, phi_max, 200)
        V_arr = potential(phi_arr, m2, lam4, lam6)

        # Find solitons at this omega2
        solitons = find_solitons_at_omega2(omega2, m2, lam4, lam6, Z, a)

        # Annotate solitons with regime classification
        for s in solitons:
            s['regime'] = regime_for(s['phi_core'], phi_b)

        return {
            'success': True,
            'message': f"Found {len(solitons)} soliton(s) at ω² = {omega2}",
            'solitons': solitons,
            'phi_barrier': phi_b,
            'phi_vacuum': phi_v,
            'potential_curve': {
                'phi': phi_arr.tolist(),
                'V': V_arr.tolist(),
            },
            'params': {
                'm2': m2, 'lam4': lam4, 'lam6': lam6,
                'Z': Z, 'a': a, 'omega2': omega2,
            },
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error: {type(e).__name__}: {e}",
            'solitons': [],
            'phi_barrier': None,
            'phi_vacuum': None,
            'potential_curve': None,
        }


def calibrate_to_electron(E_mft):
    """
    Convert MFT energy to MeV, calibrated so that the electron's E = 0.511 MeV.
    Note: this requires knowing E0 (the electron's MFT energy) for the chosen
    parameters. Default uses the canonical MFT_TO_MEV ≈ 119.67.
    """
    MFT_TO_MEV = 119.67  # canonical calibration at m2=1, lam4=2, lam6=0.5, Z=1
    return E_mft * MFT_TO_MEV


# Preset configurations matching the canonical MFT calibration.
# These omega2 values are determined by running mft_qball_lepton_masses.py and
# extracting the best-triple from a full omega2 scan.
PRESETS = {
    'electron': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'Z': 1.0, 'a': 1.0,
        'omega2': 0.8213,
        'description': 'Electron — ground-state soliton, linear vacuum (φ_core ≈ 0.022). Calibration target: 0.511 MeV.',
    },
    'muon': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'Z': 1.0, 'a': 1.0,
        'omega2': 0.6526,
        'description': 'Muon — second-family soliton, near-barrier (φ_core ≈ 0.71). Predicted: 104.3 MeV (1.2% from observed 105.7).',
    },
    'tau': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'Z': 1.0, 'a': 1.0,
        'omega2': 0.6767,
        'description': 'Tau — third-family metastable soliton, nonlinear vacuum (φ_core ≈ 1.93). Predicted: 1769 MeV (0.4% from observed 1777).',
    },
}


def get_preset(name):
    """Return preset parameters by name."""
    return PRESETS.get(name, PRESETS['electron'])
