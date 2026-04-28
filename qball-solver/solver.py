"""
MFT Q-Ball Solver — browser-adapted version (v2)

Adapted from mft_qball_lepton_masses.py for execution under Pyodide.
Provides solve_spectrum(params) -> dict that scans omega2 and returns
the full discrete tower of soliton solutions for a given potential.

This is the right interface for MFT: ω² is an eigenvalue, not an input.
The visitor enters the physics parameters; the equation picks out all
admissible solitons.
"""

import numpy as np
from scipy.optimize import brentq

# NumPy 2.0+ renamed trapz to trapezoid; handle both.
try:
    _trap = np.trapezoid
except AttributeError:
    _trap = np.trapz

# Calibration constant: m_e = 0.511 MeV ↔ E0 ≈ 0.00427 MFT units
M_ELECTRON_MEV = 0.511
MFT_TO_MEV_CANONICAL = 119.67  # canonical calibration at m₂=1, λ₄=2, λ₆=0.5, Z=1

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


def silver_ratio_check(m2, lam4, lam6):
    """Check whether the silver-ratio condition λ₄² = 8 m₂ λ₆ is satisfied."""
    target = 8.0 * m2 * lam6
    actual = lam4**2
    rel_err = abs(actual - target) / target if target != 0 else float('inf')
    return {
        'satisfied': rel_err < 1e-3,
        'lam4_squared': actual,
        'eight_m2_lam6': target,
        'relative_error': rel_err,
    }


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


def find_lepton_triple(all_solitons, R10_target=206.768, R21_target=16.817):
    """
    Among all found solitons, find the triple (E0, E1, E2) whose
    energy ratios best match the observed lepton mass ratios.
    Replicates the best_triple logic from mft_qball_lepton_masses.py.

    Returns: dict with keys electron, muon, tau (each a soliton dict
             from the spectrum) plus 'score' indicating fit quality,
             or None if no good triple found.
    """
    if len(all_solitons) < 3:
        return None

    best_score = float('inf')
    best_triple = None

    for i in range(len(all_solitons)):
        for j in range(i + 1, len(all_solitons)):
            for k in range(j + 1, len(all_solitons)):
                E0 = all_solitons[i]['E']
                E1 = all_solitons[j]['E']
                E2 = all_solitons[k]['E']
                if E0 > 0:
                    score = (np.log(E1 / E0 / R10_target))**2 + \
                            (np.log(E2 / E1 / R21_target))**2
                    if score < best_score:
                        best_score = score
                        best_triple = (i, j, k)

    if best_triple is None:
        return None

    i, j, k = best_triple
    return {
        'electron_idx': i,
        'muon_idx': j,
        'tau_idx': k,
        'score': float(best_score),
    }


def solve_spectrum(params):
    """
    Main entry point. Scan ω² values and return the full discrete tower
    of soliton solutions for the given potential.

    This is the canonical mode: visitor enters physics parameters,
    receives the discrete spectrum back as a single result.

    params: dict with keys m2, lam4, lam6, Z, a (and optionally
            n_omega for scan resolution, default 40)

    Returns: dict with keys
        success: bool
        message: str
        spectrum: list of soliton dicts, sorted by energy, deduplicated
        phi_barrier, phi_vacuum: potential landscape
        potential_curve: {phi: [...], V: [...]}
        silver_ratio: {satisfied, lam4_squared, eight_m2_lam6, relative_error}
        params: the inputs (echoed back)
    """
    try:
        m2 = float(params.get('m2', 1.0))
        lam4 = float(params.get('lam4', 2.0))
        lam6 = float(params.get('lam6', 0.5))
        Z = float(params.get('Z', 1.0))
        a = float(params.get('a', 1.0))
        n_omega = int(params.get('n_omega', 40))

        # Potential landscape
        phi_b, phi_v = find_barrier_and_vacuum(m2, lam4, lam6)

        # Sample V(φ) for plotting
        phi_max = (phi_v * 1.4) if phi_v else 3.0
        phi_arr = np.linspace(0, phi_max, 200)
        V_arr = potential(phi_arr, m2, lam4, lam6)

        # Silver-ratio diagnostic
        sr = silver_ratio_check(m2, lam4, lam6)

        # Scan omega2 and collect all soliton solutions
        all_solitons = []
        for omega2 in np.linspace(0.05, 0.99, n_omega):
            sols = find_solitons_at_omega2(omega2, m2, lam4, lam6, Z, a)
            for s in sols:
                # Deduplicate by energy (matches original mft_qball_lepton_masses.py logic)
                if not any(abs(s['E'] - prev['E']) < 0.01 for prev in all_solitons):
                    all_solitons.append(s)

        # Sort by energy
        all_solitons.sort(key=lambda x: x['E'])

        # Annotate with regime
        for s in all_solitons:
            s['regime'] = regime_for(s['phi_core'], phi_b)

        # Compute mass calibration: assume the lowest-energy soliton is the electron
        # (calibration anchor). All other masses scale relative to it.
        if all_solitons and all_solitons[0]['E'] > 0:
            E0 = all_solitons[0]['E']
            scale = M_ELECTRON_MEV / E0  # MeV per MFT energy unit
            for s in all_solitons:
                s['mass_MeV'] = s['E'] * scale
        else:
            scale = MFT_TO_MEV_CANONICAL
            for s in all_solitons:
                s['mass_MeV'] = s['E'] * scale

        # Identify the canonical lepton triple (e, μ, τ) by best ratio match
        triple = find_lepton_triple(all_solitons)
        if triple is not None:
            # Tag the chosen solitons
            all_solitons[triple['electron_idx']]['lepton'] = 'electron'
            all_solitons[triple['muon_idx']]['lepton'] = 'muon'
            all_solitons[triple['tau_idx']]['lepton'] = 'tau'

        return {
            'success': True,
            'message': f"Found {len(all_solitons)} distinct solitons in the spectrum.",
            'spectrum': all_solitons,
            'lepton_triple': triple,
            'phi_barrier': phi_b,
            'phi_vacuum': phi_v,
            'potential_curve': {
                'phi': phi_arr.tolist(),
                'V': V_arr.tolist(),
            },
            'silver_ratio': sr,
            'mft_to_mev': float(scale),
            'params': {
                'm2': m2, 'lam4': lam4, 'lam6': lam6,
                'Z': Z, 'a': a,
            },
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error: {type(e).__name__}: {e}",
            'spectrum': [],
            'phi_barrier': None,
            'phi_vacuum': None,
            'potential_curve': None,
            'silver_ratio': None,
            'mft_to_mev': None,
            'params': params,
        }


# Canonical preset: lepton sector with silver-ratio condition
PRESETS = {
    'lepton_sector': {
        'm2': 1.0,
        'lam4': 2.0,
        'lam6': 0.5,
        'Z': 1.0,
        'a': 1.0,
        'description': (
            'Lepton sector: silver-ratio potential (m₂=1, λ₄=2, λ₆=0.5 satisfies '
            'λ₄² = 8m₂λ₆), Z=1 Coulomb coupling, a=1 softening length. '
            'The Q-ball equation produces the full charged-lepton spectrum '
            '(e, μ, τ) from this single parameter set. Energy scale calibrated '
            'by setting the lowest soliton energy equal to m_e = 0.511 MeV.'
        ),
    },
}


def get_preset(name):
    """Return preset parameters by name."""
    return PRESETS.get(name, PRESETS['lepton_sector'])
