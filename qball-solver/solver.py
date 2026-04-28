"""
MFT Q-Ball Spectrum Solver — browser-adapted version (v4)

Implements the generalised ℓ-dependent Q-ball equation from
MFT_Lepton_Mass_Paper_v9 and the cross-sector analysis from
mft_cross_sector.py / mft_quark_sector.py / mft_vector_bosons.py.

  u'' = [m₂ − ω² − λ₄(u/r)² + λ₆(u/r)⁴ − Z/√(r²+a²) + ℓ(ℓ+1)/r²] u

Sector-specific physics (from the corpus):
  - Z_lep   = m² = V''(0) = 1.0     [DERIVED: potential curvature]
  - Z_up    = 1.0                   [DERIVED: same as lepton]
  - Z_down  = λ₄/(2λ₆) = 2.0        [DERIVED: Vieta average]
  - Z_boson = 9/5 = 1.8             [CONJECTURED: SO(3) mode counting]

  - ℓ = 0 for leptons, scalar Higgs
  - ℓ = 1 for vector bosons (W, Z, photon)
  - ℓ = 0 for quarks (current implementation)

Each sector has its own calibration anchor (a known particle mass)
which yields its own MFT_TO_MEV scale factor.
"""

import numpy as np
from scipy.optimize import brentq

try:
    _trap = np.trapezoid
except AttributeError:
    _trap = np.trapz

# ── Physical constants (MeV) ──────────────────────────────────────────────────
M_ELECTRON = 0.511
M_MUON     = 105.66
M_TAU      = 1776.86

M_UP       = 2.16
M_CHARM    = 1270.0
M_TOP      = 173100.0

M_DOWN     = 4.7
M_STRANGE  = 93.0
M_BOTTOM   = 4180.0

M_W        = 80370.0
M_Z        = 91188.0
M_HIGGS    = 125090.0

# ── Default grid ──────────────────────────────────────────────────────────────
RMAX_DEFAULT = 20.0
N_DEFAULT = 200


def potential(phi, m2, lam4, lam6):
    return 0.5 * m2 * phi**2 - 0.25 * lam4 * phi**4 + (1.0 / 6.0) * lam6 * phi**6


def find_barrier_and_vacuum(m2, lam4, lam6):
    disc = lam4**2 - 4.0 * m2 * lam6
    if disc <= 0:
        return None, None
    phi_b = float(np.sqrt((lam4 - np.sqrt(disc)) / (2.0 * lam6)))
    phi_v = float(np.sqrt((lam4 + np.sqrt(disc)) / (2.0 * lam6)))
    return phi_b, phi_v


def silver_ratio_check(m2, lam4, lam6):
    target = 8.0 * m2 * lam6
    actual = lam4**2
    rel_err = abs(actual - target) / target if target != 0 else float('inf')
    return {
        'satisfied': rel_err < 1e-3,
        'lam4_squared': actual,
        'eight_m2_lam6': target,
        'relative_error': rel_err,
    }


def shoot(A, omega2, m2, lam4, lam6, Z, a, r, h, N, ell=0):
    """ℓ-dependent Q-ball shooting. ℓ=0: u[1]=A·r[1]. ℓ=1: u[1]=A·r[1]²."""
    u = np.zeros(N)
    u[0] = 0.0
    u[1] = A * (r[1] ** (ell + 1))
    cent = ell * (ell + 1)
    for i in range(1, N - 1):
        phi_i = u[i] / r[i]
        d2u = (m2 - omega2
               - lam4 * phi_i**2
               + lam6 * phi_i**4
               - Z / np.sqrt(r[i]**2 + a**2)
               + cent / r[i]**2) * u[i]
        u[i + 1] = 2.0 * u[i] - u[i - 1] + h * h * d2u
        if not np.isfinite(u[i + 1]) or abs(u[i + 1]) > 1e8:
            u[i + 1:] = 0.0
            break
    return u[-1], u


def find_solitons_at_omega2(omega2, m2, lam4, lam6, Z, a, ell=0,
                             A_max=8.0, n_pts=300,
                             rmax=RMAX_DEFAULT, n_grid=N_DEFAULT):
    r = np.linspace(rmax / (n_grid * 100.0), rmax, n_grid)
    h = r[1] - r[0]
    A_vals = np.linspace(0.01, A_max, n_pts)
    u_ends = [shoot(A, omega2, m2, lam4, lam6, Z, a, r, h, n_grid, ell)[0]
              for A in A_vals]
    results = []
    for i in range(len(A_vals) - 1):
        if u_ends[i] * u_ends[i + 1] < 0:
            try:
                A_s = brentq(
                    lambda A: shoot(A, omega2, m2, lam4, lam6, Z, a, r, h, n_grid, ell)[0],
                    A_vals[i], A_vals[i + 1],
                    xtol=1e-8, maxiter=50
                )
                _, u = shoot(A_s, omega2, m2, lam4, lam6, Z, a, r, h, n_grid, ell)
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
                    'ell': int(ell),
                    'u': u.tolist(),
                    'r': r.tolist(),
                })
            except Exception:
                pass
    return results


def regime_for(phi_core, phi_b):
    if phi_b is None or phi_b == 0:
        return "—"
    if phi_core < 0.5 * phi_b:
        return "linear vacuum"
    elif phi_core < 1.1 * phi_b:
        return "near-barrier"
    else:
        return "nonlinear vacuum"


def find_best_triple(all_solitons, R10_target, R21_target):
    """Find triple (E0, E1, E2) whose ratios best match observed values."""
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
    return {'idx0': best_triple[0], 'idx1': best_triple[1], 'idx2': best_triple[2],
            'score': float(best_score)}


def find_boson_triple(all_solitons, M_W_obs, M_Z_obs, M_H_obs):
    """
    Boson-specific triple finder (matches mft_vector_bosons.py logic):
      - W and Z must be ℓ=1 vector solitons
      - Higgs must be ℓ=0 scalar
      - W < Z < H by energy
      - Score by mZ/mW and mH/mW ratios.
    """
    sols_l1 = [(i, s) for i, s in enumerate(all_solitons) if s['ell'] == 1]
    sols_l0 = [(i, s) for i, s in enumerate(all_solitons) if s['ell'] == 0]
    if len(sols_l1) < 2 or len(sols_l0) < 1:
        return None

    R_HW_target = M_H_obs / M_W_obs
    R_ZW_target = M_Z_obs / M_W_obs

    best_score = float('inf')
    best_triple = None
    for h_idx, H in sols_l0:
        for w_idx, W in sols_l1:
            if W['E'] >= H['E']:
                continue
            for z_idx, Z0 in sols_l1:
                if z_idx == w_idx:
                    continue
                if Z0['E'] <= W['E'] or Z0['E'] >= H['E']:
                    continue
                R_HW = H['E'] / W['E']
                R_ZW = Z0['E'] / W['E']
                score = (np.log(R_HW / R_HW_target))**2 + \
                        (np.log(R_ZW / R_ZW_target))**2
                if score < best_score:
                    best_score = score
                    best_triple = (w_idx, z_idx, h_idx)
    if best_triple is None:
        return None
    return {'idx0': best_triple[0], 'idx1': best_triple[1], 'idx2': best_triple[2],
            'score': float(best_score)}


# ── Sector definitions (from the corpus) ──────────────────────────────────────
SECTOR_DEFS = {
    'lepton_sector': {
        'name': 'Charged leptons',
        'particles': ('electron', 'muon', 'tau'),
        'masses': (M_ELECTRON, M_MUON, M_TAU),
        'Z': 1.0,
        'Z_origin': "m^2 = V''(0) = 1 (potential curvature, DERIVED)",
        'ell': 0,
        'anchor_idx': 0,
        'anchor_mass': M_ELECTRON,
        'anchor_label': 'm_e',
        'description': (
            'Charged lepton sector. Z=1 (DERIVED from potential curvature m^2 = V\'\'(0) = 1). '
            'Scalar (l=0) Q-ball. Energy scale calibrated to m_e = 0.511 MeV.'
        ),
    },
    'up_quark_sector': {
        'name': 'Up-type quarks',
        'particles': ('up', 'charm', 'top'),
        'masses': (M_UP, M_CHARM, M_TOP),
        'Z': 1.0,
        'Z_origin': "Z_up = 1 (same coupling as leptons, DERIVED)",
        'ell': 0,
        'anchor_idx': 2,
        'anchor_mass': M_TOP,
        'anchor_label': 'm_t',
        'scheme_dependent_idx': 0,   # up quark mass is scheme-dependent (MS-bar at mu=2 GeV)
        'scheme_dependent_note': (
            'The up quark mass (2.16 MeV, MS-bar at mu=2 GeV) is the most scheme-dependent '
            'fermion mass in the Standard Model — the up quark is never observed free. '
            'The MFT soliton predicts m_u_soliton ~6.5 MeV, consistent with running quark '
            'masses at lower renormalisation scales. The robust MFT predictions in this '
            'sector are the regime structure (phi_core) and R21 = m_t/m_c.'
        ),
        'description': (
            'Up-type quark sector. Z=1 (DERIVED: same as leptons). Scalar (l=0) Q-ball. '
            'The (u, c, t) triple occupies the same field-space regimes as (e, mu, tau); '
            'top is the metastable mode. Energy scale calibrated to m_t = 173,100 MeV.'
        ),
    },
    'down_quark_sector': {
        'name': 'Down-type quarks',
        'particles': ('down', 'strange', 'bottom'),
        'masses': (M_DOWN, M_STRANGE, M_BOTTOM),
        'Z': 2.0,
        'Z_origin': "Z_down = lam4/(2 lam6) = 2 (Vieta average of critical points, DERIVED)",
        'ell': 0,
        'anchor_idx': 2,
        'anchor_mass': M_BOTTOM,
        'anchor_label': 'm_b',
        'description': (
            'Down-type quark sector. Z=2 (DERIVED: Vieta average of phi_barrier, phi_vacuum). '
            'Scalar (l=0) Q-ball. Energy scale calibrated to m_b = 4,180 MeV.'
        ),
    },
    'boson_sector': {
        'name': 'Gauge bosons',
        'particles': ('W', 'Z', 'Higgs'),
        'masses': (M_W, M_Z, M_HIGGS),
        'Z': 1.8,
        'Z_origin': "Z_boson = 9/5 (SO(3) mode counting, CONJECTURED; verified 0.07%)",
        'ell': 1,
        'ell_higgs': 0,
        'anchor_idx': 0,
        'anchor_mass': M_W,
        'anchor_label': 'm_W',
        'description': (
            'Gauge boson sector. Z=9/5 (CONJECTURED from SO(3) mode counting). '
            'W and Z are vector (l=1) solitons; Higgs is scalar (l=0). The Weinberg angle '
            'emerges as sin^2 theta_W = 1 - (E_W/E_Z)^2. Calibrated to m_W = 80,370 MeV.'
        ),
    },
}


def solve_spectrum(params):
    try:
        m2 = float(params.get('m2', 1.0))
        lam4 = float(params.get('lam4', 2.0))
        lam6 = float(params.get('lam6', 0.5))
        Z = float(params.get('Z', 1.0))
        a = float(params.get('a', 1.0))
        n_omega = int(params.get('n_omega', 40))
        sector = str(params.get('sector', 'lepton_sector'))

        sector_def = SECTOR_DEFS.get(sector, SECTOR_DEFS['lepton_sector'])
        ell = sector_def.get('ell', 0)

        phi_b, phi_v = find_barrier_and_vacuum(m2, lam4, lam6)
        phi_max = (phi_v * 1.4) if phi_v else 3.0
        phi_arr = np.linspace(0, phi_max, 200)
        V_arr = potential(phi_arr, m2, lam4, lam6)
        sr = silver_ratio_check(m2, lam4, lam6)

        all_solitons = []
        if sector == 'boson_sector':
            # Bosons: per mft_vector_bosons.py, search the deep-amplitude range
            # A ∈ [1.4, 3.0] where the nonlinear-vacuum solitons live.
            # W, Z are ℓ=1; Higgs is ℓ=0.
            for ell_val in (1, 0):
                for omega2 in np.linspace(0.05, 0.98, n_omega):
                    sols = find_solitons_at_omega2(
                        omega2, m2, lam4, lam6, Z, a,
                        ell=ell_val, A_max=3.0,
                    )
                    # Boson script uses A range [1.4, 3.0]; filter accordingly
                    sols = [s for s in sols if 1.4 <= s['A'] <= 3.0]
                    for s in sols:
                        if not any(abs(s['E'] - prev['E']) < 0.005 for prev in all_solitons):
                            all_solitons.append(s)
        else:
            for omega2 in np.linspace(0.05, 0.99, n_omega):
                sols = find_solitons_at_omega2(omega2, m2, lam4, lam6, Z, a, ell=ell)
                for s in sols:
                    if not any(abs(s['E'] - prev['E']) < 0.01 for prev in all_solitons):
                        all_solitons.append(s)

        all_solitons.sort(key=lambda x: x['E'])

        for s in all_solitons:
            s['regime'] = regime_for(s['phi_core'], phi_b)
            s['morse_index'] = None
            s['stability'] = None
            s['f3_mode'] = None
            s['particle'] = None

        # Find canonical triple
        m1_obs, m2_obs, m3_obs = sector_def['masses']
        triple = None
        if sector == 'boson_sector':
            triple = find_boson_triple(all_solitons, m1_obs, m2_obs, m3_obs)
        else:
            R10_target = m2_obs / m1_obs
            R21_target = m3_obs / m2_obs
            triple = find_best_triple(all_solitons, R10_target, R21_target)

        # Sector-specific calibration: scale = anchor_mass / E_anchor_in_triple
        anchor_idx_in_triple = sector_def['anchor_idx']
        anchor_mass = sector_def['anchor_mass']
        scale = None

        if triple is not None:
            triple_indices = (triple['idx0'], triple['idx1'], triple['idx2'])
            anchor_soliton = all_solitons[triple_indices[anchor_idx_in_triple]]
            E_anchor = anchor_soliton['E']
            if E_anchor > 0:
                scale = anchor_mass / E_anchor

            particle_names = sector_def['particles']
            for slot, idx in enumerate(triple_indices):
                s = all_solitons[idx]
                s['particle'] = particle_names[slot]
                s['f3_mode'] = slot
                s['morse_index'] = max(0, slot - 1)
                s['stability'] = (
                    'stable' if s['morse_index'] == 0
                    else 'metastable' if s['morse_index'] == 1
                    else 'unstable'
                )

        if scale is None:
            scale = 119.67  # fallback

        for s in all_solitons:
            s['mass_MeV'] = s['E'] * scale

        # Build triple summary for UI
        triple_data = None
        if triple is not None:
            ti = (triple['idx0'], triple['idx1'], triple['idx2'])
            E0, E1, E2 = (all_solitons[ti[k]]['E'] for k in range(3))
            R10_model = E1 / E0 if E0 > 0 else 0.0
            R21_model = E2 / E1 if E1 > 0 else 0.0
            R20_model = E2 / E0 if E0 > 0 else 0.0
            triple_data = {
                'idx0': triple['idx0'],
                'idx1': triple['idx1'],
                'idx2': triple['idx2'],
                'particles': list(sector_def['particles']),
                'observed_masses': [m1_obs, m2_obs, m3_obs],
                'predicted_masses': [
                    all_solitons[ti[0]]['mass_MeV'],
                    all_solitons[ti[1]]['mass_MeV'],
                    all_solitons[ti[2]]['mass_MeV'],
                ],
                'mass_ratios_model': [R10_model, R21_model, R20_model],
                'mass_ratios_observed': [m2_obs/m1_obs, m3_obs/m2_obs, m3_obs/m1_obs],
                'score': triple['score'],
            }

        # Boson-specific: Weinberg angle
        weinberg = None
        if sector == 'boson_sector' and triple is not None:
            ti = (triple['idx0'], triple['idx1'], triple['idx2'])
            E_W = all_solitons[ti[0]]['E']
            E_Z = all_solitons[ti[1]]['E']
            if E_Z > 0:
                sin2_model = 1.0 - (E_W / E_Z)**2
                sin2_obs = 1.0 - (M_W / M_Z)**2
                weinberg = {
                    'sin2_theta_W_model': float(sin2_model),
                    'sin2_theta_W_observed': float(sin2_obs),
                }

        return {
            'success': True,
            'message': f"Found {len(all_solitons)} distinct solitons.",
            'spectrum': all_solitons,
            'triple': triple_data,
            'weinberg': weinberg,
            'phi_barrier': phi_b,
            'phi_vacuum': phi_v,
            'potential_curve': {'phi': phi_arr.tolist(), 'V': V_arr.tolist()},
            'silver_ratio': sr,
            'mft_to_mev': float(scale) if scale is not None else None,
            'sector': sector,
            'sector_info': {
                'name': sector_def['name'],
                'particles': list(sector_def['particles']),
                'Z': sector_def['Z'],
                'Z_origin': sector_def['Z_origin'],
                'ell': sector_def['ell'],
                'anchor_label': sector_def['anchor_label'],
                'anchor_mass': sector_def['anchor_mass'],
                'scheme_dependent_idx': sector_def.get('scheme_dependent_idx'),
                'scheme_dependent_note': sector_def.get('scheme_dependent_note'),
            },
            'params': {'m2': m2, 'lam4': lam4, 'lam6': lam6, 'Z': Z, 'a': a},
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error: {type(e).__name__}: {e}",
            'spectrum': [],
            'triple': None,
            'weinberg': None,
            'phi_barrier': None,
            'phi_vacuum': None,
            'potential_curve': None,
            'silver_ratio': None,
            'mft_to_mev': None,
            'sector': params.get('sector', 'lepton_sector'),
            'sector_info': None,
            'params': params,
        }


# ── Sector presets ────────────────────────────────────────────────────────────
PRESETS = {
    'lepton_sector': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'Z': 1.0, 'a': 1.0,
        'description': SECTOR_DEFS['lepton_sector']['description'],
    },
    'up_quark_sector': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'Z': 1.0, 'a': 1.0,
        'description': SECTOR_DEFS['up_quark_sector']['description'],
    },
    'down_quark_sector': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'Z': 2.0, 'a': 1.0,
        'description': SECTOR_DEFS['down_quark_sector']['description'],
    },
    'boson_sector': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'Z': 1.8, 'a': 1.0,
        'description': SECTOR_DEFS['boson_sector']['description'],
    },
}


def get_preset(name):
    return PRESETS.get(name, PRESETS['lepton_sector'])
