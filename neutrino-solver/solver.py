"""
MFT Neutrino Hierarchy Solver — browser-adapted

Implements:
  1. The δ⁴ - 1 hierarchy ratio prediction (closed-form)
  2. Absolute neutrino masses via one-loop gravitational self-energy
     with universal screening M²_s = V''(φ_v) + V''(0)
  3. Silver-ratio sensitivity: how the hierarchy changes when
     λ₄² ≠ 8 m₂ λ₆

Sources (corpus):
  - Hierarchy ratio: mft_neutrino_hierarchy.py (closed-form algebra)
  - Absolute scale:  mft_neutrino_masses.py (1-loop dim-reg integral)

THE PHYSICS
-----------
The MFT sextic potential V₆(φ) = m²φ²/2 - λ₄φ⁴/4 + λ₆φ⁶/6 has three
critical points: φ = 0, φ_b, φ_v.

When the silver-ratio condition λ₄² = 8 m² λ₆ holds:
  - V(0)     = 0
  - V(φ_b)   = 1/(3δ)         [δ = 1+√2, silver ratio]
  - V(φ_v)   = -δ/3
  - V''(0)   = 1
  - V''(φ_v) = 4δ

Family-of-Three associates one neutrino with each critical point.
The mass-squared splitting at each is proportional to V², giving:

  Δm²₃₂/Δm²₂₁ = δ⁴ - 1 = 16 + 12√2 ≈ 32.97

When the silver-ratio condition is violated, all of these formulas
reduce to direct evaluation: V at critical points, V'' at critical
points, ratios. No assumptions of the silver-ratio form.
"""

import numpy as np

try:
    _trap = np.trapezoid
except AttributeError:
    _trap = np.trapz


# ── Observed neutrino values (PDG 2024) ────────────────────────────
DM2_21_OBS = 7.53e-5      # eV² (solar)
DM2_32_OBS = 2.453e-3     # eV² (atmospheric, normal ordering)
RATIO_OBS = DM2_32_OBS / DM2_21_OBS   # ≈ 32.58

# ── Calibration ─────────────────────────────────────────────────────
M_E_MEV = 0.511
E_E_MFT_LEPTON = 0.00427
EV_PER_UNIT_LEPTON = (M_E_MEV / E_E_MFT_LEPTON) * 1e6   # ≈ 1.197×10⁸ eV

# ── Silver ratio constant (for reference) ──────────────────────────
DELTA_SILVER = 1.0 + np.sqrt(2.0)


# ─── Potential & critical points ──────────────────────────────────

def V_potential(phi, m2, lam4, lam6):
    """V₆(φ) = m²φ²/2 - λ₄φ⁴/4 + λ₆φ⁶/6"""
    return 0.5 * m2 * phi**2 - 0.25 * lam4 * phi**4 + (1.0/6.0) * lam6 * phi**6


def Vpp(phi, m2, lam4, lam6):
    """V''(φ) = m² - 3λ₄φ² + 5λ₆φ⁴"""
    return m2 - 3.0*lam4*phi**2 + 5.0*lam6*phi**4


def critical_points(m2, lam4, lam6):
    """
    Find φ_b (barrier) and φ_v (nonlinear vacuum).
    V'(φ) = m²φ - λ₄φ³ + λ₆φ⁵ = φ(m² - λ₄φ² + λ₆φ⁴)
    Need φ² roots of m² - λ₄φ² + λ₆φ⁴ = 0:
       φ² = (λ₄ ± √(λ₄² - 4 m² λ₆)) / (2 λ₆)
    """
    disc = lam4**2 - 4.0 * m2 * lam6
    if disc <= 0:
        return None, None
    phi_b_sq = (lam4 - np.sqrt(disc)) / (2.0 * lam6)
    phi_v_sq = (lam4 + np.sqrt(disc)) / (2.0 * lam6)
    if phi_b_sq <= 0 or phi_v_sq <= 0:
        return None, None
    phi_b = float(np.sqrt(phi_b_sq))
    phi_v = float(np.sqrt(phi_v_sq))
    return phi_b, phi_v


def silver_ratio_check(m2, lam4, lam6):
    """Return diagnostic on λ₄² = 8 m² λ₆ condition."""
    target = 8.0 * m2 * lam6
    actual = lam4**2
    rel_err = abs(actual - target) / target if target != 0 else float('inf')
    return {
        'satisfied': rel_err < 1e-3,
        'lam4_squared': float(actual),
        'eight_m2_lam6': float(target),
        'relative_error': float(rel_err),
        'rho': float(actual / (m2 * lam6)) if m2 * lam6 > 0 else None,
        # rho = λ₄²/(m² λ₆) — equals 8 at silver-ratio condition
    }


def field_ratio(phi_b, phi_v):
    """φ_v/φ_b — equals δ at the silver-ratio condition."""
    if phi_b is None or phi_v is None or phi_b == 0:
        return None
    return float(phi_v / phi_b)


# ─── Hierarchy ratio (the headline prediction) ────────────────────

def hierarchy_ratio(m2, lam4, lam6):
    """
    Compute Δm²₃₂/Δm²₂₁ = [V²(φ_v) - V²(φ_b)] / [V²(φ_b) - V²(0)]
    
    This is the V² mass-splitting mechanism: gravitational
    back-reaction at second order gives δm²(φ) ∝ V²(φ).
    """
    phi_b, phi_v = critical_points(m2, lam4, lam6)
    if phi_b is None:
        return None
    
    V_0 = 0.0  # V(0) = 0
    V_b = V_potential(phi_b, m2, lam4, lam6)
    V_v = V_potential(phi_v, m2, lam4, lam6)
    
    # V² values
    V2_0 = V_0**2
    V2_b = V_b**2
    V2_v = V_v**2
    
    # Splittings
    dV2_21 = V2_b - V2_0
    dV2_32 = V2_v - V2_b
    
    if dV2_21 == 0:
        return None
    
    return float(dV2_32 / dV2_21)


# ─── Absolute neutrino masses (one-loop with universal screening) ─

def absolute_masses(m2, lam4, lam6, beta=1.016e-4):
    """
    Compute m_ν₁, m_ν₂, m_ν₃ from one-loop gravitational self-energy:

      m_νᵢ = 3 β² |V(φᵢ)| × √I × (m_e/E_e)
    
    where I = 1/(32π M_s³) is the 3D one-loop integral with
    universal screening mass M_s² = V''(φ_v) + V''(0).
    
    Returns masses in eV.
    """
    phi_b, phi_v = critical_points(m2, lam4, lam6)
    if phi_b is None:
        return None
    
    V_b = V_potential(phi_b, m2, lam4, lam6)
    V_v = V_potential(phi_v, m2, lam4, lam6)
    
    # |V| for absolute mass formula
    abs_V_0 = 0.0
    abs_V_b = abs(V_b)
    abs_V_v = abs(V_v)
    
    # Universal screening mass
    Vpp_v = Vpp(phi_v, m2, lam4, lam6)
    Vpp_0 = Vpp(0.0, m2, lam4, lam6)  # = m²
    M2_screen = Vpp_v + Vpp_0
    
    if M2_screen <= 0:
        return None
    
    M_screen = np.sqrt(M2_screen)
    
    # 3D one-loop integral (dim reg)
    I_loop = 1.0 / (32.0 * np.pi * M_screen**3)
    sqrt_I = np.sqrt(I_loop)
    
    # Calibration scale
    eV_per_unit = EV_PER_UNIT_LEPTON
    
    # The three masses
    m_nu1 = 0.0
    m_nu2 = 3.0 * beta**2 * abs_V_b * sqrt_I * eV_per_unit
    m_nu3 = 3.0 * beta**2 * abs_V_v * sqrt_I * eV_per_unit
    
    # Mass-squared differences
    dm2_21 = m_nu2**2 - m_nu1**2
    dm2_32 = m_nu3**2 - m_nu2**2
    
    return {
        'm_nu1_eV': float(m_nu1),
        'm_nu2_eV': float(m_nu2),
        'm_nu3_eV': float(m_nu3),
        'sum_eV': float(m_nu1 + m_nu2 + m_nu3),
        'dm2_21_eV2': float(dm2_21),
        'dm2_32_eV2': float(dm2_32),
        'ratio_dm2': float(dm2_32 / dm2_21) if dm2_21 != 0 else None,
        'M_screen': float(M_screen),
        'M2_screen': float(M2_screen),
        'I_loop': float(I_loop),
        'beta': float(beta),
    }


# ─── Sensitivity scan: hierarchy ratio vs ρ = λ₄²/(m²λ₆) ─────────

def hierarchy_vs_rho(m2=1.0, n_pts=120, rho_min=4.5, rho_max=15.0):
    """
    Scan the dimensionless ratio ρ = λ₄²/(m²λ₆) from rho_min to rho_max,
    holding m² fixed. For each ρ, compute the hierarchy ratio.
    
    The silver-ratio condition is ρ = 8. We always include ρ=8 exactly
    in the scan grid so the curve passes through the silver-ratio point.
    """
    rhos_lin = np.linspace(rho_min, rho_max, n_pts).tolist()
    # Insert ρ=8 exactly if it's not already there
    if 8.0 not in rhos_lin and rho_min < 8.0 < rho_max:
        rhos_lin.append(8.0)
        rhos_lin.sort()
    
    ratios = []
    valid_rhos = []
    
    for rho in rhos_lin:
        # Pick lam6 = 0.5, then lam4² = ρ m² lam6 → lam4 = sqrt(ρ m² lam6)
        lam6 = 0.5
        lam4_sq = rho * m2 * lam6
        if lam4_sq <= 0:
            continue
        lam4 = np.sqrt(lam4_sq)
        # Need barrier to exist: lam4² > 4 m² lam6 → ρ > 4
        if lam4_sq <= 4.0 * m2 * lam6:
            continue
        r = hierarchy_ratio(m2, lam4, lam6)
        if r is not None:
            valid_rhos.append(float(rho))
            ratios.append(float(r))
    
    return {
        'rhos': valid_rhos,
        'ratios': ratios,
        'rho_silver': 8.0,
        'ratio_silver': float(DELTA_SILVER**4 - 1.0),
    }


# ─── Potential profile for plotting ──────────────────────────────

def potential_profile(m2, lam4, lam6, n_pts=300):
    """Return (phi_arr, V_arr) for plotting."""
    phi_b, phi_v = critical_points(m2, lam4, lam6)
    
    if phi_b is None:
        # No barrier — pick reasonable range
        phi_max = 2.5
    else:
        phi_max = phi_v * 1.2
    
    phi_min = -0.2
    phi_arr = np.linspace(phi_min, phi_max, n_pts)
    V_arr = V_potential(phi_arr, m2, lam4, lam6)
    
    return phi_arr.tolist(), V_arr.tolist()


# ─── Main solve() entry point ─────────────────────────────────────

def solve(params):
    """
    Top-level solver entry point invoked from JS.
    
    params: dict containing {
        'm2': float,
        'lam4': float,
        'lam6': float,
        'beta': float,
    }
    
    Returns a dict consumable by the JavaScript renderer.
    """
    try:
        m2 = float(params.get('m2', 1.0))
        lam4 = float(params.get('lam4', 2.0))
        lam6 = float(params.get('lam6', 0.5))
        beta = float(params.get('beta', 1.016e-4))
        
        # Validate
        if m2 <= 0 or lam6 <= 0:
            return {
                'success': False,
                'message': 'Need m² > 0 and λ₆ > 0 for stable potential.',
            }
        
        # Silver-ratio diagnostic
        sr = silver_ratio_check(m2, lam4, lam6)
        
        # Critical points
        phi_b, phi_v = critical_points(m2, lam4, lam6)
        if phi_b is None:
            return {
                'success': False,
                'message': (f"No double-well barrier exists: "
                           f"λ₄² = {lam4**2:.4f} ≤ 4 m² λ₆ = {4*m2*lam6:.4f}. "
                           f"Need λ₄² > 4 m² λ₆ for the potential to have a barrier."),
                'silver_ratio': sr,
            }
        
        # Field-space ratio
        f_ratio = field_ratio(phi_b, phi_v)
        
        # Potential values at critical points
        V_0 = 0.0
        V_b = float(V_potential(phi_b, m2, lam4, lam6))
        V_v = float(V_potential(phi_v, m2, lam4, lam6))
        
        # Curvatures at critical points (second derivative)
        Vpp_0 = float(Vpp(0.0, m2, lam4, lam6))   # = m²
        Vpp_b = float(Vpp(phi_b, m2, lam4, lam6))
        Vpp_v = float(Vpp(phi_v, m2, lam4, lam6))
        
        # Hierarchy ratio
        h_ratio = hierarchy_ratio(m2, lam4, lam6)
        
        # Absolute masses
        masses = absolute_masses(m2, lam4, lam6, beta=beta)
        
        # Sensitivity scan
        scan = hierarchy_vs_rho(m2=m2, n_pts=120)
        
        # Potential profile
        phi_arr, V_arr = potential_profile(m2, lam4, lam6)
        
        return {
            'success': True,
            'message': 'Computed successfully.',
            'params': {
                'm2': m2, 'lam4': lam4, 'lam6': lam6, 'beta': beta,
            },
            'silver_ratio': sr,
            'critical_points': {
                'phi_b': float(phi_b),
                'phi_v': float(phi_v),
                'field_ratio': f_ratio,
                'field_ratio_silver': float(DELTA_SILVER),
            },
            'V_at_crit': {
                'V_0': V_0,
                'V_b': V_b,
                'V_v': V_v,
            },
            'Vpp_at_crit': {
                'Vpp_0': Vpp_0,
                'Vpp_b': Vpp_b,
                'Vpp_v': Vpp_v,
                'M2_screen': Vpp_v + Vpp_0,
            },
            'hierarchy_ratio': {
                'predicted': h_ratio,
                'observed': float(RATIO_OBS),
                'silver_value': float(DELTA_SILVER**4 - 1.0),
                'rel_error_vs_obs': abs(h_ratio - RATIO_OBS) / RATIO_OBS if h_ratio else None,
            },
            'masses': masses,
            'sensitivity_scan': scan,
            'potential_curve': {
                'phi': phi_arr,
                'V': V_arr,
            },
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {type(e).__name__}: {e}',
        }


# ─── Presets ──────────────────────────────────────────────────────

PRESETS = {
    'silver_ratio_canonical': {
        'm2': 1.0, 'lam4': 2.0, 'lam6': 0.5, 'beta': 1.016e-4,
        'description': (
            'The canonical MFT silver-ratio condition: λ₄² = 8 m² λ₆ exactly. '
            'All thirteen silver-ratio manifestations active. Hierarchy ratio '
            'locked at δ⁴ - 1 = 32.97, matching observed 32.58 to 1.2%.'
        ),
    },
    'silver_ratio_off_5pct': {
        'm2': 1.0, 'lam4': 1.949, 'lam6': 0.5, 'beta': 1.016e-4,
        'description': (
            'Silver-ratio condition violated by 5% (lower λ₄). Watch the '
            'hierarchy ratio drift away from δ⁴ - 1.'
        ),
    },
    'silver_ratio_off_10pct': {
        'm2': 1.0, 'lam4': 1.897, 'lam6': 0.5, 'beta': 1.016e-4,
        'description': (
            'Silver-ratio condition violated by 10%. The hierarchy ratio no '
            'longer matches observation; this is the falsifiability test of '
            'the silver-ratio mechanism.'
        ),
    },
    'no_barrier': {
        'm2': 1.0, 'lam4': 1.0, 'lam6': 0.5, 'beta': 1.016e-4,
        'description': (
            'λ₄² < 4 m² λ₆: no double-well structure exists. The neutrino '
            'mechanism requires a double-well, which the Symmetric Back-'
            'Reaction theorem forces in the canonical case.'
        ),
    },
}


def get_preset(name):
    return PRESETS.get(name, PRESETS['silver_ratio_canonical'])
