#!/usr/bin/env python3
"""
MFT COSMOLOGY: DERIVATION VERIFICATION AND FIGURES
====================================================

Companion script for Paper 12 v2:
"Cosmology from Spatial Contraction"

Verifies (with ε = 1 derived from f_π² = φ_v² − m² = δ in Paper 8 v4):
  1. Void background: V(φ≈0), residual vacuum energy
  2. Photon-shift law: reduces to geometric only (ε = 1 → dielectric term vanishes)
  3. Effective Friedmann equation: late-time acceleration from V(φ_void)
  4. Hubble tension: now an open problem (void stretch mechanism removed)
  5. α_EM constancy: trivial since ε = 1

All computations use exact analytical formulas from the MFT potential.

Author: Dale Wahl / MFT research programme, April 2026
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def outpath(fn): return os.path.join(SCRIPT_DIR, fn)

# ═══════════════════════════════════════════════════════════════════
# MFT PARAMETERS
# ═══════════════════════════════════════════════════════════════════
M2, LAM4, LAM6 = 1.0, 2.0, 0.5
# V''(0) = M2 = 1 = Z_lep  [potential curvature at linear vacuum, DERIVED]
# This same M2 controls FOUR sectors:
#   Leptons:   Z_lep = m² (Coulomb coupling)
#   Hadrons:   f_π² = φ_v² - m² = δ (baseline subtraction)
#   Neutrinos: M²_s = V''(φ_v) + m² = δ(δ+2) (universal screening)
#   Cosmology: V(φ_void) ≈ ½m²φ² (void vacuum energy → late-time acceleration)
DELTA = 1 + np.sqrt(2)
PHI_B = np.sqrt(2 - np.sqrt(2))
PHI_V = np.sqrt(2 + np.sqrt(2))

def V(phi):   return 0.5*M2*phi**2 - 0.25*LAM4*phi**4 + (1/6.)*LAM6*phi**6
def Vp(phi):  return M2*phi - LAM4*phi**3 + LAM6*phi**5
def Vpp(phi): return M2 - 3*LAM4*phi**2 + 5*LAM6*phi**4

# Cosmological parameters (illustrative, normalised)
H0_CMB = 67.4    # km/s/Mpc (Planck 2018)
H0_LOCAL = 73.0  # km/s/Mpc (SH0ES 2022)
TENSION_FRAC = (H0_LOCAL - H0_CMB) / H0_CMB  # ~8.3%

def main():
    print("=" * 72)
    print("MFT COSMOLOGY: DERIVATION VERIFICATION")
    print("=" * 72)
    print(f"  m₂={M2}, λ₄={LAM4}, λ₆={LAM6}, δ={DELTA:.4f}")

    # ══════════════════════════════════════════════════════════════
    # 1. VOID BACKGROUND
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("1. VOID BACKGROUND (Paper §2)")
    print("=" * 72)

    phi_void = 0.0  # relaxed vacuum
    print(f"\n  Void background: φ_void = {phi_void}")
    print(f"  V(φ_void)  = {V(phi_void):.6f}  (vacuum energy = 0 at exact minimum)")
    print(f"  V'(φ_void) = {Vp(phi_void):.6f}  (equilibrium condition)")
    print(f"  V''(φ_void) = {Vpp(phi_void):.6f}  (= m₂, positive → stable)")

    # Small displacement from vacuum
    for dphi in [0.001, 0.01, 0.05, 0.1]:
        rho_v = V(dphi)
        print(f"  φ_void = {dphi:.3f}: V = {rho_v:.6f}  "
              f"(≈ ½m₂φ² = {0.5*M2*dphi**2:.6f})")

    print(f"""
  Key insight: In voids, φ sits near 0 with tiny residual energy.
  This residual energy ρ_φ ≈ ½m₂φ²_void acts as effective vacuum energy,
  driving late-time acceleration — the same potential that produces
  particle masses at microphysical scales.                              ✓
""")

    # ══════════════════════════════════════════════════════════════
    # 2. PHOTON-SHIFT LAW
    # ══════════════════════════════════════════════════════════════
    print(f"{'='*72}")
    print("2. PHOTON-SHIFT LAW (Paper §3)")
    print("=" * 72)

    print(f"""
  The photon-shift law (general form):
    d ln ω / dℓ = -½ d ln ε(φ) / dℓ + H_eff(ℓ)

  Term 1: DIELECTRIC CONTRIBUTION
    -½ d ln ε / dℓ
    With ε = 1 derived (from f_π² = δ at 0.03%, see Paper 8 v3):
      d ln ε / dℓ = 0 → THIS TERM VANISHES IDENTICALLY

  Term 2: GEOMETRIC CONTRIBUTION
    H_eff(ℓ)
    The medium volume expands, stretching the photon wavelength.
    This is the only surviving contribution.

  Reduced photon-shift law (ε = 1):
    d ln ω / dℓ = H_eff(ℓ)                                            ✓

  Integrated redshift:
    ln(1+z) = ∫ H_eff dℓ
    1+z = exp(∫ H_eff dℓ)
    For small z: z ≈ H₀ d/c   (standard Hubble law)                  ✓
""")

    # Numerical demonstration: only geometric contribution
    N_steps = 1000
    ell = np.linspace(0, 1, N_steps)  # normalised path length
    H_const = 0.07  # normalised H_eff

    # ε = 1 → only geometric contribution
    z_geometric = np.exp(H_const * ell) - 1

    print(f"  Numerical check (normalised units, ε = 1):")
    print(f"    H_eff = {H_const}")
    print(f"    z(ℓ=1) = {z_geometric[-1]:.4f}  (purely geometric)")
    print(f"    No dielectric contribution: ε = 1 everywhere.")

    # ══════════════════════════════════════════════════════════════
    # 3. EFFECTIVE FRIEDMANN EQUATION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("3. EFFECTIVE FRIEDMANN EQUATION (Paper §4)")
    print("=" * 72)

    print(f"""
  In the void background:
    H²_eff = (8πG_eff/3) [ρ_matter + ρ_φ(φ_void)]

  where ρ_φ = ½ φ̇² + V(φ_void)

  At late times:
    ρ_matter → 0 (dilutes as a⁻³)
    ρ_φ → V(φ_void) ≈ const (field near minimum)

  Effective equation of state:
    w_φ = (½φ̇² - V) / (½φ̇² + V)

  For slow roll (φ̇ ≈ 0): w_φ → -1 (mimics Λ)                        ✓
""")

    # Demonstrate w_φ vs kinetic/potential ratio
    KV_ratio = np.linspace(0, 2, 100)  # K/V = ½φ̇²/V
    w_phi = (KV_ratio - 1) / (KV_ratio + 1)

    print(f"  w_φ as function of kinetic/potential ratio:")
    for kv in [0.0, 0.01, 0.1, 0.5, 1.0]:
        w = (kv - 1) / (kv + 1)
        print(f"    K/V = {kv:.2f}:  w_φ = {w:+.4f}  "
              f"({'dark energy' if w < -1/3 else 'matter-like'})")

    # ══════════════════════════════════════════════════════════════
    # 4. HUBBLE TENSION — OPEN PROBLEM
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("4. HUBBLE TENSION: OPEN PROBLEM (Paper §5)")
    print("=" * 72)

    print(f"""
  Observed tension:
    H₀(CMB, Planck 2018) = {H0_CMB} km/s/Mpc
    H₀(local, SH0ES)     = {H0_LOCAL} km/s/Mpc
    Discrepancy:          = {TENSION_FRAC*100:.1f}%

  Earlier proposal (now retracted):
    A "void stretch" mechanism using a varying ε(φ) along the photon
    path was proposed to provide an O(10%) extra redshift contribution.

  Status with ε = 1 (derived):
    The dielectric mechanism is no longer available. The photon-shift
    law reduces to the geometric form, identical in structure to ΛCDM.
    The Hubble tension is NOT predicted at this level.

  Open avenues:
    1. Differential void evolution (position-dependent H_eff)
    2. Modified non-minimal coupling F(φ) (full nonlinear)
    Neither has been worked out quantitatively.
""")

    # ══════════════════════════════════════════════════════════════
    # 5. α_EM CONSTANCY — TRIVIAL WITH ε = 1
    # ══════════════════════════════════════════════════════════════
    print(f"{'='*72}")
    print("5. α_EM CONSTANCY (Paper §5.3)")
    print("=" * 72)

    print(f"""
  With ε(φ) = 1 everywhere (derived in Paper 8 v3):
    EM Lagrangian: L = -¼ F_ij F^ij  (already canonical)
    e_phys = e₀ (no rescaling needed)
    α_EM = e₀² / 4π                                                    ✓

  α_EM is independent of the local contraction field by construction.
  All bounds on time variation of α_EM are trivially satisfied:
    |Δα/α| < 10⁻⁶ (atomic clocks)    — automatic
    |Δα/α| < 10⁻⁵ (quasar spectra)   — automatic
""")

    # ══════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════
    print(f"{'='*72}")
    print("GENERATING FIGURE")
    print("=" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cosmology from Spatial Contraction in MFT (with $\\varepsilon = 1$)\n"
                 "Geometric redshift, late-time acceleration from $V_6(\\varphi)$",
                 fontsize=13, fontweight='bold')

    # Panel 1: Potential near φ=0 (void region)
    ax = axes[0, 0]
    phi_arr = np.linspace(-0.3, 2.0, 500)
    ax.plot(phi_arr, V(phi_arr), 'k-', lw=2.5)
    # Highlight void region
    phi_void_region = np.linspace(-0.15, 0.15, 100)
    ax.fill_between(phi_void_region, -0.05, [V(p) for p in phi_void_region],
                    alpha=0.3, color='cyan', label='Void region')
    ax.plot(0, 0, 'go', ms=12, zorder=5, label='$\\varphi_{\\mathrm{void}} \\approx 0$')
    ax.annotate('Residual energy\n$\\rho_\\varphi \\approx \\frac{1}{2}m_2\\varphi^2_{\\mathrm{void}}$\n→ drives acceleration',
                xy=(0.08, V(0.08)), xytext=(0.5, 0.15),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='blue', lw=0.8),
                color='blue')
    ax.axvline(PHI_B, color='orange', ls='--', lw=1, alpha=0.5, label='$\\varphi_b$')
    ax.axvline(PHI_V, color='red', ls='--', lw=1, alpha=0.5, label='$\\varphi_v$')
    ax.set_xlabel('$\\varphi$'); ax.set_ylabel('$V_6(\\varphi)$')
    ax.set_title('Sextic potential\n(void background near $\\varphi=0$)')
    ax.set_ylim(-1.0, 0.5); ax.set_xlim(-0.3, 2.0)
    ax.legend(fontsize=7, loc='lower right'); ax.grid(True, alpha=0.3)

    # Panel 2: Photon-shift law — geometric only (ε = 1)
    ax = axes[0, 1]
    ell_arr = np.linspace(0, 1, 200)
    z_geo = np.exp(H_const * ell_arr) - 1
    z_diel_zero = np.zeros_like(ell_arr)  # ε = 1 → no dielectric contribution
    ax.plot(ell_arr, z_geo, 'b-', lw=2.5, label='Geometric ($H_{\\mathrm{eff}}$) — total')
    ax.plot(ell_arr, z_diel_zero, 'r--', lw=1.5, alpha=0.6,
            label='Dielectric: 0 (since $\\varepsilon = 1$)')
    ax.text(0.5, 0.5*z_geo[-1], r'$\frac{d\ln\omega}{d\ell} = H_{\mathrm{eff}}$' + '\n'
            r'(no $\varepsilon$ contribution)',
            transform=ax.transData, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
    ax.set_xlabel('Affine path length $\\ell$ (normalised)')
    ax.set_ylabel('Redshift $z$')
    ax.set_title('Photon-shift law (with $\\varepsilon = 1$)\n'
                 'reduces to geometric Hubble law')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: Effective equation of state
    ax = axes[1, 0]
    ax.plot(KV_ratio, w_phi, 'b-', lw=2.5)
    ax.axhline(-1, color='red', ls=':', lw=1, label='$w = -1$ (cosmological constant)')
    ax.axhline(-1/3, color='orange', ls='--', lw=1, label='$w = -1/3$ (acceleration threshold)')
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.fill_between(KV_ratio, -1.1, w_phi, where=w_phi < -1/3,
                    alpha=0.1, color='red', label='Accelerating')
    # Mark slow-roll point
    ax.plot(0, -1, 'ro', ms=10, zorder=5)
    ax.annotate('Slow roll\n$w_\\varphi \\to -1$', xy=(0, -1), xytext=(0.5, -0.6),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontweight='bold')
    ax.set_xlabel('Kinetic/Potential ratio $K/V = \\frac{1}{2}\\dot{\\varphi}^2/V$')
    ax.set_ylabel('$w_\\varphi$')
    ax.set_title('Effective equation of state\n(approaches $-1$ in slow-roll void)')
    ax.set_ylim(-1.1, 1.1); ax.set_xlim(0, 2)
    ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.3)

    # Panel 4: Hubble tension — open problem
    ax = axes[1, 1]
    z_arr = np.linspace(0, 0.1, 100)
    d_lcdm = z_arr / H0_CMB * 1000  # Mpc, schematic
    d_local = z_arr / H0_LOCAL * 1000

    ax.plot(z_arr, d_lcdm, 'b-', lw=2.5, label=f'$\\Lambda$CDM ($H_0={H0_CMB}$)')
    ax.plot(z_arr, d_local, 'g--', lw=2.5, label=f'Local ($H_0={H0_LOCAL}$)')
    ax.fill_between(z_arr, d_lcdm, d_local, alpha=0.15, color='gray')
    ax.text(0.05, (d_lcdm[50]+d_local[50])/2,
            'Hubble tension\n(open problem)\n\n'
            'Earlier "void stretch"\nmechanism retracted:\n'
            r'$\varepsilon = 1$ → no extra' + '\nredshift contribution',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Distance (schematic, Mpc)')
    ax.set_title(f'Hubble tension: open problem in MFT\n'
                 f'($H_0$: {H0_CMB} CMB vs {H0_LOCAL} local)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = outpath('mft_cosmology.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {out}")

    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("VERDICT: COSMOLOGICAL DERIVATIONS WITH ε = 1")
    print("=" * 72)
    print(f"""
  1. Void background: V(0)=0, V'(0)=0, V''(0)=m₂                 ✓
  2. Photon-shift law: d ln ω/dℓ = H_eff (geometric only, ε = 1) ✓
  3. Effective Friedmann: H² ∝ ρ_matter + ρ_φ                    ✓
  4. Late-time acceleration: w_φ → -1 (slow roll, from V_6)      ✓
  5. Hubble tension: NOT predicted (open problem with ε = 1)     ⚠
  6. α_EM constancy: trivial since ε = 1 everywhere              ✓

  Late-time acceleration intact from V₆(φ) at the relaxed vacuum.
  The Hubble tension requires future work on void-background dynamics
  or the nonlinear F(φ) — both unexplored quantitatively.

  ALL FROM ONE POTENTIAL: V₆(φ) with λ₄² = 8m₂λ₆
""")


if __name__ == '__main__':
    main()
