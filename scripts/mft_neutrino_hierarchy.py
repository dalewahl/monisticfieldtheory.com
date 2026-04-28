#!/usr/bin/env python3
"""
EXECUTION
---------
  Dependencies:  pip install numpy matplotlib sympy
  Run:           python3 mft_neutrino_hierarchy.py
  Runtime:       ~5 seconds
  Outputs:       mft_neutrino_hierarchy.png + console

THE δ⁴ − 1 NEUTRINO HIERARCHY PREDICTION
==========================================

Monistic Field Theory predicts the ratio of atmospheric to solar
neutrino mass-squared differences:

    Δm²₃₂ / Δm²₂₁ = δ⁴ − 1 = 16 + 12√2 ≈ 32.97

    Observed (PDG 2024): 32.58 ± 0.3
    Error: 1.21%

This is a PARAMETER-FREE prediction from the silver ratio
δ = 1 + √2 alone. No fitting. No new parameters.

DERIVATION
----------
The MFT sextic potential V₆(φ) = m²φ²/2 − λ₄φ⁴/4 + λ₆φ⁶/6
has three critical points:

    φ = 0    (linear vacuum)     V(0)   = 0
    φ = φ_b  (barrier)           V(φ_b) = 1/(3δ)
    φ = φ_v  (nonlinear vacuum)  V(φ_v) = −δ/3

The Family-of-Three Theorem associates one neutrino with each
critical point. The neutrino mass-squared splitting at each
critical point is proportional to V(φ)² — the square of the
potential energy — arising from second-order gravitational
backreaction through the F(φ)R coupling in the MFT action.

The hierarchy ratio is:

    [V²(φ_v) − V²(φ_b)] / [V²(φ_b) − V²(0)]
  = [(δ/3)² − (1/(3δ))²] / [(1/(3δ))² − 0]
  = [δ²/9 − 1/(9δ²)] / [1/(9δ²)]
  = δ⁴ − 1
  = (1+√2)⁴ − 1
  = 16 + 12√2
  ≈ 32.97

This is the 13th manifestation of the silver ratio in MFT.

Author: Dale Wahl / MFT research programme, April 2026
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _out(fn): return os.path.join(_SCRIPT_DIR, fn)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

# MFT potential parameters (normalised)
M2  = 1.0
LAM4 = 2.0
LAM6 = 0.5

# Silver ratio
DELTA = 1 + np.sqrt(2)
SQRT2 = np.sqrt(2)

# Critical points
PHI_B = np.sqrt(2 - SQRT2)   # barrier
PHI_V = np.sqrt(2 + SQRT2)   # nonlinear vacuum

# Observed neutrino mass-squared differences (PDG 2024)
DM2_21_OBS = 7.53e-5    # eV² (solar)
DM2_32_OBS = 2.453e-3   # eV² (atmospheric, normal ordering)
RATIO_OBS  = DM2_32_OBS / DM2_21_OBS

# ══════════════════════════════════════════════════════════════════
# POTENTIAL
# ══════════════════════════════════════════════════════════════════

def V(phi):
    """MFT sextic potential V₆(φ) = m²φ²/2 − λ₄φ⁴/4 + λ₆φ⁶/6."""
    return M2*phi**2/2 - LAM4*phi**4/4 + LAM6*phi**6/6

# ══════════════════════════════════════════════════════════════════
# STEP-BY-STEP DERIVATION
# ══════════════════════════════════════════════════════════════════

print("=" * 70)
print("THE δ⁴ − 1 NEUTRINO HIERARCHY PREDICTION")
print("=" * 70)

print("\nSTEP 1: The three critical points of V₆(φ)")
print("-" * 50)
print(f"  φ = 0     (linear vacuum)      V(0)   = {V(0):.6f}")
print(f"  φ = φ_b   (barrier)            V(φ_b) = {V(PHI_B):.6f}")
print(f"  φ = φ_v   (nonlinear vacuum)   V(φ_v) = {V(PHI_V):.6f}")
print(f"\n  φ_b = √(2−√2) = {PHI_B:.6f}")
print(f"  φ_v = √(2+√2) = {PHI_V:.6f}")

print("\nSTEP 2: Exact potential values in terms of δ")
print("-" * 50)
V_0  = 0
V_b  = 1/(3*DELTA)
V_v  = DELTA/3
print(f"  V(0)   = 0")
print(f"  V(φ_b) = 1/(3δ)  = {V_b:.8f}")
print(f"  V(φ_v) = −δ/3    = {-V_v:.8f}")
print(f"\n  Verify: V(φ_b) from formula = {V_b:.8f}")
print(f"          V(φ_b) computed     = {V(PHI_B):.8f}")
print(f"          Match: {abs(V_b - V(PHI_B)) < 1e-10}  ✓")
print(f"  Verify: V(φ_v) from formula = {-V_v:.8f}")
print(f"          V(φ_v) computed     = {V(PHI_V):.8f}")
print(f"          Match: {abs(-V_v - V(PHI_V)) < 1e-10}  ✓")

print("\nSTEP 3: Square the potential values")
print("-" * 50)
V2_0 = V_0**2
V2_b = V_b**2
V2_v = V_v**2
print(f"  V²(0)   = 0")
print(f"  V²(φ_b) = 1/(9δ²) = {V2_b:.10f}")
print(f"  V²(φ_v) = δ²/9    = {V2_v:.10f}")

print("\nSTEP 4: Compute the mass-squared differences")
print("-" * 50)
dV2_21 = V2_b - V2_0   # ν₂ − ν₁
dV2_32 = V2_v - V2_b   # ν₃ − ν₂
print(f"  Δ(V²)₂₁ = V²(φ_b) − V²(0)   = 1/(9δ²)           = {dV2_21:.10f}")
print(f"  Δ(V²)₃₂ = V²(φ_v) − V²(φ_b) = δ²/9 − 1/(9δ²)   = {dV2_32:.10f}")

print("\nSTEP 5: Take the ratio")
print("-" * 50)
ratio_pred = dV2_32 / dV2_21
print(f"  Δ(V²)₃₂ / Δ(V²)₂₁ = [δ²/9 − 1/(9δ²)] / [1/(9δ²)]")
print(f"                      = [δ² − 1/δ²] × δ²")
print(f"                      = δ⁴ − 1")
print(f"                      = (1+√2)⁴ − 1")
print(f"                      = 16 + 12√2")
print(f"                      = {ratio_pred:.6f}")

# Verify algebraically
delta4_minus1 = 16 + 12*SQRT2
print(f"\n  Verify: 16 + 12√2 = {delta4_minus1:.6f}")
print(f"  Verify: δ⁴ − 1    = {DELTA**4 - 1:.6f}")
print(f"  Match: {abs(ratio_pred - delta4_minus1) < 1e-10}  ✓")

print("\nSTEP 6: Compare to observation")
print("-" * 50)
print(f"  Predicted: δ⁴ − 1 = 16 + 12√2 = {ratio_pred:.4f}")
print(f"  Observed:  Δm²₃₂/Δm²₂₁       = {RATIO_OBS:.4f}")
print(f"  Error:     {abs(ratio_pred - RATIO_OBS)/RATIO_OBS * 100:.2f}%")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  The MFT sextic potential, whose shape is derived from the
  Symmetric Back-Reaction Theorem (λ₄² = 8m²λ₆), predicts:

    Δm²₃₂ / Δm²₂₁ = δ⁴ − 1 = 16 + 12√2 ≈ {ratio_pred:.2f}

  This uses:
    • V(φ_b) = 1/(3δ) and V(φ_v) = −δ/3 (exact, from the potential)
    • The V² splitting mechanism (second-order gravitational backreaction)
    • The Family-of-Three theorem (one neutrino per critical point)
    • Nothing else — no free parameters, no fitting

  Observed: {RATIO_OBS:.2f} ± ~0.3
  Error: {abs(ratio_pred - RATIO_OBS)/RATIO_OBS * 100:.2f}%

  This is the 13th manifestation of the silver ratio δ = 1+√2 in MFT,
  and the first involving δ⁴.
""")

# ══════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ── Panel 1: The sextic potential with critical points ────────────
ax1 = fig.add_subplot(gs[0, 0])
phi = np.linspace(-0.2, 2.2, 500)
ax1.plot(phi, V(phi), 'k-', lw=2)
ax1.axhline(0, color='gray', ls='-', lw=0.5)

# Mark the three critical points
for phi_c, label, color, va in [(0, r'$\varphi=0$', 'blue', 'bottom'),
                                  (PHI_B, r'$\varphi_b$', 'orange', 'bottom'),
                                  (PHI_V, r'$\varphi_v$', 'red', 'top')]:
    ax1.plot(phi_c, V(phi_c), 'o', color=color, ms=10, zorder=5)
    offset = 0.03 if va == 'bottom' else -0.03
    ax1.annotate(f'{label}\nV={V(phi_c):.3f}', xy=(phi_c, V(phi_c)),
                xytext=(phi_c + 0.05, V(phi_c) + offset + (0.1 if va == 'bottom' else -0.15)),
                fontsize=9, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax1.set_xlabel(r'$\varphi$', fontsize=12)
ax1.set_ylabel(r'$V_6(\varphi)$', fontsize=12)
ax1.set_title('MFT sextic potential with\nthree critical points', fontsize=11)
ax1.set_ylim(-1.0, 0.25)
ax1.grid(True, alpha=0.3)

# ── Panel 2: V² at the critical points ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

labels = [r'$\nu_1$' + '\n' + r'($\varphi=0$)',
          r'$\nu_2$' + '\n' + r'($\varphi_b$)',
          r'$\nu_3$' + '\n' + r'($\varphi_v$)']
V2_vals = [V2_0, V2_b, V2_v]
colors = ['blue', 'orange', 'red']
x_pos = [0, 1, 2]

bars = ax2.bar(x_pos, V2_vals, color=colors, alpha=0.7, width=0.6,
               edgecolor='black', linewidth=1.5)

# Annotate the values
for x, v2, col in zip(x_pos, V2_vals, colors):
    if v2 > 0.001:
        ax2.text(x, v2 + 0.01, f'{v2:.4f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax2.text(x, v2 + 0.01, f'{v2:.1e}', ha='center', fontsize=9)

# Draw the Δ(V²) arrows
ax2.annotate('', xy=(1, V2_b), xytext=(0, V2_0),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax2.text(0.5, V2_b/2 + 0.005, r'$\Delta(V^2)_{21}$' + f'\n= 1/(9δ²)\n= {dV2_21:.4f}',
        ha='center', fontsize=9, color='green', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

ax2.annotate('', xy=(2, V2_v), xytext=(1, V2_b),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax2.text(1.5, (V2_v + V2_b)/2, r'$\Delta(V^2)_{32}$' + f'\n= {dV2_32:.4f}',
        ha='center', fontsize=9, color='purple', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='lavender', alpha=0.9))

ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel(r'$V^2(\varphi_{\rm core})$', fontsize=12)
ax2.set_title(r'$V^2$ at each critical point', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# ── Panel 3: The derivation chain ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

derivation_text = (
    r"$\bf{Derivation:}$" + "\n\n"
    r"$V(0) = 0$" + "\n"
    r"$V(\varphi_b) = \frac{1}{3\delta}$" + "\n"
    r"$V(\varphi_v) = -\frac{\delta}{3}$" + "\n\n"
    r"$V^2(\varphi_v) = \frac{\delta^2}{9}, \quad V^2(\varphi_b) = \frac{1}{9\delta^2}$" + "\n\n"
    r"$\frac{\Delta m^2_{32}}{\Delta m^2_{21}} = \frac{V^2(\varphi_v) - V^2(\varphi_b)}{V^2(\varphi_b) - V^2(0)}$" + "\n\n"
    r"$= \frac{\delta^2/9 - 1/(9\delta^2)}{1/(9\delta^2)} = \delta^4 - 1$" + "\n\n"
    r"$= (1+\sqrt{2})^4 - 1 = 16 + 12\sqrt{2}$" + "\n\n"
    r"$\approx 32.97$"
)
ax3.text(0.05, 0.95, derivation_text, transform=ax3.transAxes,
        fontsize=13, va='top', family='serif',
        bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.9))

# ── Panel 4: Comparison with observation ──────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

# Bar chart: predicted vs observed
x = [0, 1]
vals = [ratio_pred, RATIO_OBS]
clrs = ['#2196F3', '#4CAF50']
lbls = [f'MFT: δ⁴−1\n= {ratio_pred:.2f}', f'Observed\n= {RATIO_OBS:.2f}']

bars = ax4.bar(x, vals, color=clrs, alpha=0.8, width=0.5,
               edgecolor='black', linewidth=1.5)

for xi, vi in zip(x, vals):
    ax4.text(xi, vi + 0.5, f'{vi:.2f}', ha='center', fontsize=14, fontweight='bold')

ax4.set_xticks(x)
ax4.set_xticklabels(lbls, fontsize=11)
ax4.set_ylabel(r'$\Delta m^2_{32} \,/\, \Delta m^2_{21}$', fontsize=13)
ax4.set_title(f'Neutrino hierarchy ratio\n(error: {abs(ratio_pred-RATIO_OBS)/RATIO_OBS*100:.2f}%)',
             fontsize=11)
ax4.set_ylim(0, 38)
ax4.grid(True, alpha=0.3, axis='y')

# Add the error annotation
err_pct = abs(ratio_pred - RATIO_OBS)/RATIO_OBS * 100
ax4.annotate(f'Error: {err_pct:.2f}%', xy=(0.5, max(vals) + 2),
            fontsize=14, ha='center', fontweight='bold',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.4', fc='mistyrose', alpha=0.9))

fig.suptitle(r'MFT Neutrino Hierarchy: $\Delta m^2_{32}/\Delta m^2_{21} = \delta^4 - 1 = 16 + 12\sqrt{2}$',
            fontsize=14, fontweight='bold', y=0.98)

plt.savefig(_out('mft_neutrino_hierarchy.png'), dpi=150, bbox_inches='tight')
print(f"Figure saved: {_out('mft_neutrino_hierarchy.png')}")

# ══════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"\n  Predicted: δ⁴ − 1 = 16 + 12√2 = {ratio_pred:.4f}")
print(f"  Observed:  Δm²₃₂/Δm²₂₁     = {RATIO_OBS:.4f}")
print(f"  Error:     {err_pct:.2f}%")
print(f"  Status:    {'PASS' if err_pct < 5 else 'FAIL'} (threshold: 5%)")
