#!/usr/bin/env python3
"""
MONISTIC FIELD THEORY: FLAGSHIP OVERVIEW FIGURE
=================================================
Companion script for Paper 10 v2: "Monistic Field Theory"

All panels use exact analytical formulas or hardcoded verified values.
No Q-ball scanning or numerical solving — everything is deterministic.

Author: Dale Wahl / MFT research programme, April 2026
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def outpath(fn): return os.path.join(SCRIPT_DIR, fn)

# ═══════════════════════════════════════════════════════════════════
# EXACT MFT PARAMETERS (all derived, no fitting)
# ═══════════════════════════════════════════════════════════════════
M2, LAM4, LAM6 = 1.0, 2.0, 0.5
DELTA = 1 + np.sqrt(2)
PHI_B = np.sqrt(2 - np.sqrt(2))  # 0.7654
PHI_V = np.sqrt(2 + np.sqrt(2))  # 1.8478
PHI_CROSS = np.sqrt(2.0)          # 1.4142

def V(phi):   return 0.5*M2*phi**2 - 0.25*LAM4*phi**4 + (1/6.)*LAM6*phi**6
def Vpp(phi): return M2 - 3*LAM4*phi**2 + 5*LAM6*phi**4

V_at_b = V(PHI_B)     # 1/(3δ) ≈ 0.138
V_at_v = V(PHI_V)     # -δ/3 ≈ -0.805
Vpp_0  = Vpp(0)       # = 1.0
Vpp_b  = Vpp(PHI_B)   # = -4/δ ≈ -1.657
Vpp_v  = Vpp(PHI_V)   # = 4δ ≈ 9.657

# Verified particle positions (from Paper 4)
PARTICLES = {
    'e':     (0.022, 'green', 'o', 10),
    'μ':     (0.711, 'blue',  'o', 10),
    'τ':     (1.928, 'red',   'o', 10),
    'W':     (1.301, 'purple','s', 9),
    'Z':     (1.281, 'darkviolet','s', 9),
    'H':     (1.232, 'brown', 's', 9),
}

def main():
    phi = np.linspace(0, 2.8, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Monistic Field Theory: A Theory of Spatial Contraction\n"
                 r"One elastic medium, one derived potential $V_6(\varphi)$ with "
                 r"$\lambda_4^2 = 8m_2\lambda_6$, silver ratio $\delta = 1+\sqrt{2}$",
                 fontsize=13, fontweight='bold')

    # ═══════════════════════════════════════════════════════════════
    # PANEL 1: The silver ratio potential with all particle positions
    # ═══════════════════════════════════════════════════════════════
    ax = axes[0, 0]
    ax.plot(phi, V(phi), 'k-', lw=2.5)

    # Critical points
    ax.plot(0, V(0), 'go', ms=12, zorder=6)
    ax.plot(PHI_B, V_at_b, '^', color='orange', ms=12, zorder=6)
    ax.plot(PHI_V, V_at_v, 's', color='red', ms=12, zorder=6)

    # Vertical reference lines
    ax.axvline(PHI_B, color='orange', ls='--', lw=1, alpha=0.5)
    ax.axvline(PHI_V, color='red', ls='--', lw=1, alpha=0.5)
    ax.axvline(PHI_CROSS, color='green', ls=':', lw=1.5, alpha=0.7)

    # Particle positions
    for name, (pc, col, mk, ms) in PARTICLES.items():
        ax.plot(pc, V(pc), mk, color=col, ms=ms, zorder=5)
        offset = 0.15 if name not in ['W', 'Z', 'H'] else -0.18
        ax.annotate(name, (pc, V(pc)), textcoords="offset points",
                    xytext=(0, 12 if offset > 0 else -14),
                    ha='center', fontsize=8, fontweight='bold', color=col)

    # Annotations for silver ratio values
    ax.annotate(f'$\\varphi_b = {PHI_B:.3f}$\n$V = 1/(3\\delta) = {V_at_b:.3f}$',
                xy=(PHI_B, V_at_b), xytext=(0.15, 0.7),
                fontsize=7, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange', lw=0.8))
    ax.annotate(f'$\\varphi_v = {PHI_V:.3f}$\n$V = -\\delta/3 = {V_at_v:.3f}$',
                xy=(PHI_V, V_at_v), xytext=(2.2, -0.3),
                fontsize=7, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax.annotate(f'$\\sqrt{{2}} = \\delta-1$\n(crossing)',
                xy=(PHI_CROSS, V(PHI_CROSS)), xytext=(1.7, 1.5),
                fontsize=7, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

    # Ratio annotation
    ax.annotate('', xy=(PHI_V, -1.0), xytext=(PHI_B, -1.0),
                arrowprops=dict(arrowstyle='<->', color='navy', lw=1.5))
    ax.text((PHI_B+PHI_V)/2, -1.08, f'$\\varphi_v/\\varphi_b = \\delta = {DELTA:.3f}$',
            ha='center', fontsize=8, color='navy', fontweight='bold')

    ax.set_xlabel('$\\varphi$ (contraction field)', fontsize=10)
    ax.set_ylabel('$V_6(\\varphi)$', fontsize=10)
    ax.set_title('The silver ratio potential\nwith all particle positions', fontsize=11)
    ax.set_ylim(-1.3, 2.5); ax.set_xlim(-0.05, 2.75)
    ax.grid(True, alpha=0.2)

    # ═══════════════════════════════════════════════════════════════
    # PANEL 2: Stiffness V''(φ) showing the three regimes
    # ═══════════════════════════════════════════════════════════════
    ax = axes[0, 1]
    ax.plot(phi, Vpp(phi), 'b-', lw=2.5)
    ax.axhline(0, color='black', lw=0.5)

    # Mark the three critical stiffness values
    ax.plot(0, Vpp_0, 'go', ms=12, zorder=6)
    ax.plot(PHI_B, Vpp_b, '^', color='orange', ms=12, zorder=6)
    ax.plot(PHI_V, Vpp_v, 's', color='red', ms=12, zorder=6)

    # Shade the three regimes
    ax.axvspan(0, PHI_B, alpha=0.08, color='green', label='Linear vacuum')
    ax.axvspan(PHI_B, PHI_V, alpha=0.08, color='orange', label='Barrier region')
    ax.axvspan(PHI_V, 2.8, alpha=0.08, color='red', label='Nonlinear vacuum')

    # Stiffness labels
    ax.annotate(f"$V''(0) = {Vpp_0:.1f}$\n(baseline)", xy=(0, Vpp_0),
                xytext=(0.3, 3.0), fontsize=8, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=0.8))
    ax.annotate(f"$V''(\\varphi_b) = -4/\\delta$\n$= {Vpp_b:.2f}$\n(unstable)",
                xy=(PHI_B, Vpp_b), xytext=(0.2, -5),
                fontsize=8, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange', lw=0.8))
    ax.annotate(f"$V''(\\varphi_v) = 4\\delta$\n$= {Vpp_v:.2f}$\n(9.66× stiffer)",
                xy=(PHI_V, Vpp_v), xytext=(2.1, 6),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    # The 4δ ratio
    ax.annotate('', xy=(2.55, Vpp_v), xytext=(2.55, Vpp_0),
                arrowprops=dict(arrowstyle='<->', color='navy', lw=1.5))
    ax.text(2.65, (Vpp_v+Vpp_0)/2, f'$4\\delta$\n$≈ 9.66×$',
            fontsize=9, color='navy', fontweight='bold', va='center')

    ax.set_xlabel('$\\varphi$ (contraction field)', fontsize=10)
    ax.set_ylabel("$V''(\\varphi)$ (stiffness)", fontsize=10)
    ax.set_title('Medium stiffness across three regimes\n'
                 '(electron → barrier → tau/bosons)', fontsize=11)
    ax.set_ylim(-7, 14); ax.set_xlim(-0.05, 2.8)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2)

    # ═══════════════════════════════════════════════════════════════
    # PANEL 3: Four physical regimes (schematic)
    # ═══════════════════════════════════════════════════════════════
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Four physical regimes of MFT\n(same field equations, different limits)',
                 fontsize=11)

    regimes = [
        (1.5, 8.0, 'Dense, Weak-Field\n(Solar System)',
         'steelblue',
         '• φ screened, short-ranged\n'
         '• Reduces to GR + corrections\n'
         '• ω_BD > 40,000 (Cassini)\n'
         '• PPN parameters match GR'),
        (6.5, 8.0, 'Low-Density, Galactic',
         'forestgreen',
         '• φ unscreened across kpc\n'
         '• Nonlinear BVP for halos\n'
         '• Σχ²/dof = 1.17 (6 galaxies)\n'
         '• No dark matter needed'),
        (1.5, 3.5, 'Black-Hole Limit',
         'darkred',
         '• φ → φ_v (elastic ceiling)\n'
         '• V″(φ_v) = 4δ bounds density\n'
         '• Non-singular interiors\n'
         '• Finite curvature everywhere'),
        (6.5, 3.5, 'Cosmological (Voids)',
         'darkorange',
         '• Low-density void evolution\n'
         '• Effective Hubble law from φ\n'
         '• Acceleration without Λ\n'
         '• Dielectric → Hubble tension?'),
    ]

    for x, y, title, color, desc in regimes:
        box = mpatches.FancyBboxPatch((x-2.0, y-2.0), 4.0, 3.2,
                                       boxstyle="round,pad=0.15",
                                       facecolor=color, alpha=0.12,
                                       edgecolor=color, lw=2)
        ax.add_patch(box)
        ax.text(x, y+0.8, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)
        ax.text(x, y-0.5, desc, ha='center', va='center',
                fontsize=7.5, color='black', family='monospace')

    # Central label
    ax.text(4.0, 5.75, 'Same action\nSame φ\nSame V₆(φ)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='black', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    # ═══════════════════════════════════════════════════════════════
    # PANEL 4: The complete derivation chain
    # ═══════════════════════════════════════════════════════════════
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('The MFT derivation chain (16 papers)\n(each step is a theorem — zero free parameters in the potential)',
                 fontsize=11)

    steps = [
        (0, 'MFT Action',              'Scalar-tensor on 3D slice',    'Paper 0'),
        (1, 'K₆>0, K₂>0, K₄<0',       "Derrick's theorem",            'Paper 1'),
        (2, 'λ₄² = 8m₂λ₆',            'Back-Reaction Theorem',        'Paper 2'),
        (3, 'Exactly 3 families',       'Morse index n−1',             'Paper 3'),
        (4, 'All particle masses',      'Q-ball equation',              'Paper 4'),
        (5, 'Rotation curves',          'Nonlinear BVP',                'Paper 5'),
        (6, 'Spin-½ (14× gap)',         'Emergent Lorentz + FR',        'Paper 6'),
        (7, 'Confinement',              'Elastic topology',             'Paper 7'),
        (8, 'Microphysics synthesis',   'Skyrme + neutral solitons',    'Paper 8'),
        (9, 'Flagship (this paper)',    'Complete foundations',          'Paper 9'),
        (10,'Quantum completion',       'Hamiltonian + Fock space',     'Paper 10'),
        (11,'Propagators + vertices',   'Linearised (ω,k) space',      'Paper 11'),
        (12,'Cosmology without Λ',      'Void expansion + H₀',         'Paper 12'),
        (13,'Compact objects + PPN',    'Yukawa, saturation',           'Paper 13'),
        (14,'EM form factor + g=2',     'F_MFT = F_QED + O(10⁻⁴⁵)',    'Paper 14'),
        (15,'3D finiteness',            'Q-ball separation + [λ₆]=0',   'Paper 15'),
    ]

    n = len(steps)
    for i, (idx, result, method, paper) in enumerate(steps):
        y = 9.5 - i * 0.58
        # Step box — color by grouping:
        # Blue = foundations (P0–P2); Green = predictions (P3–P5);
        # Red = microphysics (P6–P8); Orange = quantum + astro (P9–P15)
        color = 'steelblue' if i < 3 else ('forestgreen' if i < 6 else ('darkred' if i < 9 else 'darkorange'))
        ax.add_patch(mpatches.FancyBboxPatch((0.2, y-0.21), 3.2, 0.42,
                     boxstyle="round,pad=0.06", facecolor=color, alpha=0.15,
                     edgecolor=color, lw=1.5))
        ax.text(1.8, y, result, ha='center', va='center',
                fontsize=7, fontweight='bold', color=color)
        # Arrow
        if i < n-1:
            ax.annotate('', xy=(1.8, y-0.26), xytext=(1.8, y-0.32),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        # Method
        ax.text(5.0, y, method, ha='left', va='center', fontsize=7, color='gray')
        # Paper
        ax.text(8.5, y, paper, ha='left', va='center', fontsize=7,
                color='black', style='italic')

    # Legend
    ax.text(0.2, 0.3, 'Blue = foundations    Green = predictions    Red = microphysics    Orange = quantum + astro',
            fontsize=7, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out = outpath('mft_flagship.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {out}")
    print("All panels use exact analytical formulas — zero numerical scanning.")

if __name__ == '__main__':
    main()
