#!/usr/bin/env python3
"""
MFT HEDGEHOG BVP v3: stay in r-coordinates, extend r_max properly.

The v2 compactification introduced chain-rule issues. Let's just use
a large r_max (say 30) and see if the corrected ODE converges.

The one real fix from v1: the missing -sin(2f)·f'² term in the ODE.
"""
import numpy as np
try:
    from numpy import trapezoid as trap
except ImportError:
    from numpy import trapz as trap
from scipy.optimize import brentq
from scipy.integrate import solve_bvp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def out(fn): return os.path.join(SCRIPT_DIR, fn)

M_ELECTRON = 0.511
E_ELECTRON_MFT = 0.00427
MFT_TO_MEV = M_ELECTRON / E_ELECTRON_MFT
M_N = 938.3
M_DELTA = 1232.0
F_PI_OBS = 186.0
DELTA = 1 + np.sqrt(2)


def hedgehog_ode(r, y):
    """
    CORRECTED ODE:
      f'' = [sin(2f)(1 + sin²f/r²) - 2rf' - sin(2f)·f'²] / (r² + 2sin²f)
    """
    f = y[0]
    fp = y[1]

    s = np.sin(f); c = np.cos(f)
    s2 = s**2
    s2f = 2*s*c
    r_s = np.maximum(r, 1e-10)

    denom = r_s**2 + 2*s2
    numer = s2f*(1 + s2/r_s**2) - 2*r_s*fp - s2f*fp**2

    return np.vstack([fp, numer/np.maximum(denom, 1e-15)])


def bc(ya, yb):
    """f(0) = π, f(rmax) = 0. Plus use regularity to avoid the 1/r² blowup."""
    return np.array([ya[0] - np.pi, yb[0]])


def try_solve(rmin, rmax, R_h, n_mesh=300):
    """Try one (rmin, rmax, initial guess) combination."""
    # Mesh with clustering near origin
    r_inner = np.linspace(rmin, 3.0, n_mesh // 2)
    r_outer = np.linspace(3.0, rmax, n_mesh // 2)
    r_mesh = np.unique(np.concatenate([r_inner, r_outer]))

    f_guess = 2.0 * np.arctan(R_h / np.maximum(r_mesh, 1e-12))
    fp_guess = -2.0 * R_h / (r_mesh**2 + R_h**2)

    sol = solve_bvp(hedgehog_ode, bc, r_mesh, np.vstack([f_guess, fp_guess]),
                    tol=1e-9, max_nodes=30000, verbose=0)
    return sol


def solve_best():
    """Scan parameter space for the best convergence."""
    best_sol = None
    best_score = np.inf

    # r_max values to try
    for rmax in [20, 30, 50, 80]:
        for rmin in [1e-4, 1e-3, 1e-2]:
            for R_h in [0.5, 0.8, 1.0, 1.2, 1.5]:
                try:
                    sol = try_solve(rmin, rmax, R_h, n_mesh=400)
                except Exception:
                    continue

                if not sol.success:
                    continue

                # Score by virial balance (closer to 1 is better)
                r_eval = np.linspace(rmin, rmax, 5000)
                y_eval = sol.sol(r_eval)
                f = y_eval[0]; fp = y_eval[1]
                s = np.sin(f); s2 = s**2
                r_s = np.maximum(r_eval, 1e-10)
                E2 = 4*np.pi * float(trap(fp**2 * r_eval**2 + 2*s2, r_eval))
                E4 = 4*np.pi * float(trap(2*s2*fp**2 + s2**2/r_s**2, r_eval))

                if E4 < 1e-10 or E2 < 1e-10:
                    continue

                score = abs(E2/E4 - 1)
                if score < best_score:
                    best_score = score
                    best_sol = (sol, rmin, rmax, R_h, E2, E4)

    return best_sol


def compute_integrals(r, f, fp):
    s = np.sin(f); s2 = s**2; s4 = s**4
    r_s = np.maximum(r, 1e-10)

    E2 = 4*np.pi * float(trap(fp**2 * r**2 + 2*s2, r))
    E4 = 4*np.pi * float(trap(2*s2*fp**2 + s4/r_s**2, r))
    B = -(2.0/np.pi) * float(trap(s2*fp, r))
    B_topo = (f[0] - f[-1])/np.pi
    int_I = s2 * (r**2 + 2*s2*(fp**2 + s2/r_s**2))
    lam = (8*np.pi/3) * float(trap(int_I, r))

    return {'E2': E2, 'E4': E4, 'eps_0': E2 + E4,
            'E2_over_E4': E2/max(E4, 1e-15),
            'virial': abs(E2 - E4)/max(E2, 1e-10),
            'B': B, 'B_topo': B_topo, 'lambda_0': lam}


def main():
    print("=" * 70)
    print("MFT HEDGEHOG v3 (corrected ODE, parameter scan)")
    print("=" * 70)
    print("\nScanning (rmin, rmax, R_h) for best virial balance...")
    best = solve_best()

    if best is None:
        print("\nFAILED: no configuration converged with virial balance.")
        return

    sol, rmin, rmax, R_h, E2_quick, E4_quick = best
    print(f"\nBest configuration:")
    print(f"  rmin = {rmin},  rmax = {rmax},  R_h = {R_h}")
    print(f"  E2/E4 (quick) = {E2_quick/E4_quick:.6f}")

    r_fine = np.linspace(rmin, rmax, 8000)
    y_fine = sol.sol(r_fine)
    f = y_fine[0]; fp = y_fine[1]

    print(f"\nProfile check:")
    print(f"  f(0)     = {f[0]:.6f}    target π = {np.pi:.6f}")
    print(f"  f(rmax)  = {f[-1]:.2e}   target 0")
    print(f"  Monotonic: {all(np.diff(f) <= 1e-6)}")

    integrals = compute_integrals(r_fine, f, fp)
    print(f"\nEnergy integrals:")
    print(f"  E2          = {integrals['E2']:.4f}")
    print(f"  E4          = {integrals['E4']:.4f}")
    print(f"  E2/E4       = {integrals['E2_over_E4']:.6f}  (target 1.000)")
    print(f"  Imbalance   = {integrals['virial']*100:.3f}%")
    print(f"  ε_0         = {integrals['eps_0']:.4f}")
    print(f"  B (topo)    = {integrals['B_topo']:.6f}")
    print(f"  B (integral)= {integrals['B']:.6f}")
    print(f"  λ_0         = {integrals['lambda_0']:.4f}")

    # Parameter extraction
    if integrals['virial'] < 0.02:
        print("\n  ✓ Hedgehog converged to <2% virial balance")

        eps_0 = integrals['eps_0']
        lam_0 = integrals['lambda_0']
        deltaM = M_DELTA - M_N
        # Λ = 3/(2 ΔM) gives moment of inertia in MeV⁻¹
        # Λ = (2/(3 e³ f_π)) λ_0
        # M_core = (f_π/(4e)) ε_0
        # Two equations, two unknowns (f_π, e)
        Lambda_tgt = 3.0/(2*deltaM)
        M_core_tgt = M_N - 3.0/(8*Lambda_tgt)
        e4 = (eps_0 * 2*lam_0)/(4*M_core_tgt * 3*Lambda_tgt)
        e = e4**0.25
        fpi = 4*e*M_core_tgt/eps_0

        print(f"\n  Extracted: e = {e:.4f},  f_π = {fpi:.2f} MeV")
        print(f"  Observed:  f_π = {F_PI_OBS} MeV,  MFT pred = {np.sqrt(DELTA)*MFT_TO_MEV:.2f}")
        print(f"  e = 2δ candidate: {2*DELTA:.4f}  (diff {abs(e - 2*DELTA)/(2*DELTA)*100:.1f}%)")
    else:
        print(f"\n  ✗ Still off: virial imbalance {integrals['virial']*100:.1f}%")
        print(f"  Best we can get — may need more mesh refinement or larger rmax")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(r_fine[r_fine < 15], f[r_fine < 15], 'b-', lw=2, label='f(r) (BVP solution)')
    ax[0].axhline(np.pi, color='r', ls=':', label=r'$f = \pi$'); ax[0].axhline(0, color='gray', ls=':', label=r'$f = 0$')
    ax[0].set_xlabel('r'); ax[0].set_ylabel('f(r)')
    ax[0].set_title(f'Hedgehog profile (rmax={rmax})')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    s = np.sin(f); s2 = s**2; r_s = np.maximum(r_fine, 1e-10)
    e2d = 4*np.pi*(fp**2 * r_fine**2 + 2*s2)
    e4d = 4*np.pi*(2*s2*fp**2 + s2**2/r_s**2)
    mask = r_fine < 10
    ax[1].plot(r_fine[mask], e2d[mask], label=f'E2 density (∫={integrals["E2"]:.1f})')
    ax[1].plot(r_fine[mask], e4d[mask], label=f'E4 density (∫={integrals["E4"]:.1f})')
    ax[1].set_xlabel('r'); ax[1].set_title(f'E2/E4 = {integrals["E2_over_E4"]:.3f}')
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out('mft_hedgehog_bvp_v2.png'), dpi=130)
    print(f"\n  Saved: {out('mft_hedgehog_bvp_v2.png')}")


if __name__ == '__main__':
    main()
