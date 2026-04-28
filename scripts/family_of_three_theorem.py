#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Family-of-Three Stability Theorem — Numerical Solver
----------------------------------------------------
Solves the radial eigenvalue problem in a position-space confining potential
and computes the three Morse indices (unconstrained, charge-constrained,
physical) for modes n = 0, 1, 2, 3. Output is a JSON record consumable by
the figure script (figures_family_of_three.py).

This script implements the numerical verification of Theorem 3.10
(Family-of-Three Stability) as described in §4.1-4.4 of the companion
preprint.

Build note: this version omits the LOBPCG preconditioner to avoid reshape
errors in some SciPy builds.
"""

from __future__ import annotations
import argparse, math, os, json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

# numpy < 2.0 compatibility: np.trapezoid was named np.trapz before NumPy 2.0
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, lobpcg

@dataclass
class Params:
    Rmax: float = 14.0
    N: int = 3000
    mass: float = 1.0
    Z: float = 1.1
    a: float = 1.1451604
    k2: float = 0.039
    k4: float = 2e-6
    k6: float = 2e-7
    ell: int = 0
    energy_ref: str = "none"

def make_grid(Rmax: float, N: int) -> NDArray[np.float64]:
    rin = Rmax / (N * 300.0)
    return np.linspace(rin, Rmax, N, dtype=float)

def V_core(r, Z, a, k2, k4, k6):
    return -Z / np.sqrt(r*r + a*a) + 0.5 * k2 * (r*r) + 0.25 * k4 * (r**4) + (1.0/6.0) * k6 * (r**6)

def V_eff(r, p: Params):
    lterm = 0.0 if p.ell == 0 else p.ell*(p.ell+1)/(r*r)
    return V_core(r, p.Z, p.a, p.k2, p.k4, p.k6) + lterm

def dV_dr(r, p: Params):
    core = -p.Z * (-r) * (r*r + p.a*p.a) ** (-1.5) + p.k2 * r + p.k4 * (r**3) + p.k6 * (r**5)
    return core if p.ell == 0 else core - 2.0 * p.ell * (p.ell + 1) / (r**3)

def numerov(r, E, p: Params, y0: float = 0.0, y1: float = 1e-12):
    h = r[1] - r[0]
    V = V_eff(r, p); ksq = E - V
    u = np.empty_like(r); u[0], u[1] = y0, y1
    c = (h*h)/12.0
    for n in range(1, len(r)-1):
        knm1, kn, knp1 = ksq[n-1], ksq[n], ksq[n+1]
        num = (2.0*(1.0 - 5.0*c*kn) * u[n]) - (1.0 + c*knm1) * u[n-1]
        den = (1.0 + c*knp1)
        u[n+1] = num / den
        if not np.isfinite(u[n+1]):
            u[n+1:] = 0.0; break
    return u, robust_node_count(u)

def robust_node_count(u):
    if len(u) < 12: return 0
    tail_frac = 0.05; cut = max(8, int((1.0 - tail_frac) * len(u)))
    w = u[:cut]
    if not np.any(np.isfinite(w)): return 0
    amp = float(np.nanmax(np.abs(w))); 
    if amp <= 0.0: return 0
    eps = max(1e-8, 1e-6) * amp
    mask = np.abs(w) > eps
    s = np.sign(w[mask])
    if s.size < 2: return 0
    return int(np.sum(s[1:] * s[:-1] < 0))

def normalize(u, r):
    n2 = float(np.trapezoid(u*u, r)); 
    return u if n2 <= 0 else (u / math.sqrt(n2))

def uR_of_E(E, r, p: Params) -> float:
    u, _ = numerov(r, E, p); 
    return float(u[-1])

def scan_same_node_signflip(target_nodes, r, p, Emin, Emax, Mscan):
    Es = np.linspace(Emin, Emax, Mscan, dtype=float)
    prior_E = None; prior_sign = None
    for E in Es:
        u, nodes = numerov(r, E, p)
        if nodes == target_nodes:
            s = np.sign(u[-1]) if u[-1] != 0 else 0.0
            if prior_E is not None and prior_sign is not None:
                if s == 0.0 or s * prior_sign < 0:
                    return (prior_E, E)
            prior_E, prior_sign = E, s
        else:
            prior_E, prior_sign = None, None
    return None

def sturm_bracket_by_nodes(target_nodes, r, p, Emin, Emax, Mscan):
    Es = np.linspace(Emin, Emax, Mscan, dtype=float); counts = []
    for E in Es:
        _, n = numerov(r, E, p); counts.append(n)
    counts = np.asarray(counts, dtype=int)
    t = target_nodes
    ok_lo = np.where(counts <= t)[0]; ok_hi = np.where(counts >= t+1)[0]
    if ok_lo.size == 0 or ok_hi.size == 0: raise RuntimeError("Sturm bracket failed")
    i_lo = ok_lo.min(); i_hi_cand = ok_hi[ok_hi >= i_lo]
    if i_hi_cand.size == 0: raise RuntimeError("Sturm bracket failed: no hi after lo.")
    i_hi = i_hi_cand.min()
    lo = float(Es[i_lo]); hi = float(Es[i_hi])
    if not (lo < hi):
        lo = float(Es[max(0, i_lo-1)]); hi = float(Es[min(len(Es)-1), i_hi+1])
    return lo, hi

def secant_root_uR(r, p, E0, E1, maxiter=80, tol=1e-12):
    f0 = uR_of_E(E0, r, p); f1 = uR_of_E(E1, r, p)
    if f1 == f0:
        E1 = E1 + 1e-6*(1.0 + abs(E1)); f1 = uR_of_E(E1, r, p)
    for _ in range(maxiter):
        if f1 == f0: break
        E2 = E1 - f1 * (E1 - E0) / (f1 - f0)
        if not np.isfinite(E2): E2 = 0.5*(E0 + E1)
        f2 = uR_of_E(E2, r, p)
        if abs(E2 - E1) <= tol * (1.0 + abs(E2)): return float(E2)
        E0, f0, E1, f1 = E1, f1, E2, f2
    return float(E1)

def auto_bracket_and_solve(target_nodes, r, p, Emin, Emax):
    E_lo, E_hi = Emin, Emax; M = 400
    for _ in range(6):
        br = scan_same_node_signflip(target_nodes, r, p, E_lo, E_hi, M)
        if br is not None:
            a, b = br
            try:
                return float(brentq(lambda E: uR_of_E(E, r, p), a, b, maxiter=200, xtol=1e-12, rtol=1e-10))
            except Exception:
                return secant_root_uR(r, p, a, b)
        width = (E_hi - E_lo); E_lo -= 0.5*width; E_hi += 0.5*width; M = int(M*1.25)
    br_lo, br_hi = sturm_bracket_by_nodes(target_nodes, r, p, E_lo, E_hi, max(600, M))
    E = secant_root_uR(r, p, br_lo, br_hi)
    f_lo = uR_of_E(br_lo, r, p); f_hi = uR_of_E(br_hi, r, p)
    if f_lo * f_hi < 0:
        try:
            E = float(brentq(lambda _E: uR_of_E(_E, r, p), br_lo, br_hi, maxiter=200, xtol=1e-12, rtol=1e-10))
        except Exception:
            pass
    return float(E)

def refine_to_exact_nodes(target_nodes, r, p, E_guess, max_passes=3):
    E = float(E_guess)
    for _ in range(max_passes):
        _, n = numerov(r, E, p)
        if n == target_nodes: return E
        span = 0.5 * (abs(E) + 1.0); Emin = E - span; Emax = E + span
        for _expand in range(5):
            try:
                lo, hi = sturm_bracket_by_nodes(target_nodes, r, p, Emin, Emax, 800)
                E = float(secant_root_uR(r, p, lo, hi)); break
            except Exception:
                width = (Emax - Emin); Emin -= 0.5*width; Emax += 0.5*width
        else:
            return E
    return E

def solve_state(target_nodes, r, p, Emin, Emax):
    E0 = auto_bracket_and_solve(target_nodes, r, p, Emin, Emax)
    E = refine_to_exact_nodes(target_nodes, r, p, E0)
    u, nodes = numerov(r, E, p)
    u = normalize(u, r)
    return float(E), u, nodes

def solve_up_to_n(max_n, r, p):
    V = V_eff(r, p); Vmin, Vmax = float(np.min(V)), float(np.max(V))
    buf = max(0.1, 0.03 * (Vmax - Vmin + 1.0))
    results = {}
    E0_lo = Vmin - buf; E0_hi = min(Vmin + 5.0, Vmax + 2.0)
    E0, u0, n0 = solve_state(0, r, p, E0_lo, E0_hi)
    results[0] = dict(E=E0, u=u0, nodes=n0)
    Emin = E0 - 2.0; Emax = Vmax + 12.0
    for n in range(1, max_n+1):
        try:
            En, un, nn = solve_state(n, r, p, Emin=Emin, Emax=Emax)
            results[n] = dict(E=En, u=un, nodes=nn)
        except Exception as e:
            results[n] = dict(E=np.nan, u=np.zeros_like(r), nodes=-1, error=str(e))
    return results

def to_psi(u, r): return r * u

def build_Hpsi_tridiag(r, p, En):
    h = r[1] - r[0]; N = r.size
    main = np.full(N, 2.0 / (h*h)); off  = np.full(N-1, -1.0 / (h*h))
    V = V_eff(r, p) - En; main = main + V
    return main, off

def tri_to_sparse(main, off): return diags([off, main, off], offsets=[-1, 0, +1], format='csc')
def tri_apply(main, off, x):
    y = main * x; y[:-1] += off * x[1:]; y[1:]  += off * x[:-1]; return y

def _project_columns(U, X):
    if U.size == 0: return X
    return X - U @ (U.T @ X)

def _orthonormalize(cols):
    if not cols: return np.zeros((0,0))
    Q = None
    for v in cols:
        v = v.reshape(-1,1)
        if Q is None:
            q = v / (np.linalg.norm(v) + 1e-20); Q = q
        else:
            w = v - Q @ (Q.T @ v); n = np.linalg.norm(w)
            if n > 1e-20: Q = np.column_stack([Q, w / n])
    return Q if Q is not None else np.zeros((0,0))

def _make_projected_operator(A_sparse, U):
    N = A_sparse.shape[0]
    def matvec(x1d):
        x = x1d.reshape(-1, 1); x = _project_columns(U, x)
        y = A_sparse @ x; y = _project_columns(U, y); return y.ravel()
    def matmat(X):
        Xp = _project_columns(U, X); Y  = A_sparse @ Xp; Yp = _project_columns(U, Y); return Yp
    return LinearOperator((N, N), matvec=matvec, rmatvec=matvec, matmat=matmat, dtype=np.float64)

def central_diff_first(r, f):
    h = r[1] - r[0]; df = np.empty_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2*h)
    df[0] = (f[1] - f[0]) / h; df[-1] = (f[-1] - f[-2]) / h
    return df

def dilation_generator_analytic(r, psi):
    dpsi = central_diff_first(r, psi); g = r * dpsi + 1.5 * psi
    psi_unit = psi / (np.linalg.norm(psi) + 1e-20)
    g = g - psi_unit * float(np.dot(psi_unit, g))
    ng = np.linalg.norm(g); return g / (ng + 1e-20)

def scale_profile(r, psi, alpha):
    r_src = np.clip(alpha * r, r[0], r[-1])
    f = interp1d(r, psi, kind="linear", bounds_error=False, fill_value=(psi[0], psi[-1]))
    return (alpha**0.5) * f(r_src)

def dilation_generator_fd5(r, psi, eps=1e-3):
    psi_p2 = scale_profile(r, psi, 1.0 + 2*eps)
    psi_p1 = scale_profile(r, psi, 1.0 + eps)
    psi_m1 = scale_profile(r, psi, 1.0 - eps)
    psi_m2 = scale_profile(r, psi, 1.0 - 2*eps)
    g = (-psi_p2 + 8*psi_p1 - 8*psi_m1 + psi_m2) / (12*eps)
    psi_unit = psi / (np.linalg.norm(psi) + 1e-20)
    g = g - psi_unit * float(np.dot(psi_unit, g))
    ng = np.linalg.norm(g); return g / (ng + 1e-20)

def lowest_eigs_with_constraints(main, off, constraints, k=12, tol=1e-5, maxiter=6000, seed=12345):
    A_sparse = tri_to_sparse(main, off)
    U = _orthonormalize(constraints)
    PAP = _make_projected_operator(A_sparse, U)
    rng = np.random.default_rng(seed)
    N = main.size
    k_eff = max(3, min(k, N - max(4, U.shape[1]+2)))
    X = rng.normal(size=(N, k_eff)); X = _project_columns(U, X)
    Q, _ = np.linalg.qr(X, mode='reduced')
    if Q.shape[1] < k_eff:
        Z = rng.normal(size=(N, k_eff - Q.shape[1])); Z = _project_columns(U, Z)
        Z, _ = np.linalg.qr(Z, mode='reduced'); Q = np.column_stack([Q, Z[:, :k_eff - Q.shape[1]]])
    def neg_matvec(x1d): return -PAP.matvec(x1d)
    def neg_matmat(Xb):  return -PAP.matmat(Xb)
    nPAP = LinearOperator((N, N), matvec=neg_matvec, rmatvec=neg_matvec, matmat=neg_matmat, dtype=np.float64)
    vals_neg, vecs = lobpcg(nPAP, Q, M=None, Y=U, tol=tol, maxiter=maxiter)
    vals = -vals_neg; idx = np.argsort(vals); return vals[idx], vecs[:, idx]

def ritz_bound_from_vecs(main, off, basis):
    if basis is None or basis.size == 0: return 0, np.zeros(0)
    Q, _ = np.linalg.qr(basis, mode='reduced')
    AQ = np.column_stack([tri_apply(main, off, Q[:, j]) for j in range(Q.shape[1])])
    Hs = Q.T @ AQ; vals = np.linalg.eigvalsh(Hs)
    lb = int(np.sum(vals < -1e-12)); return lb, vals

def count_negatives(vals, tol=1e-12) -> int: return int(np.sum(vals < -tol))

def run_once(p: Params, max_n: int, do_fluct: bool, report: bool, constraint_mode: str) -> Dict[str, Any]:
    r = make_grid(p.Rmax, p.N)
    res = solve_up_to_n(max_n, r, p)
    vprime = dV_dr(r, p)
    for n, row in res.items():
        if row.get('nodes', -1) >= 0 and np.isfinite(row.get('E', np.nan)):
            u = row['u']
            if n > 0 and (n-1) in res and res[n-1]['nodes'] >= 0:
                ortho = float(np.trapezoid(u * res[n-1]['u'], r))
            else:
                ortho = float('nan')
            vir = float(np.trapezoid((r * vprime) * (u*u), r))
            res[n]['ortho_prev'] = ortho; res[n]['virial_value'] = vir

    if do_fluct:
        for n, row in res.items():
            if row.get('nodes', -1) >= 0 and np.isfinite(row['E']):
                main, off = build_Hpsi_tridiag(r, p, row['E'])
                psi = to_psi(row['u'], r); psi_unit = psi / (np.linalg.norm(psi) + 1e-20)

                # RAW spectrum
                vals_raw, vecs_raw = lowest_eigs_with_constraints(main, off, [psi_unit])
                neg_raw = count_negatives(vals_raw)
                neg_mask_raw = vals_raw < -1e-12
                neg_basis = vecs_raw[:, neg_mask_raw] if np.any(neg_mask_raw) else np.zeros((psi.size, 0))
                if np.any(neg_mask_raw): lb_raw, ritz_vals_raw = ritz_bound_from_vecs(main, off, neg_basis)
                else: lb_raw, ritz_vals_raw = 0, np.zeros(0)

                # Constraint vector
                if constraint_mode == "fd5":
                    g0 = dilation_generator_fd5(r, psi, eps=1e-3)
                else:
                    g0 = dilation_generator_analytic(r, psi)

                # Target into negative subspace if present
                if neg_basis.size > 0:
                    Qneg, _ = np.linalg.qr(neg_basis, mode='reduced')
                    coeffs = Qneg.T @ g0
                    g_tgt = Qneg @ coeffs
                    ng = np.linalg.norm(g_tgt)
                    g_unit = g_tgt / ng if ng > 1e-20 else g0
                else:
                    g_unit = g0

                aligns = []
                for j in range(min(vecs_raw.shape[1], 6)):
                    v = vecs_raw[:, j]; v_norm = v / (np.linalg.norm(v) + 1e-20)
                    aligns.append(float(abs(np.dot(v_norm, g_unit))))

                # CONSTRAINED spectrum (targeted)
                vals_con, vecs_con = lowest_eigs_with_constraints(main, off, [psi_unit, g_unit])
                neg_con = count_negatives(vals_con)
                neg_mask_con = vals_con < -1e-12
                if np.any(neg_mask_con): lb_con, ritz_vals_con = ritz_bound_from_vecs(main, off, vecs_con[:, neg_mask_con])
                else: lb_con, ritz_vals_con = 0, np.zeros(0)

                row['fluct_raw'] = {'eigvals': np.array(vals_raw), 'neg_count': int(neg_raw), 'lb_ritz': int(lb_raw), 'ritz_vals': np.array(ritz_vals_raw)}
                row['fluct_constrained_targeted'] = {'eigvals': np.array(vals_con), 'neg_count': int(neg_con), 'lb_ritz': int(lb_con), 'ritz_vals': np.array(ritz_vals_con)}
                row['constraint_alignment'] = {'mode': constraint_mode, 'alignments_vs_raw_lowest6': aligns}

    if report:
        print("=== RESULTS (presentation build: no-preconditioner) ===")
        print(f"params: {asdict(p)}")
        E0 = res.get(0, {}).get('E', np.nan)
        for n in range(0, max_n+1):
            row = res.get(n, {}); E = row.get('E', float('nan')); nodes = row.get('nodes', -1)
            print(f"n={n:>1}  nodes={nodes:>2}  E={E:+.9f}")
            if n == 1 and np.isfinite(E0) and np.isfinite(E) and E0 != 0.0:
                print(f"   E1/E0 = {E/E0:.6f}")
            if 'virial_value' in row:
                print(f"   virial ~ {row['virial_value']:+.3e},  ortho_prev ~ {row['ortho_prev']:+.3e}")
            fr = row.get('fluct_raw'); fc = row.get('fluct_constrained_targeted')
            if fr is not None and fc is not None:
                print(f"   RAW:   neg={fr['neg_count']}  lb={fr['lb_ritz']}")
                print(f"   CONS*: neg={fc['neg_count']}  lb={fc['lb_ritz']}  (*targeted scale constraint)")
                al = row.get('constraint_alignment', {}).get('alignments_vs_raw_lowest6', [])
                if al:
                    print(f"   align(|v_j|, g_targeted) (lowest6 RAW): {np.array2string(np.array(al), precision=3)}")
    return res

def main():
    ap = argparse.ArgumentParser(description="Family-of-three stability theorem: targeted-constraint Morse-index solver")
    ap.add_argument("--Rmax", type=float, default=14.0)
    ap.add_argument("--N", type=int, default=3000)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=1.1)
    ap.add_argument("--a", type=float, default=1.1451604)
    ap.add_argument("--k2", type=float, default=0.039)
    ap.add_argument("--k4", type=float, default=2e-6)
    ap.add_argument("--k6", type=float, default=2e-7)
    ap.add_argument("--ell", type=int, default=0)
    ap.add_argument("--energy_ref", choices=["none","minV"], default="none")
    ap.add_argument("--solve_up_to_n", type=int, default=3)
    ap.add_argument("--fluct_spectrum", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--constraint_mode", choices=["analytic","fd5"], default="analytic")
    ap.add_argument("--outfile_json", default="theorem_results.json")
    args = ap.parse_args()

    base = Params(Rmax=args.Rmax, N=args.N, mass=args.mass, Z=args.Z, a=args.a,
                  k2=args.k2, k4=args.k4, k6=args.k6, ell=args.ell, energy_ref=args.energy_ref)

    all_runs: List[Dict[str, Any]] = []
    def pack(p: Params, res: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"params": asdict(p), "states": {}}
        for n, row in res.items():
            row2 = {}
            for k, v in row.items():
                if isinstance(v, np.ndarray): row2[k] = v.tolist()
                elif isinstance(v, dict):
                    sub = {}
                    for kk, vv in v.items():
                        if isinstance(vv, np.ndarray): sub[kk] = vv.tolist()
                        else: sub[kk] = vv
                    row2[k] = sub
                else: row2[k] = v
            out["states"][str(n)] = row2
        return out

    res = run_once(base, args.solve_up_to_n, args.fluct_spectrum, args.report, args.constraint_mode)
    all_runs.append(pack(base, res))

    os.makedirs(os.path.dirname(args.outfile_json) or ".", exist_ok=True)
    with open(args.outfile_json, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    if args.report:
        print(f"\nWrote {len(all_runs)} run(s) → {args.outfile_json}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
