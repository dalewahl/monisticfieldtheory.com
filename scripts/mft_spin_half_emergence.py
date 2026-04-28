#!/usr/bin/env python3
"""MFT SPIN-1/2 EMERGENCE: SPECTRAL ASYMMETRY OF Q-BALL FLUCTUATIONS"""
import numpy as np
try:
    from numpy import trapezoid as trap
except ImportError:
    from numpy import trapz as trap
from scipy.optimize import brentq
from scipy.linalg import eigvalsh
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def outpath(fn): return os.path.join(SCRIPT_DIR, fn)

M2, LAM4, LAM6 = 1.0, 2.0, 0.5
DELTA = 1.0 + np.sqrt(2)
PHI_B = np.sqrt(2 - np.sqrt(2))
PHI_V = np.sqrt(2 + np.sqrt(2))
PHI_CROSS = np.sqrt(2.0)
RMAX = 20.0; N = 200
r = np.linspace(RMAX/(N*100), RMAX, N)
h = r[1] - r[0]

def Vpp(phi):  return M2 - 3*LAM4*phi**2 + 5*LAM6*phi**4
def Vp_over_phi(phi): return M2 - LAM4*phi**2 + LAM6*phi**4
def DeltaV(phi): return Vpp(phi) - Vp_over_phi(phi)

def shoot(A, omega2, Z=1.0, a=1.0, ell=0):
    u = np.zeros(N); u[0] = 0.0
    u[1] = A * r[1]**(ell+1) if ell > 0 else A * r[1]
    for i in range(1, N-1):
        phi_i = u[i]/r[i] if r[i]>1e-15 else 0.0
        d2u = (M2-omega2 - LAM4*phi_i**2 + LAM6*phi_i**4
               - Z/np.sqrt(r[i]**2+a**2) + ell*(ell+1)/r[i]**2)*u[i]
        u[i+1] = 2*u[i] - u[i-1] + h*h*d2u
        if not np.isfinite(u[i+1]) or abs(u[i+1])>1e8: u[i+1:]=0.0; break
    return u[-1], u

def find_solitons(omega2, Z=1.0, a=1.0, ell=0, A_max=8.0, n_pts=300):
    A_vals = np.linspace(0.01, A_max, n_pts)
    u_ends = [shoot(A, omega2, Z, a, ell)[0] for A in A_vals]
    results = []
    for i in range(len(A_vals)-1):
        if np.isfinite(u_ends[i]) and np.isfinite(u_ends[i+1]) and u_ends[i]*u_ends[i+1]<0:
            try:
                A_s = brentq(lambda A: shoot(A, omega2, Z, a, ell)[0], A_vals[i], A_vals[i+1], xtol=1e-8)
                _, u = shoot(A_s, omega2, Z, a, ell)
                Q = float(trap(u**2, r)); E = omega2*Q
                nc = int(np.sum(np.diff(np.sign(u[:int(0.95*N)]))!=0))
                phi_c = u[1]/r[1] if ell==0 else A_s
                results.append({'E':E,'Q':Q,'omega2':omega2,'A':A_s,'n_nodes':nc,'phi_core':phi_c,'u':u.copy()})
            except: pass
    return results

def scan_solitons(Z=1.0, a=1.0, ell=0, n_omega=50):
    all_s = []
    for omega2 in np.linspace(0.05, 0.99, n_omega):
        for s in find_solitons(omega2, Z, a, ell):
            if s['E']>0 and not any(abs(s['E']-p['E'])<0.005 for p in all_s):
                all_s.append(s)
    all_s.sort(key=lambda x: x['E'])
    return all_s

def spectral_asymmetry(phi_prof, omega2, Z, a, ell, r_grid, n_eigs=40):
    Npts = len(r_grid); dr = r_grid[1]-r_grid[0]
    def build(mode):
        Veff = np.zeros(Npts)
        for i in range(Npts):
            phi = phi_prof[i]
            Vloc = Vpp(phi) if mode=='amp' else Vp_over_phi(phi)
            Veff[i] = Vloc - omega2 + ell*(ell+1)/r_grid[i]**2 - Z/np.sqrt(r_grid[i]**2+a**2)
        H = np.zeros((Npts,Npts))
        for i in range(Npts):
            H[i,i] = 2.0/dr**2 + Veff[i]
            if i>0: H[i,i-1] = -1.0/dr**2
            if i<Npts-1: H[i,i+1] = -1.0/dr**2
        return H, Veff
    H1,Veff1 = build('amp'); H2,Veff2 = build('phase')
    ne = min(n_eigs, Npts-1)
    eigs1 = eigvalsh(H1, subset_by_index=[0,ne-1])
    eigs2 = eigvalsh(H2, subset_by_index=[0,ne-1])
    n1,n2 = int(np.sum(eigs1<0)), int(np.sum(eigs2<0))
    return {'n_neg_L1':n1,'n_neg_L2':n2,'Delta_n':n1-n2,'eigs_L1':eigs1,'eigs_L2':eigs2}

def main():
    print("="*72)
    print("MFT SPIN-1/2 EMERGENCE: SPECTRAL ASYMMETRY ANALYSIS")
    print("="*72)
    print(f"\n  Silver ratio: δ = {DELTA:.6f}")
    print(f"  Crossing: φ = √2 = δ-1 = {PHI_CROSS:.6f}")
    print(f"  φ_b = {PHI_B:.6f}  φ_v = {PHI_V:.6f}")
    print(f"  (φ_b²+φ_v²)/2 = {(PHI_B**2+PHI_V**2)/2:.6f} = 2 = φ_cross²  ✓")
    print(f"  ΔV(φ) = 2φ²(φ²-2), zero at φ=√2=δ-1")

    # Find leptons
    print(f"\n{'='*72}\nFINDING SOLITONS\n{'='*72}")
    print("\n  Leptons (ℓ=0, Z=1.0)...")
    lep = scan_solitons(Z=1.0, ell=0, n_omega=50)
    print(f"  Found {len(lep)} solutions")
    for i,s in enumerate(lep[:8]):
        print(f"    {i}: E={s['E']:.6f} φ_c={s['phi_core']:.4f} nodes={s['n_nodes']} ω²={s['omega2']:.3f}")

    # Best triple
    R10_T, R21_T = 206.768, 16.817
    best=None; bsc=1e9
    for i in range(len(lep)):
        for j in range(i+1,len(lep)):
            for k in range(j+1,len(lep)):
                E0,E1,E2 = lep[i]['E'],lep[j]['E'],lep[k]['E']
                if E0>0:
                    sc = (np.log(E1/E0/R10_T))**2 + (np.log(E2/E1/R21_T))**2
                    if sc<bsc: bsc=sc; best=(i,j,k)

    if best is None: print("ERROR: no triple found"); return
    s_e,s_mu,s_tau = lep[best[0]], lep[best[1]], lep[best[2]]
    print(f"\n  Triple: e(E={s_e['E']:.5f} φ={s_e['phi_core']:.3f}) "
          f"μ(E={s_mu['E']:.5f} φ={s_mu['phi_core']:.3f}) "
          f"τ(E={s_tau['E']:.5f} φ={s_tau['phi_core']:.3f})")
    print(f"  R10={s_mu['E']/s_e['E']:.1f}  R21={s_tau['E']/s_mu['E']:.2f}")

    # Bosons
    print("\n  Vector bosons (ℓ=1, Z=1.8)...")
    bos = scan_solitons(Z=1.8, ell=1, n_omega=50)
    print(f"  Found {len(bos)} solutions")
    for i,s in enumerate(bos[:4]):
        print(f"    {i}: E={s['E']:.6f} φ_c={s['phi_core']:.4f} nodes={s['n_nodes']}")

    print("\n  Higgs (ℓ=0, Z=1.8)...")
    higgs = scan_solitons(Z=1.8, ell=0, n_omega=50)
    s_H = None
    for s in higgs:
        if s['phi_core'] > PHI_B: s_H = s; break
    if s_H: print(f"    H: E={s_H['E']:.6f} φ_c={s_H['phi_core']:.4f}")

    # Catalog
    cat = {}
    cat['electron'] = (s_e, 0, 1.0, '1/2', 'fermion')
    cat['muon']     = (s_mu, 0, 1.0, '1/2', 'fermion')
    cat['tau']      = (s_tau, 0, 1.0, '1/2', 'fermion')
    if len(bos)>0: cat['W']  = (bos[0], 1, 1.8, '1', 'boson')
    if len(bos)>1: cat['Z⁰'] = (bos[1], 1, 1.8, '1', 'boson')
    if s_H: cat['Higgs'] = (s_H, 0, 1.8, '0', 'boson')

    # Spectral asymmetry
    print(f"\n{'='*72}\nSPECTRAL ASYMMETRY\n{'='*72}")
    res_all = {}
    for name,(sol,ell,Z,exp_sp,exp_st) in cat.items():
        u = sol['u']
        phi_prof = np.array([u[i]/r[i] if r[i]>1e-15 else sol['phi_core'] for i in range(N)])
        res = spectral_asymmetry(phi_prof, sol['omega2'], Z, 1.0, ell, r)
        dn = res['Delta_n']; is_f = (dn%2!=0); J = ell + abs(dn)/2.0
        pred = 'fermion' if is_f else 'boson'; m = '✓' if pred==exp_st else '✗'
        res_all[name] = {**res,'ell':ell,'J_eff':J,'pred':pred,'exp_st':exp_st,
                         'exp_sp':exp_sp,'match':m,'phi_core':sol['phi_core'],'phi_prof':phi_prof}
        regime = 'φ>√2' if sol['phi_core']>PHI_CROSS else 'φ<√2'
        print(f"\n  {name:10s} ℓ={ell} φ_c={sol['phi_core']:.4f} ({regime})")
        print(f"    n_neg(L₁)={res['n_neg_L1']:2d}  n_neg(L₂)={res['n_neg_L2']:2d}  Δn={dn:+d}  J_eff={J:.1f}")
        print(f"    → {pred} (exp: {exp_st}, spin {exp_sp}) {m}")
        print(f"    Low L₁: [{', '.join(f'{e:.3f}' for e in res['eigs_L1'][:5])}]")
        print(f"    Low L₂: [{', '.join(f'{e:.3f}' for e in res['eigs_L2'][:5])}]")

    # Summary
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"\n  {'Part':<10} {'ℓ':>2} {'φ_c':>7} {'rgm':>5} {'n₁':>4} {'n₂':>4} {'Δn':>4} {'J':>5} {'pred':>8} {'exp':>8} {'':>2}")
    print("  "+"-"*70)
    nm = 0
    for name,res in res_all.items():
        rgm = '> √2' if res['phi_core']>PHI_CROSS else '< √2'
        print(f"  {name:<10} {res['ell']:>2} {res['phi_core']:>7.4f} {rgm:>5} "
              f"{res['n_neg_L1']:>4} {res['n_neg_L2']:>4} {res['Delta_n']:>+4d} "
              f"{res['J_eff']:>5.1f} {res['pred']:>8} {res['exp_st']:>8} {res['match']:>2}")
        if res['match']=='✓': nm+=1
    print(f"\n  Score: {nm}/{len(res_all)}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(r"MFT Spin-$\frac{1}{2}$ Emergence: Spectral Asymmetry of Q-Ball Fluctuations"
                 "\n" r"$\Delta V = 2\varphi^2(\varphi^2 - 2)$ crosses zero at "
                 r"$\varphi = \sqrt{2} = \delta - 1$ (silver ratio)", fontsize=13, fontweight='bold')
    cols = {'electron':'green','muon':'blue','tau':'red','W':'purple','Z⁰':'darkviolet','Higgs':'brown'}
    phi_arr = np.linspace(0.001, 2.5, 500)

    ax=axes[0,0]; ax.plot(phi_arr, DeltaV(phi_arr), 'k-', lw=2.5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(PHI_CROSS, color='red', lw=2, ls='--', label='φ=√2=δ−1')
    ax.axvline(PHI_B, color='orange', lw=1.5, ls=':'); ax.axvline(PHI_V, color='purple', lw=1.5, ls=':')
    ax.fill_between(phi_arr, DeltaV(phi_arr), 0, where=DeltaV(phi_arr)<0, alpha=0.12, color='blue')
    ax.fill_between(phi_arr, DeltaV(phi_arr), 0, where=DeltaV(phi_arr)>0, alpha=0.12, color='red')
    for n,res in res_all.items():
        ax.plot(res['phi_core'], DeltaV(res['phi_core']), 'o', color=cols.get(n,'gray'), ms=8, label=n)
    ax.set_xlabel('φ'); ax.set_ylabel('ΔV(φ)'); ax.set_title('Operator splitting ΔV(φ)')
    ax.legend(fontsize=7); ax.set_ylim(-6,14); ax.grid(True, alpha=0.3)

    ax=axes[0,1]; ax.plot(phi_arr, Vpp(phi_arr), 'b-', lw=2, label="V″(φ) [L₁]")
    ax.plot(phi_arr, Vp_over_phi(phi_arr), 'r--', lw=2, label="V′(φ)/φ [L₂]")
    ax.axvline(PHI_CROSS, color='green', lw=2, ls='--', label='crossing: √2')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('φ'); ax.set_ylabel('Eff. potential'); ax.set_title('Amplitude vs phase potentials')
    ax.legend(fontsize=8); ax.set_ylim(-5,15); ax.grid(True,alpha=0.3)

    ax=axes[0,2]
    for n,res in res_all.items():
        ax.plot(r, DeltaV(res['phi_prof']), lw=2, color=cols.get(n,'gray'), label=f"{n} Δn={res['Delta_n']:+d}")
    ax.axhline(0, color='gray', lw=1); ax.set_xlabel('r'); ax.set_ylabel('ΔV(φ(r))')
    ax.set_title('ΔV along soliton profiles'); ax.legend(fontsize=7); ax.set_xlim(0,15); ax.grid(True,alpha=0.3)

    ax=axes[1,0]; names=list(res_all.keys()); x=np.arange(len(names)); w=0.35
    n1s=[res_all[n]['n_neg_L1'] for n in names]; n2s=[res_all[n]['n_neg_L2'] for n in names]
    ax.bar(x-w/2, n1s, w, label='n_neg(L₁)', color='steelblue', alpha=0.8)
    ax.bar(x+w/2, n2s, w, label='n_neg(L₂)', color='coral', alpha=0.8)
    for i,n in enumerate(names):
        dn=res_all[n]['Delta_n']
        ax.text(i, max(n1s[i],n2s[i])+0.3, f'Δn={dn:+d}', ha='center', fontsize=9, fontweight='bold',
                color='green' if dn%2!=0 else 'darkred')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Neg eigenvalues'); ax.set_title('Bound states: L₁ vs L₂'); ax.legend(); ax.grid(True,alpha=0.3,axis='y')

    ax=axes[1,1]
    for i,n in enumerate(names):
        e1=res_all[n]['eigs_L1'][:10]; e2=res_all[n]['eigs_L2'][:10]
        ax.scatter(e1,[i-0.12]*len(e1), marker='|', s=120, color='steelblue', linewidths=2, zorder=3)
        ax.scatter(e2,[i+0.12]*len(e2), marker='|', s=120, color='coral', linewidths=2, zorder=3)
    ax.axvline(0, color='black', lw=1.5); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('Eigenvalue'); ax.set_title('Spectra (blue=L₁, red=L₂)'); ax.set_xlim(-5,3); ax.grid(True,alpha=0.3)

    ax=axes[1,2]; Js=[res_all[n]['J_eff'] for n in names]
    bc=['green' if res_all[n]['match']=='✓' else 'red' for n in names]
    ax.bar(x, Js, color=bc, alpha=0.7, edgecolor='black')
    ax.axhline(0.5, color='blue', ls='--', lw=1, label='spin ½')
    ax.axhline(1.0, color='orange', ls='--', lw=1, label='spin 1')
    ax.axhline(0.0, color='gray', ls='--', lw=1, label='spin 0')
    for i,n in enumerate(names): ax.text(i, Js[i]+0.08, f"exp:{res_all[n]['exp_sp']}", ha='center', fontsize=8, color='gray')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('J_eff'); ax.set_title('Predicted spin'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3,axis='y')

    plt.tight_layout(rect=[0,0,1,0.91])
    out = outpath('mft_spin_half_emergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); print(f"\n  Plot saved: {out}")

if __name__=='__main__': main()
