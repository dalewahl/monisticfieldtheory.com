#!/usr/bin/env python3
"""
MFT SPIN-1/2 v7: 4D GRAVITON PHASE-LOCKING COMPUTATION
=========================================================
Tests the prediction from the 4D framework document:
  Locking energy E_lock ∝ β × ω² × Q
  Fermions: high ω (E_lock large → locked → spin 1/2)
  Bosons:   low ω  (E_lock small → unlocked → integer spin)

The locking threshold separates fermions from bosons.
"""
import numpy as np
try:
    from numpy import trapezoid as trap
except ImportError:
    from numpy import trapz as trap
from scipy.optimize import brentq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def outpath(fn): return os.path.join(SCRIPT_DIR, fn)

M2, LAM4, LAM6 = 1.0, 2.0, 0.5
DELTA = 1+np.sqrt(2)
PHI_B = np.sqrt(2-np.sqrt(2))
PHI_V = np.sqrt(2+np.sqrt(2))

RMAX = 20.0; N = 200
r = np.linspace(RMAX/(N*100), RMAX, N)
dr = r[1]-r[0]

def V(phi): return 0.5*M2*phi**2 - 0.25*LAM4*phi**4 + (1/6.)*LAM6*phi**6
def Vpp(phi): return M2 - 3*LAM4*phi**2 + 5*LAM6*phi**4

def shoot(A, omega2, Z=1.0, a=1.0, ell=0):
    u=np.zeros(N); u[0]=0; u[1]=A*r[1]**(ell+1) if ell>0 else A*r[1]
    for i in range(1,N-1):
        p=u[i]/r[i] if r[i]>1e-15 else 0.0
        d2u=(M2-omega2-LAM4*p**2+LAM6*p**4-Z/np.sqrt(r[i]**2+a**2)+ell*(ell+1)/r[i]**2)*u[i]
        u[i+1]=2*u[i]-u[i-1]+dr*dr*d2u
        if not np.isfinite(u[i+1]) or abs(u[i+1])>1e8: u[i+1:]=0; break
    return u[-1],u

def scan(Z=1.0, a=1.0, ell=0, nw=60):
    a_s=[]
    for w2 in np.linspace(0.05,0.99,nw):
        Av=np.linspace(0.01,8.0,400)
        ue=[shoot(A,w2,Z,a,ell)[0] for A in Av]
        for i in range(len(Av)-1):
            if np.isfinite(ue[i]) and np.isfinite(ue[i+1]) and ue[i]*ue[i+1]<0:
                try:
                    As=brentq(lambda A:shoot(A,w2,Z,a,ell)[0],Av[i],Av[i+1],xtol=1e-10)
                    _,u=shoot(As,w2,Z,a,ell); Q=float(trap(u**2,r)); E=w2*Q
                    nc=int(np.sum(np.diff(np.sign(u[:int(0.95*N)]))!=0))
                    pc=u[1]/r[1] if ell==0 else As
                    s={'E':E,'Q':Q,'omega2':w2,'A':As,'n_nodes':nc,'phi_core':pc,'u':u.copy()}
                    if E>0 and not any(abs(E-p['E'])<0.005 for p in a_s): a_s.append(s)
                except: pass
    a_s.sort(key=lambda x:x['E']); return a_s

def main():
    print("="*72)
    print("MFT SPIN-1/2 v7: 4D GRAVITON PHASE-LOCKING")
    print("="*72)

    # Find all solitons
    print("\n  Finding solitons...")
    lep = scan(Z=1.0, ell=0, nw=60)
    R10_T, R21_T = 206.768, 16.817
    best=None; bsc=1e9
    for i in range(len(lep)):
        for j in range(i+1,len(lep)):
            for k in range(j+1,len(lep)):
                E0,E1,E2=lep[i]['E'],lep[j]['E'],lep[k]['E']
                if E0>0:
                    sc=(np.log(E1/E0/R10_T))**2+(np.log(E2/E1/R21_T))**2
                    if sc<bsc: bsc=sc;best=(i,j,k)
    if not best: print("ERROR"); return
    s_e,s_mu,s_tau=lep[best[0]],lep[best[1]],lep[best[2]]

    bos=scan(Z=1.8, ell=1, nw=60)
    hig=scan(Z=1.8, ell=0, nw=60)
    s_H=None
    for s in hig:
        if s['phi_core']>PHI_B: s_H=s; break

    cat = {}
    cat['electron'] = (s_e,  0, 1.0, '1/2', 'fermion')
    cat['muon']     = (s_mu, 0, 1.0, '1/2', 'fermion')
    cat['tau']      = (s_tau,0, 1.0, '1/2', 'fermion')
    if len(bos)>0: cat['W']    = (bos[0], 1, 1.8, '1', 'boson')
    if len(bos)>1: cat['Z⁰']   = (bos[1], 1, 1.8, '1', 'boson')
    if s_H:        cat['Higgs'] = (s_H,    0, 1.8, '0', 'boson')

    # Compute locking quantities for each particle
    print(f"\n{'='*72}")
    print("LOCKING ENERGY ANALYSIS")
    print("="*72)
    print(f"\n  Locking energy: E_lock ∝ β × ω² × Q")
    print(f"  ω² controls the coupling between Q-ball phase and graviton")
    print(f"  Q is the contraction charge (integral of u²)")
    print(f"  ℓ determines whether spatial angular momentum pre-empts locking")

    print(f"\n  {'Particle':<10} {'ℓ':>2} {'Z':>4} {'ω²':>8} {'ω':>8} {'Q':>10} "
          f"{'ω²Q':>10} {'φ_c':>7} {'exp':>8}")
    print("  "+"-"*75)

    results = {}
    for pname,(sol,ell,Z,exp_sp,exp_st) in cat.items():
        omega2 = sol['omega2']
        omega = np.sqrt(omega2)
        Q = sol['Q']
        lock_param = omega2 * Q  # proportional to locking energy

        # Soliton radius (where u drops to half-max)
        u = sol['u']
        u_max = np.max(np.abs(u))
        r_half = RMAX
        for i in range(len(u)):
            if np.abs(u[i]) >= u_max * 0.5:
                for j in range(i, len(u)):
                    if np.abs(u[j]) < u_max * 0.5:
                        r_half = r[j]; break
                break

        # Stiffness at core
        phi_prof = np.array([u[i]/r[i] if r[i]>1e-15 else sol['phi_core'] for i in range(N)])
        Vpp_core = Vpp(sol['phi_core'])

        results[pname] = {
            'omega2': omega2, 'omega': omega, 'Q': Q,
            'lock_param': lock_param, 'ell': ell, 'Z': Z,
            'phi_core': sol['phi_core'], 'r_half': r_half,
            'Vpp_core': Vpp_core, 'E': sol['E'],
            'exp_sp': exp_sp, 'exp_st': exp_st,
        }

        print(f"  {pname:<10} {ell:>2} {Z:>4.1f} {omega2:>8.4f} {omega:>8.4f} {Q:>10.4f} "
              f"{lock_param:>10.4f} {sol['phi_core']:>7.4f} {exp_st:>8}")

    # Find the threshold that separates fermions from bosons
    print(f"\n{'='*72}")
    print("THRESHOLD SEARCH")
    print("="*72)

    fermion_locks = [results[n]['lock_param'] for n in results if results[n]['exp_st']=='fermion']
    boson_locks = [results[n]['lock_param'] for n in results if results[n]['exp_st']=='boson']

    if fermion_locks and boson_locks:
        min_fermion = min(fermion_locks)
        max_boson = max(boson_locks)
        print(f"\n  Fermion ω²Q values: {[f'{x:.4f}' for x in sorted(fermion_locks)]}")
        print(f"  Boson ω²Q values:   {[f'{x:.4f}' for x in sorted(boson_locks)]}")
        print(f"\n  Min fermion ω²Q = {min_fermion:.4f}")
        print(f"  Max boson ω²Q   = {max_boson:.4f}")

        if min_fermion > max_boson:
            threshold = (min_fermion + max_boson) / 2
            gap = min_fermion / max_boson
            print(f"\n  ✓ CLEAN SEPARATION EXISTS")
            print(f"  Threshold: ω²Q = {threshold:.4f}")
            print(f"  Gap ratio: {gap:.2f}×")

            # Test classification at this threshold
            print(f"\n  Classification at threshold ω²Q = {threshold:.4f}:")
            n_ok = 0
            for pname, res in results.items():
                is_fermion = res['lock_param'] > threshold
                pred = 'fermion' if is_fermion else 'boson'
                match = '✓' if pred == res['exp_st'] else '✗'
                if match == '✓': n_ok += 1
                print(f"    {pname:10s} ω²Q={res['lock_param']:>10.4f}  "
                      f"→ {pred:7s} (exp: {res['exp_st']:7s}) {match}")
            print(f"\n  Score: {n_ok}/{len(results)}")
        else:
            print(f"\n  ✗ NO CLEAN SEPARATION by ω²Q alone")
            print(f"  Overlap: fermion min = {min_fermion:.4f}, boson max = {max_boson:.4f}")

    # Also test ω² alone (without Q)
    print(f"\n  Testing ω² alone (without charge Q):")
    fermion_w2 = [results[n]['omega2'] for n in results if results[n]['exp_st']=='fermion']
    boson_w2 = [results[n]['omega2'] for n in results if results[n]['exp_st']=='boson']
    min_f_w2 = min(fermion_w2)
    max_b_w2 = max(boson_w2)
    print(f"  Fermion ω²: {[f'{x:.4f}' for x in sorted(fermion_w2)]}")
    print(f"  Boson ω²:   {[f'{x:.4f}' for x in sorted(boson_w2)]}")
    if min_f_w2 > max_b_w2:
        thresh_w2 = (min_f_w2 + max_b_w2) / 2
        gap_w2 = min_f_w2 / max_b_w2
        print(f"  ✓ CLEAN SEPARATION by ω² alone!")
        print(f"  Threshold: ω² = {thresh_w2:.4f}")
        print(f"  Gap: {gap_w2:.1f}×")
        n_ok2 = 0
        for pname, res in results.items():
            pred = 'fermion' if res['omega2'] > thresh_w2 else 'boson'
            match = '✓' if pred == res['exp_st'] else '✗'
            if match == '✓': n_ok2 += 1
            print(f"    {pname:10s} ω²={res['omega2']:.4f}  → {pred:7s} (exp: {res['exp_st']:7s}) {match}")
        print(f"  Score: {n_ok2}/{len(results)}")
    else:
        print(f"  ✗ No clean separation by ω² alone")

    # Test ℓ-dependent classification
    print(f"\n  Testing combined (ℓ, ω²) classification:")
    print(f"  Rule: fermion if ℓ=0 AND ω² > threshold; boson otherwise")
    n_ok3 = 0
    for pname, res in results.items():
        is_fermion = (res['ell'] == 0 and res['omega2'] > 0.5)
        # Exception: Higgs is ℓ=0 but boson — check if its ω² is below threshold
        pred = 'fermion' if is_fermion else 'boson'
        match = '✓' if pred == res['exp_st'] else '✗'
        if match == '✓': n_ok3 += 1
        print(f"    {pname:10s} ℓ={res['ell']} ω²={res['omega2']:.4f}  "
              f"→ {pred:7s} (exp: {res['exp_st']:7s}) {match}")
    print(f"  Score: {n_ok3}/{len(results)}")

    # ── PLOT ─────────────────────────────────────────────────────
    cols = {'electron':'green','muon':'blue','tau':'red','W':'purple','Z⁰':'darkviolet','Higgs':'brown'}
    names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(r"MFT Spin-$\frac{1}{2}$ v7: 4D Graviton Phase-Locking"
                 "\n" r"Fermion/boson separation by $\omega^2$ (Q-ball internal frequency)",
                 fontsize=13, fontweight='bold')

    # Panel 1: ω² for each particle
    ax = axes[0]
    x = np.arange(len(names))
    w2s = [results[n]['omega2'] for n in names]
    bc = ['green' if results[n]['exp_st']=='fermion' else 'orange' for n in names]
    ax.bar(x, w2s, color=bc, alpha=0.7, edgecolor='black')
    for i,n in enumerate(names):
        ax.text(i, w2s[i]+0.02, f"ℓ={results[n]['ell']}", ha='center', fontsize=8)
    if min_f_w2 > max_b_w2:
        ax.axhline((min_f_w2+max_b_w2)/2, color='red', ls='--', lw=2, label='threshold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('ω² (Q-ball frequency²)')
    ax.set_title('Internal frequency: fermions vs bosons')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: ω² vs φ_core (scatter)
    ax = axes[1]
    for n in names:
        res = results[n]
        marker = 'o' if res['exp_st']=='fermion' else 's'
        ax.plot(res['phi_core'], res['omega2'], marker, color=cols.get(n,'gray'),
                ms=12, label=f"{n} ({res['exp_st'][:1].upper()})")
    ax.axvline(PHI_B, color='orange', ls=':', label='φ_b')
    ax.axvline(PHI_V, color='purple', ls=':', label='φ_v')
    ax.axvline(np.sqrt(2), color='red', ls='--', lw=1.5, label='√2=δ−1')
    if min_f_w2 > max_b_w2:
        ax.axhline((min_f_w2+max_b_w2)/2, color='red', ls='--', lw=2, alpha=0.5)
        ax.fill_between([0,3], max_b_w2, 1.0, alpha=0.08, color='green', label='fermion zone')
        ax.fill_between([0,3], 0, max_b_w2, alpha=0.08, color='orange', label='boson zone')
    ax.set_xlabel('φ_core'); ax.set_ylabel('ω²')
    ax.set_title('ω² vs core field: the classification plane')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 1.0)

    # Panel 3: ω²Q (full locking parameter)
    ax = axes[2]
    lps = [results[n]['lock_param'] for n in names]
    ax.bar(x, lps, color=bc, alpha=0.7, edgecolor='black')
    for i,n in enumerate(names):
        ax.text(i, lps[i]+max(lps)*0.02, f"{lps[i]:.2f}", ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('ω²Q (locking energy parameter)')
    ax.set_title('Locking energy: fermions vs bosons')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0,0,1,0.90])
    out = outpath('mft_spin_4d_locking.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {out}")

if __name__=='__main__': main()
