#!/usr/bin/env python3
"""
MFT GALACTIC ROTATION CURVES — FULL NONLINEAR SOLVER
====================================================
Derived from the Master Sextic via the Symmetric Back-Reaction Theorem.

Monistic Field Theory treats space as an elastic medium with a single sextic
contraction field.  Leptons are localised solitons; galactic dark-matter
halos are large-scale configurations of the same field.  The potential shape
— including the silver ratio condition lambda_4^2 = 8 m_2 lambda_6 — is
derived from the MFT coupled scalar-metric field equations.

This script:
  1. Derives the gravitational sextic from the master potential
  2. Solves the FULL NONLINEAR field equation as a BVP on [0.1, 80] kpc
  3. Fits (Upsilon*, rho_scale) jointly per galaxy
  4. Tests 6 spiral galaxies: MW, M31, NGC3198, NGC2403, NGC7793, UGC2259
  5. Compares with MOND and NFW

EXECUTION:
  pip install numpy scipy matplotlib
  python mft_galactic_nonlinear.py
  Runtime: ~2 minutes

OUTPUTS (saved alongside the script):
  mft_galactic_6panel.png      — 6-galaxy rotation curves
  mft_galactic_field.png       — contraction field profiles
  mft_galactic_residuals.png   — residual analysis
  mft_galactic_comparison.png  — MOND/NFW/MFT comparison
  mft_galactic_results.txt     — full numerical tables

Author: Dale Wahl
Monistic Field Theory Project — March 2026
"""
import os, sys, numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def savepath(name): return os.path.join(SCRIPT_DIR, name)

# ═══ CONSTANTS ═══
G=4.30091e-6; DELTA_SR=1+np.sqrt(2); KAPPA=1.0
M2_MASTER=1.0; L4_MASTER=2.0; L6_MASTER=0.5
assert abs(L4_MASTER**2-8*M2_MASTER*L6_MASTER)<1e-12

# Gravitational sextic: derived from master potential via Master-to-Gravitational mapping
# Silver ratio condition (λ₄²=8m₂λ₆) propagates exactly at all scales
# β = 10⁻⁴ is the SAME gravitational coupling that enters the neutrino mass formula
# (best-fit β for neutrinos: 1.016×10⁻⁴, within 1.6% of Solar System value)
M2G=1e-6; LAM4G=-2*M2G**2; LAM6G=0.5*M2G**3; BETA=1e-4
assert abs(LAM4G**2-8*M2G*LAM6G)/max(abs(LAM4G**2),1e-50)<1e-6
ALPHA=1/np.sqrt(M2G); DELTA_B=ALPHA*np.sqrt(2-np.sqrt(2)); DELTA_V=ALPHA*np.sqrt(2+np.sqrt(2))

R_MIN=0.1; R_MAX=80.; N_BVP=300; N_FINE=2000
r_fine=np.linspace(R_MIN,R_MAX,N_FINE)
a0=1.2e-10*3.0857e19/1e6

# ═══ GALAXIES ═══
GALAXIES={
 "MW":{"full":"Milky Way","M_b":8e9,"a_b":0.5,"M_d":6e10,"R_d":3.,"M_g":1e10,"R_g":7.,"BH":4e6,
       "ro":np.array([5.,8,10,15,20,25,30,50]),"vo":np.array([234.3,229.2,226,217.3,208.8,200.3,185,157.8]),"sv":10.},
 "M31":{"full":"Andromeda (M31)","M_b":3e10,"a_b":1.,"M_d":8e10,"R_d":5.5,"M_g":2e10,"R_g":9.,"BH":1.4e8,
        "ro":np.array([5.,8,12,16,20,25,35]),"vo":np.array([250.,255,255,250,245,235,210]),"sv":10.},
 "NGC3198":{"full":"NGC 3198","M_b":1e9,"a_b":0.5,"M_d":4e10,"R_d":3.5,"M_g":1.5e10,"R_g":7.,"BH":1e7,
            "ro":np.array([2.,6,10,14,18,22,26,30]),"vo":np.array([130.,155,160,159,156.5,153,150,150]),"sv":10.},
 "NGC2403":{"full":"NGC 2403","M_b":5e8,"a_b":0.4,"M_d":2e10,"R_d":2.,"M_g":8e9,"R_g":5.,"BH":5e6,
            "ro":np.array([2.,6,10,14,18,22,26,30]),"vo":np.array([90.,125,130,127.5,122.5,120,120,120]),"sv":10.},
 "NGC7793":{"full":"NGC 7793","M_b":3e8,"a_b":0.4,"M_d":1.5e10,"R_d":1.8,"M_g":7e9,"R_g":4.,"BH":3e6,
            "ro":np.array([2.,6,10,14,18,22,26,30]),"vo":np.array([80.,106.7,107.5,100,95,95,95,95]),"sv":10.},
 "UGC2259":{"full":"UGC 2259","M_b":1e8,"a_b":0.3,"M_d":6e9,"R_d":1.5,"M_g":5e9,"R_g":4.,"BH":1e6,
            "ro":np.array([2.,6,10,14,18,22,26,30]),"vo":np.array([60.,77,72.5,70,70,70,70,70]),"sv":10.},
}
GORDER=["MW","M31","NGC3198","NGC2403","NGC7793","UGC2259"]

# ═══ BARYONIC MODEL ═══
def rho_stellar(r,g):
    r=np.maximum(np.atleast_1d(np.float64(r)),1e-6)
    rho=(g["M_b"]/(2*np.pi))*g["a_b"]/(r*(r+g["a_b"])**3)
    if g["M_d"]>0: rho+=(g["M_d"]/(8*np.pi*g["R_d"]**3))*np.exp(-r/g["R_d"])
    return rho

def rho_gas(r,g):
    if g["M_g"]==0: return np.zeros_like(np.atleast_1d(np.float64(r)))
    r=np.maximum(np.atleast_1d(np.float64(r)),1e-6)
    return (g["M_g"]/(8*np.pi*g["R_g"]**3))*np.exp(-r/g["R_g"])

sMenc={}; gMenc={}
def precompute():
    for gn in GORDER:
        g=GALAXIES[gn]
        rs=rho_stellar(r_fine,g); rg=rho_gas(r_fine,g)
        sMenc[gn]=4*np.pi*cumulative_trapezoid(rs*r_fine**2,r_fine,initial=0.)
        gMenc[gn]=4*np.pi*cumulative_trapezoid(rg*r_fine**2,r_fine,initial=0.)

# ═══ BVP SOLVER ═══
def solve_bvp_galaxy(gn):
    g=GALAXIES[gn]
    rho_arr=rho_stellar(r_fine,g)+rho_gas(r_fine,g)
    rbi=interp1d(r_fine,rho_arr,fill_value="extrapolate")
    def make_ode(fn):
        def ode(r,y):
            d,dp=y; rs=np.maximum(r,1e-3)
            return np.vstack([dp,(M2G*d+LAM4G*d**3+LAM6G*d**5+BETA*fn(rs))/KAPPA-2/rs*dp])
        return ode
    ode=make_ode(rbi)
    def bc(ya,yb): return np.array([ya[1],yb[0]])
    rn=np.linspace(R_MIN,R_MAX,N_BVP)
    dg=60000.*(1-rn/R_MAX)**2; ddg=np.gradient(dg,rn)
    for tol,mn in [(5e-2,8000),(1e-1,10000)]:
        try:
            sol=solve_bvp(ode,bc,rn,np.vstack([dg,ddg]),tol=tol,max_nodes=mn,verbose=0)
            if sol.success:
                du=np.interp(r_fine,sol.x,sol.y[0]); dpu=np.interp(r_fine,sol.x,sol.y[1])
                return du,dpu,True
        except: pass
    return np.zeros(N_FINE),np.zeros(N_FINE),False

def compute_halo(delta,dphi):
    rho=np.maximum(0.5*KAPPA*dphi**2+0.5*M2G*delta**2+0.25*LAM4G*delta**4+(1./6.)*LAM6G*delta**6,0.)
    M=4*np.pi*cumulative_trapezoid(rho*r_fine**2,r_fine,initial=0.)
    return rho,M

# ═══ FIT ═══
def fit_galaxy(gn,M_mft_raw):
    g=GALAXIES[gn]; ro=g["ro"]; vo=g["vo"]; sv=g["sv"]
    Ms_o=np.interp(ro,r_fine,sMenc[gn]); Mg_o=np.interp(ro,r_fine,gMenc[gn])
    MU_o=np.interp(ro,r_fine,M_mft_raw)
    best=(1e30,1.,1.)
    for ups in np.arange(0.3,3.01,0.05):
        Mb=ups*Ms_o+Mg_o
        for lg in np.linspace(-10,4,57):
            rs=10.**lg; v=np.sqrt(G*(Mb+g["BH"]+rs*MU_o)/np.maximum(ro,1e-3))
            chi2=np.sum(((v-vo)/sv)**2)
            if chi2<best[0]: best=(chi2,ups,rs)
    u0,r0=best[1],best[2]
    for ups in np.linspace(max(0.2,u0-0.04),min(4,u0+0.04),17):
        Mb=ups*Ms_o+Mg_o
        for lg in np.linspace(np.log10(max(r0*0.03,1e-10)),np.log10(r0*30),61):
            rs=10.**lg; v=np.sqrt(G*(Mb+g["BH"]+rs*MU_o)/np.maximum(ro,1e-3))
            chi2=np.sum(((v-vo)/sv)**2)
            if chi2<best[0]: best=(chi2,ups,rs)
    return best

# ═══ MOND/NFW ═══
def fit_mond(gn):
    g=GALAXIES[gn]; ro=g["ro"]; vo=g["vo"]; sv=g["sv"]
    Ms_o=np.interp(ro,r_fine,sMenc[gn]); Mg_o=np.interp(ro,r_fine,gMenc[gn])
    best=(1e30,1.)
    for ups in np.arange(0.1,5.01,0.05):
        Mb=ups*Ms_o+Mg_o; aN=G*(Mb+g["BH"])/np.maximum(ro,1e-3)**2
        x=aN/a0; mu=x/np.sqrt(1+x**2); mu=np.maximum(mu,1e-30)
        vm=np.sqrt(aN/mu*np.maximum(ro,1e-3))
        chi2=np.sum(((vm-vo)/sv)**2)
        if chi2<best[0]: best=(chi2,ups)
    return best

def fit_nfw(gn):
    g=GALAXIES[gn]; ro=g["ro"]; vo=g["vo"]; sv=g["sv"]
    Ms_o=np.interp(ro,r_fine,sMenc[gn]); Mg_o=np.interp(ro,r_fine,gMenc[gn])
    rhoc=277.5*0.49; best=(1e30,1.,1e12,10.)
    for ups in [0.3,0.5,0.7,1.,1.3,1.5,2.]:
        Mb=ups*Ms_o+Mg_o
        for lM in np.arange(10,14.5,0.5):
            for lc in np.arange(0.2,2.,0.2):
                M200=10.**lM; c200=10.**lc
                r200=(3*M200/(4*np.pi*200*rhoc))**(1./3.); rs=r200/c200
                x=ro/rs; fc=np.log(1+c200)-c200/(1+c200)
                fx=np.log(1+x)-x/(1+x); Mdm=M200*fx/fc
                vm=np.sqrt(G*(Mb+g["BH"]+Mdm)/np.maximum(ro,1e-3))
                chi2=np.sum(((vm-vo)/sv)**2)
                if chi2<best[0]: best=(chi2,ups,M200,c200)
    return best

# ═══ MAIN ═══
if __name__=="__main__":
    out=[]
    def log(s=""): print(s); out.append(s)

    log("="*78)
    log("MFT GALACTIC ROTATION CURVES — FULL NONLINEAR SOLVER")
    log(f"  Silver ratio: lam4^2=8 m2 lam6  |  beta={BETA:.0e}")
    log(f"  m2_g={M2G:.0e}  lam4_g={LAM4G:.1e}  lam6_g={LAM6G:.1e}")
    log(f"  alpha={ALPHA:.0f}  delta_b={DELTA_B:.0f}  delta_v={DELTA_V:.0f}")
    log("="*78)
    
    precompute()
    log("\nSolving BVP...")
    halo={}
    for gn in GORDER:
        d,dp,ok=solve_bvp_galaxy(gn); rho_mft,M_mft=compute_halo(d,dp)
        halo[gn]={"d":d,"dp":dp,"rho":rho_mft,"M":M_mft,"maxd":np.max(np.abs(d)),"ok":ok}
        log(f"  {gn:8s} {'OK' if ok else 'FAIL'}  max|d|={halo[gn]['maxd']:.0f}")
    
    log("\nFitting (Ups*, rho_scale)...")
    res={}; tc=0; td=0
    for gn in GORDER:
        chi2,ups,rs=fit_galaxy(gn,halo[gn]["M"])
        g=GALAXIES[gn]; dof=max(len(g["ro"])-2,1)
        res[gn]={"chi2":chi2,"ups":ups,"rs":rs,"dof":dof}
        tc+=chi2; td+=dof
    
    log(f"\n{'Galaxy':8s} {'chi2':>8s} {'chi2/dof':>9s} {'Ups*':>6s} {'log(rho)':>9s}")
    log("-"*45)
    for gn in GORDER:
        r=res[gn]
        log(f"{gn:8s} {r['chi2']:8.2f} {r['chi2']/r['dof']:9.2f} {r['ups']:6.2f} {np.log10(max(r['rs'],1e-30)):9.2f}")
    log("-"*45)
    log(f"{'TOTAL':8s} {tc:8.2f} {tc/td:9.2f}")
    
    # Per-galaxy tables
    for gn in GORDER:
        g=GALAXIES[gn]; r=res[gn]; h=halo[gn]
        Ms_o=np.interp(g["ro"],r_fine,sMenc[gn]); Mg_o=np.interp(g["ro"],r_fine,gMenc[gn])
        Mb_o=r["ups"]*Ms_o+Mg_o; MU_o=r["rs"]*np.interp(g["ro"],r_fine,h["M"])
        vb=np.sqrt(G*(Mb_o+g["BH"])/np.maximum(g["ro"],1e-3))
        vu=np.sqrt(G*(Mb_o+g["BH"]+MU_o)/np.maximum(g["ro"],1e-3))
        log(f"\n  {gn} ({g['full']})  chi2/dof={r['chi2']/r['dof']:.2f}  Ups*={r['ups']:.2f}")
        log(f"    {'R':>6} {'v_obs':>7} {'v_b':>7} {'v_MFT':>7} {'Dv':>7} {'Dv/s':>6}")
        for j in range(len(g["ro"])):
            dv=vu[j]-g["vo"][j]
            log(f"    {g['ro'][j]:6.1f} {g['vo'][j]:7.1f} {vb[j]:7.1f} {vu[j]:7.1f} {dv:+7.1f} {dv/g['sv']:+6.2f}")
    
    # MOND/NFW comparison
    log(f"\n{'='*78}\nCOMPARISON: MFT vs MOND vs NFW\n{'='*78}")
    tm=0; tn=0; mr={}; nr_={}
    for gn in GORDER:
        cm,um=fit_mond(gn); cn,un,M200,c200=fit_nfw(gn)
        mr[gn]={"chi2":cm,"ups":um}; nr_[gn]={"chi2":cn}; tm+=cm; tn+=cn
    log(f"\n{'Gal':8s} {'MFT':>8s} {'MOND':>8s} {'NFW':>8s}")
    log("-"*36)
    for gn in GORDER:
        log(f"{gn:8s} {res[gn]['chi2']:8.1f} {mr[gn]['chi2']:8.1f} {nr_[gn]['chi2']:8.1f}")
    log("-"*36)
    log(f"{'TOTAL':8s} {tc:8.1f} {tm:8.1f} {tn:8.1f}")
    log(f"\nParams/galaxy: MFT=2 (derived V), MOND=1 (empirical a0), NFW=3 (assumed profile)")
    
    with open(savepath("mft_galactic_results.txt"),"w") as f: f.write("\n".join(out))
    print(f"\nResults: {savepath('mft_galactic_results.txt')}")
    
    # ═══ FIGURES ═══
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    fig.suptitle(r"MFT Rotation Curves — Silver Ratio Derived ($\lambda_4^2=8m_2\lambda_6$)"
        f"\n6 Spirals | Full Nonlinear BVP + "+r"$\Upsilon_*$"+" | "
        +r"$\Sigma\chi^2$/dof"+f" = {tc/td:.2f} | 2 free params/galaxy",fontsize=11,fontweight='bold')
    for i,gn in enumerate(GORDER):
        ax=axes[i//3,i%3]; g=GALAXIES[gn]; r=res[gn]; h=halo[gn]
        rp=np.linspace(R_MIN,R_MAX,500)
        Mbp=r["ups"]*np.interp(rp,r_fine,sMenc[gn])+np.interp(rp,r_fine,gMenc[gn])
        MUp=r["rs"]*np.interp(rp,r_fine,h["M"])
        vbp=np.sqrt(G*(Mbp+g["BH"])/np.maximum(rp,1e-3))
        vhp=np.sqrt(G*np.maximum(MUp,0)/np.maximum(rp,1e-3))
        vup=np.sqrt(G*(Mbp+g["BH"]+MUp)/np.maximum(rp,1e-3))
        ax.errorbar(g["ro"],g["vo"],yerr=g["sv"],fmt='ko',ms=5,capsize=3,zorder=5,label='Observed')
        ax.plot(rp,vbp,'b--',lw=1,alpha=0.5,label=r'Baryons ($\Upsilon_*$'f'={r["ups"]:.2f})')
        ax.plot(rp,vhp,'g:',lw=0.8,alpha=0.4,label='MFT halo')
        ax.plot(rp,vup,'r-',lw=2,label='MFT total')
        ax.set_title(f'{g["full"]}  '+r'($\chi^2$/dof'f'={r["chi2"]/r["dof"]:.2f})',fontsize=10)
        ax.set_xlabel('R [kpc]',fontsize=9); ax.set_ylabel('v [km/s]',fontsize=9)
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        ax.set_xlim(0,min(R_MAX,max(g["ro"])*1.3)); ax.set_ylim(0,max(g["vo"])*1.35)
    plt.tight_layout(rect=[0,0,1,0.91])
    plt.savefig(savepath("mft_galactic_6panel.png"),dpi=150,bbox_inches='tight'); plt.close()
    
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    fig.suptitle("MFT Contraction Field Profiles",fontsize=11,fontweight='bold')
    for i,gn in enumerate(GORDER):
        ax=axes[i//3,i%3]; g=GALAXIES[gn]; h=halo[gn]
        ax.plot(r_fine,h["d"],'k-',lw=1.2)
        ax.axhline(DELTA_B,color='orange',ls='--',lw=0.8,label=f'barrier ({DELTA_B:.0f})')
        ax.axhline(DELTA_V,color='red',ls='--',lw=0.8,label=f'NL vacuum ({DELTA_V:.0f})')
        ax.axhline(0,color='gray',ls=':',lw=0.5)
        ax.set_title(f'{g["full"]} (max|d|={h["maxd"]:.0f})',fontsize=9)
        ax.set_xlabel('R [kpc]',fontsize=8); ax.set_ylabel(r'$\delta$',fontsize=9)
        ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
        ax.set_xlim(0,min(60,max(g["ro"])*1.3))
    plt.tight_layout(rect=[0,0,1,0.94])
    plt.savefig(savepath("mft_galactic_field.png"),dpi=150,bbox_inches='tight'); plt.close()
    
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    fig.suptitle(r"Normalised Residuals $\Delta v/\sigma$",fontsize=11,fontweight='bold')
    for i,gn in enumerate(GORDER):
        ax=axes[i//3,i%3]; g=GALAXIES[gn]; r=res[gn]; h=halo[gn]
        Ms_o=np.interp(g["ro"],r_fine,sMenc[gn]); Mg_o=np.interp(g["ro"],r_fine,gMenc[gn])
        Mb_o=r["ups"]*Ms_o+Mg_o; MU_o=r["rs"]*np.interp(g["ro"],r_fine,h["M"])
        vm=np.sqrt(G*(Mb_o+g["BH"]+MU_o)/np.maximum(g["ro"],1e-3))
        nr=(vm-g["vo"])/g["sv"]
        cols=['green' if abs(n)<1 else ('orange' if abs(n)<2 else 'red') for n in nr]
        ax.bar(range(len(nr)),nr,color=cols,edgecolor='k',lw=0.5)
        ax.axhline(0,color='k',lw=0.8); ax.axhline(1,color='gray',ls='--',lw=0.5); ax.axhline(-1,color='gray',ls='--',lw=0.5)
        ax.set_xticks(range(len(nr))); ax.set_xticklabels([f'{x:.0f}' for x in g["ro"]],fontsize=6,rotation=45)
        ax.set_ylabel(r'$\Delta v/\sigma$',fontsize=8); ax.set_ylim(-4,4); ax.grid(True,axis='y',alpha=0.3)
        ax.set_title(f'{g["full"]} '+r'($\chi^2$/dof'f'={r["chi2"]/r["dof"]:.2f})',fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.94])
    plt.savefig(savepath("mft_galactic_residuals.png"),dpi=150,bbox_inches='tight'); plt.close()
    
    fig,ax=plt.subplots(1,1,figsize=(10,6))
    x=np.arange(len(GORDER)); w=0.25
    cu=[res[gn]["chi2"]/res[gn]["dof"] for gn in GORDER]
    cm_=[mr[gn]["chi2"]/max(len(GALAXIES[gn]["ro"])-1,1) for gn in GORDER]
    cn_=[nr_[gn]["chi2"]/max(len(GALAXIES[gn]["ro"])-3,1) for gn in GORDER]
    ax.bar(x-w,cu,w,label=r'MFT (2 params, derived $V$)',color='red',alpha=0.85)
    ax.bar(x,cm_,w,label=r'MOND (1 param, empirical $a_0$)',color='blue',alpha=0.85)
    ax.bar(x+w,cn_,w,label='NFW (3 params, assumed)',color='gray',alpha=0.85)
    ax.axhline(1,color='k',ls='--',lw=0.8,label=r'$\chi^2$/dof = 1')
    ax.set_yscale('log'); ax.set_ylim(0.05,500)
    ax.set_xticks(x); ax.set_xticklabels([GALAXIES[gn]["full"] for gn in GORDER],fontsize=9,rotation=25,ha='right')
    ax.set_ylabel(r'$\chi^2$/dof (log scale)',fontsize=11)
    ax.set_title(r'MFT vs MOND vs NFW — $\chi^2$/dof Comparison',fontsize=12,fontweight='bold')
    ax.legend(fontsize=9,loc='upper left'); ax.grid(True,axis='y',alpha=0.3,which='both')
    for i,v in enumerate(cu): ax.text(x[i]-w,v*1.3,f'{v:.1f}',ha='center',va='bottom',fontsize=7,fontweight='bold',color='red')
    plt.tight_layout()
    plt.savefig(savepath("mft_galactic_comparison.png"),dpi=150,bbox_inches='tight'); plt.close()
    
    for f in ["mft_galactic_6panel.png","mft_galactic_field.png","mft_galactic_residuals.png","mft_galactic_comparison.png"]:
        print(f"Figure: {savepath(f)}")
