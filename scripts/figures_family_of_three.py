#!/usr/bin/env python3
"""
FIGURES FOR: A Family-of-Three Stability Theorem for Confined Solitons
              under Charge and Dilation Constraints
Reads theorem_results.json and generates publication figures.

Outputs (saved alongside this script):
  fig_f3_profiles.png   — Soliton profiles u(r) for n=0,1,2,3 with node structure
  fig_f3_morse.png      — Morse index diagram: unconstrained → constrained → physical

Author: Dale Wahl, March 2026
"""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def savepath(f): return os.path.join(SCRIPT_DIR, f)

# ═══ Load JSON results ═══
json_path = os.path.join(SCRIPT_DIR, "theorem_results.json")
if not os.path.exists(json_path):
    # Try parent directory or project directory, including legacy filenames
    for alt in ["../theorem_results.json",
                "theorem_results.json",
                "family3_theorem_results_presentation.json",
                "../family3_theorem_results_presentation.json"]:
        if os.path.exists(alt):
            json_path = alt; break

with open(json_path) as f:
    data = json.load(f)

run = data[0]
params = run["params"]
states = run["states"]
Rmax = params["Rmax"]
N = params["N"]

# Reconstruct radial grid
rin = Rmax / (N * 300.0)
r = np.linspace(rin, Rmax, N)

# Extract eigenvalues and profiles
energies = {}
profiles = {}
for key in ["0", "1", "2", "3"]:
    if key in states:
        energies[int(key)] = states[key]["E"]
        profiles[int(key)] = np.array(states[key]["u"])

E0, E1, E2, E3 = energies[0], energies[1], energies[2], energies[3]

print(f"=== Family-of-Three Theorem: Numerical Results ===")
print(f"  E0 = {E0:.10f}  (n=0)")
print(f"  E1 = {E1:.10f}  (n=1)")
print(f"  E2 = {E2:.10f}  (n=2)")
print(f"  E3 = {E3:.10f}  (n=3)")
print(f"  E1/E0 = {E1/E0:.2f}")
print(f"  E2/E1 = {E2/E1:.2f}")
print()

# ═══ Figure 1: Soliton profiles ═══
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Family-of-Three: Radial Soliton Modes $u_n(r)$\n"
             "Node count = mode number $n$; higher modes increasingly unstable",
             fontsize=12, fontweight='bold')

colors = ['#2166ac', '#4393c3', '#d6604d', '#b2182b']
labels = ['$n=0$: 0 nodes — stable',
          '$n=1$: 1 node — stable',
          '$n=2$: 2 nodes — metastable',
          '$n=3$: 3 nodes — multiply unstable']
titles = [f'$n=0$: $E_0 = {E0:.6f}$',
          f'$n=1$: $E_1 = {E1:.6f}$',
          f'$n=2$: $E_2 = {E2:.6f}$',
          f'$n=3$: $E_3 = {E3:.6f}$']

for idx, ax in enumerate(axes.flat):
    if idx not in profiles:
        continue
    u = profiles[idx]
    # Normalise for display
    umax = np.max(np.abs(u))
    if umax > 0:
        u_plot = u / umax
    else:
        u_plot = u
    
    ax.plot(r, u_plot, color=colors[idx], lw=2)
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('$r$', fontsize=11)
    ax.set_ylabel('$u_n(r)$ (normalised)', fontsize=11)
    ax.set_title(titles[idx], fontsize=10)
    ax.set_xlim(0, Rmax * 0.8)
    ax.grid(True, alpha=0.3)
    
    # Mark nodes
    u_trimmed = u_plot[:int(0.95*len(u_plot))]
    sign_changes = np.where(np.diff(np.sign(u_trimmed)))[0]
    for sc in sign_changes:
        ax.axvline(r[sc], color='red', ls='--', lw=1, alpha=0.5)
    ax.text(0.98, 0.95, f'{len(sign_changes)} node{"s" if len(sign_changes)!=1 else ""}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # Stability annotation
    if idx == 0:
        stab = "STABLE ($m_{\\mathrm{phys}}=0$)"
        col = 'darkgreen'
    elif idx == 1:
        stab = "STABLE ($m_{\\mathrm{phys}}=0$)"
        col = 'darkgreen'
    elif idx == 2:
        stab = "METASTABLE ($m_{\\mathrm{phys}}=1$)"
        col = 'darkorange'
    else:
        stab = "EXCLUDED ($m_{\\mathrm{phys}}=2$)"
        col = 'red'
    ax.text(0.98, 0.82, stab, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color=col, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(savepath("fig_f3_profiles.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_f3_profiles.png")

# ═══ Figure 2: Morse index diagram ═══
fig, ax = plt.subplots(figsize=(10, 5))

modes = [0, 1, 2, 3, 4]
m_un = [0, 1, 2, 3, 4]       # unconstrained = n
m_Q  = [0, 0, 1, 2, 3]       # charge-constrained = max(0, n-1)
m_phys = [0, 0, 1, 2, 3]     # physical = m_Q (dilation doesn't reduce further)

x = np.arange(len(modes))
w = 0.25

bars1 = ax.bar(x - w, m_un, w, label='Unconstrained $m_{\\mathrm{un}} = n$',
               color='#4393c3', alpha=0.8)
bars2 = ax.bar(x, m_Q, w, label='Charge-constrained $m_Q = \\max(0, n-1)$',
               color='#f4a582', alpha=0.8)
bars3 = ax.bar(x + w, m_phys, w, label='Physical $m_{\\mathrm{phys}} = m_Q$',
               color='#d6604d', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([f'$n={n}$' for n in modes], fontsize=10)
ax.set_ylabel('Morse index (number of negative directions)', fontsize=11)
ax.set_title('Family-of-Three Stability Theorem: Morse Index Reduction\n'
             'Charge constraint removes exactly one negative direction; '
             'dilation does not reduce further',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Annotate stability regions
ax.axhspan(-0.1, 0.5, color='green', alpha=0.08)
ax.axhspan(0.5, 1.5, color='orange', alpha=0.08)
ax.axhspan(1.5, 4.5, color='red', alpha=0.08)
ax.text(0.5, 0.25, 'STABLE', ha='center', fontsize=9, color='darkgreen', fontweight='bold')
ax.text(2.0, 0.75, 'METASTABLE', ha='center', fontsize=9, color='darkorange', fontweight='bold')
ax.text(3.5, 2.8, 'EXCLUDED', ha='center', fontsize=9, color='red', fontweight='bold')

ax.set_ylim(-0.3, 4.5)
plt.tight_layout()
plt.savefig(savepath("fig_f3_morse.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_f3_morse.png")

print("All Family-of-Three figures generated.")
