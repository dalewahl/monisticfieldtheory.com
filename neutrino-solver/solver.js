/* MFT Neutrino Hierarchy Solver — JavaScript layer
 *
 * Visualises the δ⁴ - 1 hierarchy prediction and the absolute neutrino
 * masses from the MFT one-loop self-energy formula. Lets the user vary
 * (m², λ₄, λ₆, β) and watch the silver-ratio condition gate the
 * hierarchy ratio and absolute masses.
 */

let pyodide = null;
let solverReady = false;
let activePreset = 'silver_ratio_canonical';

const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const statusSection = document.getElementById('status');
const solverUI = document.getElementById('solver-ui');

function setProgress(pct, msg) {
    progressFill.style.width = `${pct}%`;
    if (msg) statusText.textContent = msg;
}

async function initPyodide() {
    try {
        setProgress(10, 'Loading Pyodide runtime…');
        pyodide = await loadPyodide();

        setProgress(50, 'Loading NumPy…');
        await pyodide.loadPackage(['numpy']);

        setProgress(80, 'Loading MFT solver…');
        const solverPyResp = await fetch('solver.py');
        const solverPyText = await solverPyResp.text();
        pyodide.runPython(solverPyText);

        setProgress(100, 'Ready.');
        solverReady = true;

        setTimeout(() => {
            statusSection.style.display = 'none';
            solverUI.style.display = 'grid';
            // Auto-solve at default values to populate the page
            solve();
        }, 400);
    } catch (err) {
        statusText.textContent = `Error loading solver: ${err.message}`;
        progressFill.style.background = '#c0392b';
        console.error(err);
    }
}

// === Plotting helpers ===

function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function plotCurves(canvas, curves, options = {}) {
    /* curves: array of { xs, ys, color, label, lineWidth, dashed } */
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    const padding = { left: 60, right: 110, top: 25, bottom: 42 };
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;

    clearCanvas(canvas);

    if (curves.length === 0) {
        ctx.fillStyle = '#888';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data', W / 2, H / 2);
        return;
    }

    // Auto-range across all curves
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const c of curves) {
        for (const x of c.xs) { if (x < xmin) xmin = x; if (x > xmax) xmax = x; }
        for (const y of c.ys) { if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
    }
    if (options.xmin !== undefined) xmin = options.xmin;
    if (options.xmax !== undefined) xmax = options.xmax;
    if (options.ymin !== undefined) ymin = options.ymin;
    if (options.ymax !== undefined) ymax = options.ymax;
    const yrange = ymax - ymin || 1;
    const xrange = xmax - xmin || 1;

    const xToPixel = x => padding.left + ((x - xmin) / xrange) * plotW;
    const yToPixel = y => padding.top + plotH - ((y - ymin) / yrange) * plotH;

    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(padding.left, padding.top, plotW, plotH);

    // Axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotH);
    ctx.lineTo(padding.left + plotW, padding.top + plotH);
    ctx.stroke();

    // Zero line
    if (ymin < 0 && ymax > 0) {
        const yZero = yToPixel(0);
        ctx.strokeStyle = '#aaa';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, yZero);
        ctx.lineTo(padding.left + plotW, yZero);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Tick labels
    ctx.fillStyle = '#444';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(ymax.toFixed(2), padding.left - 5, padding.top + 4);
    ctx.fillText(((ymax + ymin) / 2).toFixed(2), padding.left - 5, padding.top + plotH / 2 + 4);
    ctx.fillText(ymin.toFixed(2), padding.left - 5, padding.top + plotH);
    ctx.textAlign = 'center';
    ctx.fillText(xmin.toFixed(1), padding.left, padding.top + plotH + 14);
    ctx.fillText(((xmax + xmin) / 2).toFixed(1), padding.left + plotW / 2, padding.top + plotH + 14);
    ctx.fillText(xmax.toFixed(1), padding.left + plotW, padding.top + plotH + 14);

    if (options.xlabel) {
        ctx.textAlign = 'center';
        ctx.font = '12px sans-serif';
        ctx.fillText(options.xlabel, padding.left + plotW / 2, H - 10);
    }
    if (options.ylabel) {
        ctx.save();
        ctx.translate(15, padding.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText(options.ylabel, 0, 0);
        ctx.restore();
    }

    // Curves
    for (const c of curves) {
        ctx.strokeStyle = c.color || '#2c5aa0';
        ctx.lineWidth = c.lineWidth || 2;
        if (c.dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (let i = 0; i < c.xs.length; i++) {
            const px = xToPixel(c.xs[i]);
            const py = yToPixel(c.ys[i]);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Vertical lines and point markers
    if (options.vlines) {
        for (const m of options.vlines) {
            const px = xToPixel(m.x);
            if (px < padding.left || px > padding.left + plotW) continue;
            ctx.strokeStyle = m.color || '#888';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(px, padding.top);
            ctx.lineTo(px, padding.top + plotH);
            ctx.stroke();
            ctx.setLineDash([]);
            if (m.label) {
                ctx.fillStyle = m.color || '#888';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(m.label, px + 3, padding.top + 12);
            }
        }
    }
    if (options.points) {
        for (const p of options.points) {
            const px = xToPixel(p.x);
            const py = yToPixel(p.y);
            ctx.fillStyle = p.color || '#c0392b';
            ctx.beginPath();
            ctx.arc(px, py, p.r || 5, 0, 2 * Math.PI);
            ctx.fill();
            if (p.label) {
                ctx.fillStyle = '#222';
                ctx.font = 'bold 11px sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(p.label, px + 8, py - 4);
            }
        }
    }

    // Legend
    if (curves.some(c => c.label)) {
        let yLegend = padding.top + 10;
        const xLegend = padding.left + plotW + 10;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        for (const c of curves) {
            if (!c.label) continue;
            ctx.strokeStyle = c.color;
            ctx.lineWidth = 2;
            if (c.dashed) ctx.setLineDash([5, 4]);
            ctx.beginPath();
            ctx.moveTo(xLegend, yLegend);
            ctx.lineTo(xLegend + 18, yLegend);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#333';
            ctx.fillText(c.label, xLegend + 22, yLegend + 4);
            yLegend += 16;
        }
    }
}

// === Render results ===

const NU_COLORS = ['#27ae60', '#2c5aa0', '#c0392b'];  // ν₁, ν₂, ν₃ — green, blue, red

function fmt(n, dig) {
    if (n === null || n === undefined || !isFinite(n)) return '—';
    return Number(n).toFixed(dig === undefined ? 4 : dig);
}

function fmtSci(n, dig) {
    if (n === null || n === undefined || !isFinite(n)) return '—';
    return Number(n).toExponential(dig === undefined ? 3 : dig);
}

function renderResults(result) {
    const summary = document.getElementById('results-summary');
    const cardsEl = document.getElementById('cards-container');
    const tableEl = document.getElementById('details-table-container');

    if (!result.success) {
        summary.innerHTML = `<p class="error">${result.message}</p>`;
        clearCanvas(document.getElementById('potential-canvas'));
        clearCanvas(document.getElementById('sensitivity-canvas'));
        cardsEl.innerHTML = '';
        tableEl.innerHTML = '';
        return;
    }

    const sr = result.silver_ratio;
    const cp = result.critical_points;
    const Vc = result.V_at_crit;
    const Vpp = result.Vpp_at_crit;
    const h = result.hierarchy_ratio;
    const m = result.masses;

    // === Summary block ===
    const srBadge = sr.satisfied
        ? '<span class="badge ok">silver-ratio satisfied</span>'
        : `<span class="badge warn">silver-ratio violated (ρ=${sr.rho.toFixed(3)} ≠ 8.000)</span>`;

    const errorPct = (h.predicted !== null && h.predicted !== undefined)
        ? (Math.abs(h.predicted - h.observed) / h.observed * 100).toFixed(2)
        : null;

    summary.innerHTML = `
        <p>
            Computed at ρ = λ₄²/(m²λ₆) = <strong>${sr.rho.toFixed(4)}</strong>. ${srBadge}
        </p>
        <p class="hint">
            Critical points: φ_b = ${fmt(cp.phi_b, 4)}, φ_v = ${fmt(cp.phi_v, 4)}.
            Field-space ratio φ_v/φ_b = ${fmt(cp.field_ratio, 4)}
            (silver-ratio target: δ = ${fmt(cp.field_ratio_silver, 4)}).
            Universal screening mass M²_s = V''(φ_v) + V''(0) = ${fmt(Vpp.M2_screen, 4)}.
        </p>
    `;

    // === Headline cards ===
    let cardsHTML = `<div class="result-cards">`;

    // Card: hierarchy ratio
    const ratioOK = errorPct !== null && parseFloat(errorPct) < 5;
    cardsHTML += `
        <div class="result-card-big">
            <p class="card-label">Hierarchy ratio</p>
            <p class="card-formula">Δm²₃₂ / Δm²₂₁</p>
            <p class="card-value">${fmt(h.predicted, 3)}</p>
            <div class="card-comparison">
                <div><span>Predicted:</span><strong>${fmt(h.predicted, 3)}</strong></div>
                <div><span>Silver value δ⁴-1:</span><strong>${fmt(h.silver_value, 3)}</strong></div>
                <div><span>Observed (PDG):</span><strong>${fmt(h.observed, 2)}</strong></div>
            </div>
            <div class="card-error">
                Error vs observed: <strong>${errorPct !== null ? errorPct + '%' : '—'}</strong>
                ${ratioOK ? '<span class="badge ok">good</span>' : '<span class="badge warn">large</span>'}
            </div>
        </div>
    `;

    // Card: absolute masses
    if (m) {
        const sumMev = (m.sum_eV * 1000).toFixed(2);
        const planckOK = m.sum_eV < 0.12;
        cardsHTML += `
            <div class="result-card-big">
                <p class="card-label">Absolute neutrino masses</p>
                <p class="card-formula">m_νᵢ = 3β² |V(φᵢ)| × √I × (m_e/E_e)</p>
                <table class="mass-mini-table">
                    <tr style="color:${NU_COLORS[0]}"><td>m_ν₁:</td><td>${(m.m_nu1_eV * 1000).toFixed(3)} meV</td></tr>
                    <tr style="color:${NU_COLORS[1]}"><td>m_ν₂:</td><td>${(m.m_nu2_eV * 1000).toFixed(3)} meV</td></tr>
                    <tr style="color:${NU_COLORS[2]}"><td>m_ν₃:</td><td>${(m.m_nu3_eV * 1000).toFixed(3)} meV</td></tr>
                </table>
                <div class="card-error">
                    Σm_ν = <strong>${sumMev} meV</strong>
                    ${planckOK ? '<span class="badge ok">below Planck (120 meV)</span>'
                              : '<span class="badge warn">exceeds Planck (120 meV)</span>'}
                </div>
            </div>
        `;
    }

    cardsHTML += `</div>`;
    cardsEl.innerHTML = cardsHTML;

    // === Plot 1: the sextic potential with three critical points ===
    const phiArr = result.potential_curve.phi;
    const VArr = result.potential_curve.V;

    plotCurves(
        document.getElementById('potential-canvas'),
        [
            { xs: phiArr, ys: VArr, color: '#222', label: 'V(φ)', lineWidth: 2 },
        ],
        {
            xlabel: 'φ',
            ylabel: 'V(φ)',
            points: [
                { x: 0, y: Vc.V_0, color: NU_COLORS[0], r: 6, label: 'ν₁' },
                { x: cp.phi_b, y: Vc.V_b, color: NU_COLORS[1], r: 6, label: 'ν₂' },
                { x: cp.phi_v, y: Vc.V_v, color: NU_COLORS[2], r: 6, label: 'ν₃' },
            ],
        }
    );

    // === Plot 2: hierarchy ratio vs ρ ===
    const sc = result.sensitivity_scan;
    const observedHline_xs = [Math.min(...sc.rhos), Math.max(...sc.rhos)];
    const observedHline_ys = [h.observed, h.observed];

    plotCurves(
        document.getElementById('sensitivity-canvas'),
        [
            { xs: sc.rhos, ys: sc.ratios, color: '#2c5aa0', label: 'predicted ratio', lineWidth: 2.5 },
            { xs: observedHline_xs, ys: observedHline_ys, color: '#c0392b', label: 'observed (32.58)', dashed: true, lineWidth: 1.5 },
        ],
        {
            xlabel: 'ρ = λ₄² / (m² λ₆)',
            ylabel: 'Δm²₃₂ / Δm²₂₁',
            ymin: 0,
            ymax: Math.max(50, sc.ratio_silver * 1.2),
            vlines: [
                { x: 8.0, color: '#2c5aa0', label: 'silver (ρ=8)' },
                { x: sr.rho, color: '#386641', label: `current` },
            ],
            points: [
                { x: 8.0, y: sc.ratio_silver, color: '#2c5aa0', r: 6 },
            ],
        }
    );

    // === Details table ===
    let tableHTML = `<h3>Detailed values</h3>`;
    tableHTML += `<table class="details-table"><tbody>`;

    tableHTML += `
        <tr class="section-header"><td colspan="2">Silver-ratio diagnostic</td></tr>
        <tr><td>λ₄²</td><td>${fmt(sr.lam4_squared, 4)}</td></tr>
        <tr><td>8 m² λ₆ (target)</td><td>${fmt(sr.eight_m2_lam6, 4)}</td></tr>
        <tr><td>ρ = λ₄² / (m² λ₆)</td><td>${fmt(sr.rho, 4)} (silver-ratio: 8.0000)</td></tr>
        <tr><td>Relative error</td><td>${(sr.relative_error * 100).toFixed(3)}%</td></tr>
    `;

    tableHTML += `
        <tr class="section-header"><td colspan="2">Critical points</td></tr>
        <tr><td>φ_b (barrier)</td><td>${fmt(cp.phi_b, 6)}</td></tr>
        <tr><td>φ_v (nonlinear vacuum)</td><td>${fmt(cp.phi_v, 6)}</td></tr>
        <tr><td>φ_v / φ_b</td><td>${fmt(cp.field_ratio, 6)} (silver-ratio: ${fmt(cp.field_ratio_silver, 6)})</td></tr>
    `;

    tableHTML += `
        <tr class="section-header"><td colspan="2">Potential values V(φᵢ)</td></tr>
        <tr><td>V(0)</td><td>${fmt(Vc.V_0, 6)} (always 0)</td></tr>
        <tr><td>V(φ_b)</td><td>${fmt(Vc.V_b, 6)} (silver: 1/3δ ≈ 0.138071)</td></tr>
        <tr><td>V(φ_v)</td><td>${fmt(Vc.V_v, 6)} (silver: -δ/3 ≈ -0.804738)</td></tr>
    `;

    tableHTML += `
        <tr class="section-header"><td colspan="2">Curvatures V''(φᵢ)</td></tr>
        <tr><td>V''(0) = m²</td><td>${fmt(Vpp.Vpp_0, 4)}</td></tr>
        <tr><td>V''(φ_b)</td><td>${fmt(Vpp.Vpp_b, 4)} (silver: -4/δ ≈ -1.6569)</td></tr>
        <tr><td>V''(φ_v)</td><td>${fmt(Vpp.Vpp_v, 4)} (silver: 4δ ≈ 9.6569)</td></tr>
        <tr><td>M²_screen = V''(φ_v) + V''(0)</td><td>${fmt(Vpp.M2_screen, 4)} (silver: δ(δ+2) ≈ 10.6569)</td></tr>
    `;

    if (m) {
        tableHTML += `
            <tr class="section-header"><td colspan="2">Absolute neutrino masses (β = ${fmtSci(m.beta, 3)})</td></tr>
            <tr><td>m_ν₁</td><td><span style="color:${NU_COLORS[0]}">${(m.m_nu1_eV*1000).toFixed(4)} meV</span></td></tr>
            <tr><td>m_ν₂</td><td><span style="color:${NU_COLORS[1]}">${(m.m_nu2_eV*1000).toFixed(4)} meV</span></td></tr>
            <tr><td>m_ν₃</td><td><span style="color:${NU_COLORS[2]}">${(m.m_nu3_eV*1000).toFixed(4)} meV</span></td></tr>
            <tr><td>Σm_ν</td><td>${(m.sum_eV*1000).toFixed(4)} meV (Planck bound: &lt; 120 meV)</td></tr>
            <tr><td>Δm²₂₁</td><td>${fmtSci(m.dm2_21_eV2, 3)} eV² (obs: 7.53e-5)</td></tr>
            <tr><td>Δm²₃₂</td><td>${fmtSci(m.dm2_32_eV2, 3)} eV² (obs: 2.453e-3)</td></tr>
            <tr><td>One-loop integral I</td><td>${fmtSci(m.I_loop, 4)}</td></tr>
        `;
    }

    tableHTML += `</tbody></table>`;
    tableEl.innerHTML = tableHTML;
}

// === Solve handler ===

async function solve() {
    if (!solverReady) return;

    const solveBtn = document.getElementById('solve-btn');
    if (solveBtn) {
        solveBtn.disabled = true;
        solveBtn.textContent = 'Computing…';
    }

    // Yield to let UI update
    await new Promise(r => setTimeout(r, 30));

    const m2 = parseFloat(document.getElementById('m2').value);
    const lam4 = parseFloat(document.getElementById('lam4').value);
    const lam6 = parseFloat(document.getElementById('lam6').value);
    const beta = parseFloat(document.getElementById('beta').value);

    try {
        // Set the params into Python globals, call solve, return JS dict
        const paramsPy = pyodide.toPy({ m2, lam4, lam6, beta });
        pyodide.globals.set('_params', paramsPy);
        const resultPy = pyodide.runPython('solve(_params if isinstance(_params, dict) else _params.to_py())');
        const result = resultPy.toJs({ dict_converter: Object.fromEntries });
        renderResults(result);
    } catch (err) {
        document.getElementById('results-summary').innerHTML = 
            `<p class="error">Solver error: ${err.message}</p>`;
        console.error(err);
    } finally {
        if (solveBtn) {
            solveBtn.disabled = false;
            solveBtn.textContent = 'Compute';
        }
    }
}

// === Preset handler ===

function applyPreset(name) {
    if (!solverReady) return;
    
    const presetPy = pyodide.runPython(`get_preset('${name}')`);
    const preset = presetPy.toJs({ dict_converter: Object.fromEntries });

    document.getElementById('m2').value = preset.m2;
    document.getElementById('lam4').value = preset.lam4;
    document.getElementById('lam6').value = preset.lam6;
    document.getElementById('beta').value = preset.beta;

    document.getElementById('preset-description').textContent = preset.description;
    activePreset = name;

    // Update active class
    document.querySelectorAll('.preset').forEach(b => {
        b.classList.toggle('active', b.dataset.preset === name);
    });

    // Auto-solve on preset change
    solve();
}

// === Wire up event handlers ===

document.addEventListener('DOMContentLoaded', () => {
    initPyodide();

    document.getElementById('solve-btn').addEventListener('click', solve);

    document.querySelectorAll('.preset').forEach(btn => {
        btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
    });

    // Auto-mark canonical preset as active on load
    setTimeout(() => {
        document.querySelector('.preset[data-preset="silver_ratio_canonical"]')?.classList.add('active');
    }, 200);
});
