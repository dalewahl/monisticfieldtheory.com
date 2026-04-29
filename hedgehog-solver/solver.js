/* MFT Hedgehog / Skyrmion BVP Solver — JavaScript layer
 *
 * Solves the radial profile equation for the B=1 hedgehog Skyrmion
 * via SciPy's solve_bvp. After convergence, computes E2, E4, ε₀, λ₀,
 * the virial balance, and extracts (f_π, e) by matching M_N and
 * M_Δ - M_N. Compares to the MFT chiral-stiffness prediction
 * f_π = √δ × MFT_TO_MEV ≈ 185.94 MeV (which uses a different mechanism).
 */

let pyodide = null;
let solverReady = false;
let activePreset = 'standard';

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

        setProgress(35, 'Loading NumPy…');
        await pyodide.loadPackage(['numpy']);

        setProgress(70, 'Loading SciPy (this is the largest package, ~10 MB)…');
        await pyodide.loadPackage(['scipy']);

        setProgress(90, 'Loading MFT solver…');
        const solverPyResp = await fetch('solver.py');
        const solverPyText = await solverPyResp.text();
        pyodide.runPython(solverPyText);

        setProgress(100, 'Ready.');
        solverReady = true;

        setTimeout(() => {
            statusSection.style.display = 'none';
            solverUI.style.display = 'grid';
            solve();  // Auto-solve at default values
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

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(padding.left, padding.top, plotW, plotH);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotH);
    ctx.lineTo(padding.left + plotW, padding.top + plotH);
    ctx.stroke();

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

    // Horizontal reference lines
    if (options.hlines) {
        for (const m of options.hlines) {
            const py = yToPixel(m.y);
            if (py < padding.top || py > padding.top + plotH) continue;
            ctx.strokeStyle = m.color || '#888';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(padding.left, py);
            ctx.lineTo(padding.left + plotW, py);
            ctx.stroke();
            ctx.setLineDash([]);
            if (m.label) {
                ctx.fillStyle = m.color || '#888';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(m.label, padding.left + 4, py - 3);
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
        clearCanvas(document.getElementById('profile-canvas'));
        clearCanvas(document.getElementById('energy-density-canvas'));
        cardsEl.innerHTML = '';
        tableEl.innerHTML = '';
        return;
    }

    const pc = result.profile_check;
    const integ = result.integrals;
    const sk = result.skyrme_params;

    // === Summary ===
    const virialOK = integ.virial_imbalance < 0.05;  // <5%
    const virialBadge = virialOK
        ? '<span class="badge ok">virial balanced</span>'
        : '<span class="badge warn">virial imbalanced</span>';

    summary.innerHTML = `
        <p>
            BVP converged at rmax=${result.params_used.rmax}, R_h=${result.params_used.R_h},
            n_mesh=${result.params_used.n_mesh}. ${virialBadge}
        </p>
        <p class="hint">
            Profile: f(0) = ${fmt(pc.f_at_origin, 4)} (target π = ${fmt(pc.f_at_origin_target, 4)});
            f(rmax) = ${fmtSci(pc.f_at_infty, 2)} (target 0).
            Topological charge B = ${fmt(integ.B, 4)} (integral) = ${fmt(integ.B_topo, 4)} (boundary).
            Virial imbalance |E2-E4|/E2 = ${(integ.virial_imbalance * 100).toFixed(3)}%.
        </p>
    `;

    // === Headline cards ===
    let cardsHTML = `<div class="result-cards">`;

    // Card: ε₀ (the headline number)
    cardsHTML += `
        <div class="result-card-big">
            <p class="card-label">Total soliton energy</p>
            <p class="card-formula">ε₀ = E2 + E4 (Skyrmion units)</p>
            <p class="card-value">${fmt(integ.eps_0, 3)}</p>
            <div class="card-comparison">
                <div><span>E2:</span><strong>${fmt(integ.E2, 4)}</strong></div>
                <div><span>E4:</span><strong>${fmt(integ.E4, 4)}</strong></div>
                <div><span>E2/E4:</span><strong>${fmt(integ.e2_over_e4, 6)}</strong></div>
            </div>
            <div class="card-error">
                AKW reference value: <strong>~145.85</strong>
                ${Math.abs(integ.eps_0 - 145.85) < 0.5 ? '<span class="badge ok">matches</span>' : ''}
            </div>
        </div>
    `;

    // Card: extracted Skyrme parameters
    if (sk) {
        const fpiOK = sk.f_pi_error_vs_obs_pct < 50;
        cardsHTML += `
            <div class="result-card-big">
                <p class="card-label">Extracted Skyrme parameters</p>
                <p class="card-formula">From M_N = 938.3, M_Δ = 1232 MeV</p>
                <table class="mass-mini-table">
                    <tr><td>e:</td><td>${fmt(sk.e_extracted, 4)}</td></tr>
                    <tr><td>f_π (BVP):</td><td>${fmt(sk.f_pi_extracted_MeV, 2)} MeV</td></tr>
                    <tr><td>f_π (obs):</td><td>${fmt(sk.f_pi_observed_MeV, 1)} MeV</td></tr>
                    <tr><td>f_π (MFT):</td><td>${fmt(sk.f_pi_mft_predicted_MeV, 2)} MeV</td></tr>
                </table>
                <div class="card-error">
                    f_π BVP-extract vs observed: <strong>${sk.f_pi_error_vs_obs_pct.toFixed(1)}%</strong>
                </div>
            </div>
        `;
    }

    // Card: topological charge & virial
    cardsHTML += `
        <div class="result-card-big">
            <p class="card-label">Soliton structure</p>
            <p class="card-formula">B = topological charge, must = 1</p>
            <table class="mass-mini-table">
                <tr><td>B (integral):</td><td>${fmt(integ.B, 6)}</td></tr>
                <tr><td>B (topology):</td><td>${fmt(integ.B_topo, 6)}</td></tr>
                <tr><td>λ₀:</td><td>${fmt(integ.lambda_0, 4)}</td></tr>
                <tr><td>Virial imb.:</td><td>${(integ.virial_imbalance * 100).toFixed(4)}%</td></tr>
            </table>
            <div class="card-error">
                ${integ.virial_imbalance < 0.001 ? '<span class="badge ok">excellent virial</span>'
                  : integ.virial_imbalance < 0.05 ? '<span class="badge ok">good virial</span>'
                  : '<span class="badge warn">poor virial</span>'}
            </div>
        </div>
    `;

    cardsHTML += `</div>`;
    cardsEl.innerHTML = cardsHTML;

    // === Plot 1: hedgehog profile f(r) ===
    const prof = result.plot_profile;
    plotCurves(
        document.getElementById('profile-canvas'),
        [
            { xs: prof.r, ys: prof.f, color: '#2c5aa0', label: 'f(r)', lineWidth: 2.2 },
        ],
        {
            xlabel: 'r',
            ylabel: 'f(r)',
            ymin: -0.2,
            ymax: Math.PI + 0.3,
            hlines: [
                { y: Math.PI, color: '#c0392b', label: 'f = π' },
                { y: 0, color: '#888', label: 'f = 0' },
            ],
        }
    );

    // === Plot 2: energy density E2 and E4 ===
    const ed = result.plot_energy_density;
    plotCurves(
        document.getElementById('energy-density-canvas'),
        [
            { xs: ed.r, ys: ed.e2_density, color: '#2c5aa0', label: 'E2 density', lineWidth: 2 },
            { xs: ed.r, ys: ed.e4_density, color: '#bc6c25', label: 'E4 density', lineWidth: 2 },
        ],
        {
            xlabel: 'r',
            ylabel: 'energy density',
        }
    );

    // === Details table ===
    let tableHTML = `<h3>Detailed values</h3>`;
    tableHTML += `<table class="details-table"><tbody>`;

    tableHTML += `
        <tr class="section-header"><td colspan="2">BVP convergence</td></tr>
        <tr><td>rmin</td><td>${fmtSci(result.params_used.rmin, 1)}</td></tr>
        <tr><td>rmax</td><td>${fmt(result.params_used.rmax, 1)}</td></tr>
        <tr><td>R_h (initial guess scale)</td><td>${fmt(result.params_used.R_h, 3)}</td></tr>
        <tr><td>n_mesh</td><td>${result.params_used.n_mesh}</td></tr>
        <tr><td>tolerance</td><td>${fmtSci(result.params_used.tol, 1)}</td></tr>
        <tr><td>f(0)</td><td>${fmt(pc.f_at_origin, 8)} (target ${fmt(pc.f_at_origin_target, 6)})</td></tr>
        <tr><td>f(rmax)</td><td>${fmtSci(pc.f_at_infty, 2)} (target 0)</td></tr>
        <tr><td>monotonic</td><td>${pc.is_monotonic}</td></tr>
    `;

    tableHTML += `
        <tr class="section-header"><td colspan="2">Energy integrals</td></tr>
        <tr><td>E2 (quadratic)</td><td>${fmt(integ.E2, 6)}</td></tr>
        <tr><td>E4 (quartic)</td><td>${fmt(integ.E4, 6)}</td></tr>
        <tr><td>E2/E4</td><td>${fmt(integ.e2_over_e4, 8)} (target 1.000000)</td></tr>
        <tr><td>Virial imbalance |E2-E4|/E2</td><td>${(integ.virial_imbalance * 100).toFixed(6)}%</td></tr>
        <tr><td>ε₀ = E2 + E4</td><td>${fmt(integ.eps_0, 6)}</td></tr>
        <tr><td>λ₀ (moment of inertia)</td><td>${fmt(integ.lambda_0, 6)}</td></tr>
    `;

    tableHTML += `
        <tr class="section-header"><td colspan="2">Topological charge</td></tr>
        <tr><td>B = -(2/π) ∫ sin²f f' dr</td><td>${fmt(integ.B, 8)}</td></tr>
        <tr><td>B = (f(0) - f(∞))/π</td><td>${fmt(integ.B_topo, 8)}</td></tr>
        <tr><td>Difference</td><td>${fmtSci(Math.abs(integ.B - integ.B_topo), 2)}</td></tr>
    `;

    if (sk) {
        tableHTML += `
            <tr class="section-header"><td colspan="2">Skyrme parameters extracted from BVP</td></tr>
            <tr><td>Λ (moment of inertia, MeV⁻¹)</td><td>${fmtSci(sk.Lambda_MeV_inv, 4)}</td></tr>
            <tr><td>M_core (MeV)</td><td>${fmt(sk.M_core_MeV, 2)}</td></tr>
            <tr><td>e</td><td>${fmt(sk.e_extracted, 4)}</td></tr>
            <tr><td>e candidate (2δ)</td><td>${fmt(sk.e_candidate_2delta, 4)} (diff ${sk.e_diff_vs_2delta_pct.toFixed(2)}%)</td></tr>
            <tr><td>f_π (BVP-extracted)</td><td>${fmt(sk.f_pi_extracted_MeV, 2)} MeV</td></tr>
            <tr><td>f_π (observed)</td><td>${fmt(sk.f_pi_observed_MeV, 1)} MeV (BVP error: ${sk.f_pi_error_vs_obs_pct.toFixed(2)}%)</td></tr>
            <tr><td>f_π (MFT prediction = √δ × MFT_TO_MEV)</td><td>${fmt(sk.f_pi_mft_predicted_MeV, 2)} MeV</td></tr>
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
        solveBtn.textContent = 'Solving BVP…';
    }

    await new Promise(r => setTimeout(r, 30));

    const rmax = parseFloat(document.getElementById('rmax').value);
    const R_h = parseFloat(document.getElementById('R_h').value);
    const n_mesh = parseInt(document.getElementById('n_mesh').value, 10);

    try {
        const paramsPy = pyodide.toPy({ rmax, R_h, n_mesh });
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
            solveBtn.textContent = 'Solve BVP';
        }
    }
}

// === Preset handler ===

function applyPreset(name) {
    if (!solverReady) return;
    
    const presetPy = pyodide.runPython(`get_preset('${name}')`);
    const preset = presetPy.toJs({ dict_converter: Object.fromEntries });

    document.getElementById('rmax').value = preset.rmax;
    document.getElementById('R_h').value = preset.R_h;
    document.getElementById('n_mesh').value = preset.n_mesh;

    document.getElementById('preset-description').textContent = preset.description;
    activePreset = name;

    document.querySelectorAll('.preset').forEach(b => {
        b.classList.toggle('active', b.dataset.preset === name);
    });

    solve();
}

// === Wire up event handlers ===

document.addEventListener('DOMContentLoaded', () => {
    initPyodide();

    document.getElementById('solve-btn').addEventListener('click', solve);

    document.querySelectorAll('.preset').forEach(btn => {
        btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
    });

    setTimeout(() => {
        document.querySelector('.preset[data-preset="standard"]')?.classList.add('active');
    }, 200);
});
