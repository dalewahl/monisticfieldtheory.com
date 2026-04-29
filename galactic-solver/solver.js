/* MFT Galactic Rotation Curves Solver — JavaScript layer
 *
 * Loads Pyodide + NumPy + SciPy, then calls solver.py to:
 *  1. Solve the contraction-field BVP for the chosen galaxy at chosen β
 *  2. Fit (Υ*, ρ_scale) jointly
 *  3. Return the rotation curve, residuals, and contraction field profile
 */

let pyodide = null;
let solverReady = false;

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

        setProgress(30, 'Loading NumPy…');
        await pyodide.loadPackage(['numpy']);

        setProgress(60, 'Loading SciPy (~10 MB; cached after first load)…');
        await pyodide.loadPackage(['scipy']);

        setProgress(90, 'Loading galactic solver…');
        const solverPyResp = await fetch('solver.py');
        const solverPyText = await solverPyResp.text();
        pyodide.runPython(solverPyText);

        setProgress(100, 'Ready.');
        solverReady = true;

        setTimeout(() => {
            statusSection.style.display = 'none';
            solverUI.style.display = 'grid';
            solve();   // auto-solve at default values
        }, 400);
    } catch (err) {
        statusText.textContent = `Error loading solver: ${err.message}`;
        progressFill.style.background = '#c0392b';
        console.error(err);
    }
}

// === Plot helpers ===

function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function plotCurves(canvas, curves, options = {}) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const padding = { left: 60, right: 110, top: 25, bottom: 42 };
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;

    clearCanvas(canvas);
    if (curves.length === 0) return;

    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const c of curves) {
        for (const x of c.xs) { if (x < xmin) xmin = x; if (x > xmax) xmax = x; }
        for (const y of c.ys) {
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
        }
    }
    if (options.xmin !== undefined) xmin = options.xmin;
    if (options.xmax !== undefined) xmax = options.xmax;
    if (options.ymin !== undefined) ymin = options.ymin;
    if (options.ymax !== undefined) ymax = options.ymax;
    const yrange = ymax - ymin || 1;
    const xrange = xmax - xmin || 1;

    const xToPx = x => padding.left + ((x - xmin) / xrange) * plotW;
    const yToPx = y => padding.top + plotH - ((y - ymin) / yrange) * plotH;

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(padding.left, padding.top, plotW, plotH);

    // Subtle grid
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 0.5;
    const ticks_y = 5;
    for (let i = 0; i <= ticks_y; i++) {
        const y = ymin + (ymax - ymin) * i / ticks_y;
        const py = yToPx(y);
        ctx.beginPath();
        ctx.moveTo(padding.left, py);
        ctx.lineTo(padding.left + plotW, py);
        ctx.stroke();
    }

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
        const py = yToPx(0);
        ctx.strokeStyle = '#aaa';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, py);
        ctx.lineTo(padding.left + plotW, py);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Tick labels
    ctx.fillStyle = '#444';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    [ymin, (ymin + ymax) / 2, ymax].forEach(y => {
        const lbl = (Math.abs(y) > 1000 || Math.abs(y) < 0.01) ? y.toExponential(1) : y.toFixed(2);
        ctx.fillText(lbl, padding.left - 5, yToPx(y) + 4);
    });
    ctx.textAlign = 'center';
    [xmin, (xmin + xmax) / 2, xmax].forEach(x => {
        ctx.fillText(x.toFixed(1), xToPx(x), padding.top + plotH + 14);
    });

    if (options.xlabel) {
        ctx.font = '12px sans-serif';
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.fillText(options.xlabel, padding.left + plotW / 2, H - 10);
    }
    if (options.ylabel) {
        ctx.save();
        ctx.translate(15, padding.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.font = '12px sans-serif';
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.fillText(options.ylabel, 0, 0);
        ctx.restore();
    }

    // Curves
    for (const c of curves) {
        ctx.strokeStyle = c.color || '#2c5aa0';
        ctx.lineWidth = c.lineWidth || 2;
        if (c.dashed) ctx.setLineDash(c.dashed === true ? [5, 4] : c.dashed);
        ctx.beginPath();
        for (let i = 0; i < c.xs.length; i++) {
            const px = xToPx(c.xs[i]);
            const py = yToPx(c.ys[i]);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Error bars (for observed data)
    if (options.errorbars) {
        const eb = options.errorbars;
        for (let i = 0; i < eb.xs.length; i++) {
            const px = xToPx(eb.xs[i]);
            const yc = yToPx(eb.ys[i]);
            const yu = yToPx(eb.ys[i] + eb.sigma);
            const yd = yToPx(eb.ys[i] - eb.sigma);
            ctx.strokeStyle = eb.color || '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(px, yu);
            ctx.lineTo(px, yd);
            ctx.moveTo(px - 3, yu);
            ctx.lineTo(px + 3, yu);
            ctx.moveTo(px - 3, yd);
            ctx.lineTo(px + 3, yd);
            ctx.stroke();
            // Center dot
            ctx.fillStyle = eb.color || '#000';
            ctx.beginPath();
            ctx.arc(px, yc, 3.5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    // Horizontal reference lines
    if (options.hlines) {
        for (const m of options.hlines) {
            const py = yToPx(m.y);
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
        let yL = padding.top + 10;
        const xL = padding.left + plotW + 10;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        for (const c of curves) {
            if (!c.label) continue;
            ctx.strokeStyle = c.color;
            ctx.lineWidth = 2;
            if (c.dashed) ctx.setLineDash(c.dashed === true ? [5, 4] : c.dashed);
            ctx.beginPath();
            ctx.moveTo(xL, yL);
            ctx.lineTo(xL + 18, yL);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#333';
            ctx.fillText(c.label, xL + 22, yL + 4);
            yL += 16;
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

function chi2QualityTag(chi2_per_dof) {
    if (chi2_per_dof < 1.0) return { cls: 'chi2-tag-excellent', text: 'excellent fit' };
    if (chi2_per_dof < 1.5) return { cls: 'chi2-tag-good',      text: 'good fit' };
    if (chi2_per_dof < 3.0) return { cls: 'chi2-tag-marginal',  text: 'marginal fit' };
    return { cls: 'chi2-tag-poor', text: 'poor fit' };
}

function renderResults(result) {
    const summary = document.getElementById('results-summary');
    const fitBlock = document.getElementById('fit-results-block');
    const galaxyInfo = document.getElementById('galaxy-info');
    const residualBars = document.getElementById('residual-bars');

    if (!result.success) {
        summary.innerHTML = `<p class="error">${result.message}</p>`;
        fitBlock.innerHTML = '';
        galaxyInfo.innerHTML = '';
        residualBars.innerHTML = '';
        clearCanvas(document.getElementById('rotation-canvas'));
        clearCanvas(document.getElementById('contraction-canvas'));
        return;
    }

    const fr = result.fit_results;
    const obs = result.observed;
    const rc = result.rotation_curve;
    const cf = result.contraction_field;
    const res = result.residuals;
    const gp = result.galaxy_params;

    const tag = chi2QualityTag(fr.chi2_per_dof);

    // Summary
    summary.innerHTML = `
        <p>
            <strong>${result.galaxy_full_name}</strong> at β = ${fmtSci(result.beta, 3)}.
            ${result.fit_mode === 'auto'
              ? 'Auto-fitted (Υ*, ρ_scale).'
              : 'Manual values used.'}
        </p>
        <p class="hint">
            ${obs.r.length} observation points; assumed σ_v = ${obs.sigma_v} km/s.
            BVP solved on [0.1, 80] kpc with the silver-ratio sextic
            (m²_g = 10⁻⁶, λ_4_g² = 8 m²_g λ_6_g exactly enforced).
        </p>
    `;

    // Fit results card
    fitBlock.innerHTML = `
        <h4>Fit results</h4>
        <p class="chi2-label">χ² / dof</p>
        <p class="chi2-headline">${fmt(fr.chi2_per_dof, 2)}</p>
        <p class="chi2-quality-tag ${tag.cls}">${tag.text}</p>
        <div class="fit-row">
            <span class="fit-name">χ²</span>
            <span class="fit-val">${fmt(fr.chi2, 2)}</span>
        </div>
        <div class="fit-row">
            <span class="fit-name">dof</span>
            <span class="fit-val">${fr.dof}</span>
        </div>
        <div class="fit-row">
            <span class="fit-name">Υ* (stellar M/L)</span>
            <span class="fit-val">${fmt(fr.ups, 3)}</span>
        </div>
        <div class="fit-row">
            <span class="fit-name">ρ_scale</span>
            <span class="fit-val">10^${fmt(fr.log10_rho_scale, 2)}</span>
        </div>
    `;

    // Galaxy info
    galaxyInfo.innerHTML = `
        <h4 style="margin: 0 0 8px; font-size: 12px; color: #234a85; text-transform: uppercase; letter-spacing: 0.5px;">Galaxy parameters</h4>
        <div class="info-row"><span>Bulge mass:</span> <strong>${fmtSci(gp.M_b, 1)} M☉</strong></div>
        <div class="info-row"><span>Disk mass:</span> <strong>${fmtSci(gp.M_d, 1)} M☉</strong></div>
        <div class="info-row"><span>Disk scale R_d:</span> <strong>${gp.R_d} kpc</strong></div>
        <div class="info-row"><span>Gas mass:</span> <strong>${fmtSci(gp.M_g, 1)} M☉</strong></div>
        <div class="info-row"><span>BH mass:</span> <strong>${fmtSci(gp.BH, 1)} M☉</strong></div>
    `;

    // Plot 1: Rotation curve
    const yMaxCurve = Math.max(
        Math.max(...obs.v) + obs.sigma_v,
        Math.max(...rc.v_total)
    ) * 1.15;

    plotCurves(
        document.getElementById('rotation-canvas'),
        [
            { xs: rc.r, ys: rc.v_baryon, color: '#2c5aa0', dashed: true,
              lineWidth: 1.5, label: 'baryons' },
            { xs: rc.r, ys: rc.v_halo, color: '#386641', dashed: [3, 3],
              lineWidth: 1.5, label: 'MFT halo' },
            { xs: rc.r, ys: rc.v_total, color: '#c0392b', lineWidth: 2.5,
              label: 'MFT total' },
        ],
        {
            xlabel: 'R [kpc]',
            ylabel: 'v [km/s]',
            ymin: 0,
            ymax: yMaxCurve,
            errorbars: {
                xs: obs.r,
                ys: obs.v,
                sigma: obs.sigma_v,
                color: '#000',
            },
        }
    );

    // Plot 2: Contraction field profile
    const cfMax = Math.max(cf.delta_v_silver * 1.2, cf.max_delta * 1.05);
    plotCurves(
        document.getElementById('contraction-canvas'),
        [
            { xs: cf.r, ys: cf.delta, color: '#234a85', lineWidth: 2.2,
              label: 'δ(r)' },
        ],
        {
            xlabel: 'R [kpc]',
            ylabel: 'δ(r)',
            ymin: 0,
            ymax: cfMax,
            hlines: [
                { y: cf.delta_b_silver, color: '#d4a017',
                  label: `silver barrier (${cf.delta_b_silver.toFixed(0)})` },
                { y: cf.delta_v_silver, color: '#c0392b',
                  label: `silver vacuum (${cf.delta_v_silver.toFixed(0)})` },
            ],
        }
    );

    // Residual bars (Δv/σ)
    let residualHTML = '';
    const maxAbsRes = Math.max(...res.delta_v_over_sigma.map(Math.abs), 2.0);
    for (let i = 0; i < res.r.length; i++) {
        const dvs = res.delta_v_over_sigma[i];
        const heightPct = Math.min(50, (Math.abs(dvs) / maxAbsRes) * 50);
        const cls = dvs >= 0 ? 'positive' : 'negative';
        residualHTML += `
            <div class="residual-bar ${cls}" title="R=${res.r[i]} kpc: ${dvs.toFixed(2)}σ">
                <div class="bar-fill" style="height: ${heightPct}%"></div>
                <div class="bar-label">${res.r[i]}</div>
            </div>
        `;
    }
    residualBars.innerHTML = `
        <h3 style="margin: 14px 0 6px; font-size: 14px; color: #234a85;">Residuals (Δv / σ)</h3>
        <p style="font-size: 12px; color: #888; margin: 0 0 4px;">
            R [kpc] →
        </p>
        <div class="residuals-bar-container">${residualHTML}</div>
        <p style="font-size: 12px; color: #888; margin: 28px 0 0;">
            Bars above the centerline: MFT overpredicts. Below: underpredicts.
            Within ±1σ band: well-fit.
        </p>
    `;
}

// === Solve handler ===

async function solve() {
    if (!solverReady) return;

    const solveBtn = document.getElementById('solve-btn');
    if (solveBtn) {
        solveBtn.disabled = true;
        solveBtn.textContent = 'Solving…';
    }

    await new Promise(r => setTimeout(r, 30));

    const galaxy = document.getElementById('galaxy-select').value;
    const beta = parseFloat(document.getElementById('beta').value);

    try {
        const paramsPy = pyodide.toPy({ galaxy, beta, fit_mode: 'auto' });
        pyodide.globals.set('_params', paramsPy);
        const resultPy = pyodide.runPython(
            'solve(_params if isinstance(_params, dict) else _params.to_py())'
        );
        const result = resultPy.toJs({ dict_converter: Object.fromEntries });
        renderResults(result);
    } catch (err) {
        document.getElementById('results-summary').innerHTML = 
            `<p class="error">Solver error: ${err.message}</p>`;
        console.error(err);
    } finally {
        if (solveBtn) {
            solveBtn.disabled = false;
            solveBtn.textContent = 'Solve';
        }
    }
}

// === Wire up ===

document.addEventListener('DOMContentLoaded', () => {
    initPyodide();

    document.getElementById('solve-btn').addEventListener('click', solve);
    document.getElementById('galaxy-select').addEventListener('change', solve);

    // Update the β display as the input changes
    const betaInput = document.getElementById('beta');
    const betaDisplay = document.getElementById('beta-display-value');
    const updateBetaDisplay = () => {
        const v = parseFloat(betaInput.value);
        betaDisplay.textContent = v.toExponential(3);
    };
    betaInput.addEventListener('input', updateBetaDisplay);
    updateBetaDisplay();
});
