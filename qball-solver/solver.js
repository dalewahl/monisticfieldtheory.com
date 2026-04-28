/* MFT Q-Ball Lepton Spectrum Solver — JavaScript layer (v2)
 *
 * Scan ω² internally, return the full discrete tower, highlight the
 * canonical lepton triple. ω² is never an input.
 */

let pyodide = null;
let solverReady = false;
let activeSector = 'lepton_sector';  // tracks which preset is active

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

        setProgress(40, 'Loading NumPy and SciPy…');
        await pyodide.loadPackage(['numpy', 'scipy']);

        setProgress(80, 'Loading MFT solver…');
        const solverPyResp = await fetch('solver.py');
        const solverPyText = await solverPyResp.text();
        pyodide.runPython(solverPyText);

        setProgress(100, 'Ready.');
        solverReady = true;

        setTimeout(() => {
            statusSection.style.display = 'none';
            solverUI.style.display = 'grid';
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
    /* curves: array of { xs, ys, color, label } */
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    const padding = { left: 50, right: 100, top: 20, bottom: 35 };
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
    ctx.fillText(ymin.toFixed(2), padding.left - 5, padding.top + plotH);
    ctx.textAlign = 'center';
    ctx.fillText(xmin.toFixed(1), padding.left, padding.top + plotH + 14);
    ctx.fillText(xmax.toFixed(1), padding.left + plotW, padding.top + plotH + 14);

    if (options.xlabel) {
        ctx.textAlign = 'center';
        ctx.font = '12px sans-serif';
        ctx.fillText(options.xlabel, padding.left + plotW / 2, H - 8);
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
        ctx.beginPath();
        for (let i = 0; i < c.xs.length; i++) {
            const px = xToPixel(c.xs[i]);
            const py = yToPixel(c.ys[i]);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    // Markers
    if (options.markers) {
        for (const m of options.markers) {
            const px = xToPixel(m.x);
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
            ctx.beginPath();
            ctx.moveTo(xLegend, yLegend);
            ctx.lineTo(xLegend + 18, yLegend);
            ctx.stroke();
            ctx.fillStyle = '#333';
            ctx.fillText(c.label, xLegend + 22, yLegend + 4);
            yLegend += 16;
        }
    }
}

// === Render results ===

// Color palette by particle position in the triple (slot 0/1/2)
const SLOT_COLORS = ['#27ae60', '#2c5aa0', '#c0392b'];

function renderResults(result) {
    const summary = document.getElementById('results-summary');
    const tripleContainer = document.getElementById('lepton-triple-container');
    const tableContainer = document.getElementById('spectrum-table-container');

    if (!result.success) {
        summary.innerHTML = `<p class="error">${result.message}</p>`;
        clearCanvas(document.getElementById('potential-canvas'));
        clearCanvas(document.getElementById('profile-canvas'));
        tableContainer.innerHTML = '';
        tripleContainer.innerHTML = '';
        return;
    }

    const sectorInfo = result.sector_info;
    const triple = result.triple;
    const errorPct = (model, observed) =>
        observed === 0 ? 'n/a' : (100 * Math.abs(model - observed) / observed).toFixed(2);

    // === Summary ===
    const sr = result.silver_ratio;
    const srBadge = sr.satisfied
        ? '<span class="badge ok">silver-ratio satisfied</span>'
        : `<span class="badge warn">silver-ratio violated (λ₄²=${sr.lam4_squared.toFixed(3)}, 8m₂λ₆=${sr.eight_m2_lam6.toFixed(3)})</span>`;

    summary.innerHTML = `
        <p>
            <strong>${result.spectrum.length}</strong> distinct solitons found in the
            <strong>${sectorInfo.name}</strong> sector. ${srBadge}
        </p>
        <p class="hint">
            <strong>Z = ${sectorInfo.Z}</strong> (${sectorInfo.Z_origin}).
            ℓ = ${sectorInfo.ell}${result.sector === 'boson_sector' ? ' for vector solitons (W, Z); ℓ=0 for scalar (Higgs)' : ''}.
            Energy scale calibrated by ${sectorInfo.anchor_label} = ${sectorInfo.anchor_mass.toLocaleString()} MeV
            (1 MFT unit ≈ ${result.mft_to_mev.toFixed(2)} MeV).
            φ_barrier = ${result.phi_barrier.toFixed(4)}, φ_vacuum = ${result.phi_vacuum.toFixed(4)}.
        </p>
    `;

    // === Triple table ===
    if (triple) {
        const idxs = [triple.idx0, triple.idx1, triple.idx2];
        const rows = idxs.map((idx, slot) => {
            const s = result.spectrum[idx];
            const name = triple.particles[slot];
            const observed = triple.observed_masses[slot];
            const predicted = triple.predicted_masses[slot];
            const isAnchor = slot === sectorInfo.anchor_idx;
            const errorCell = isAnchor
                ? '<em>calibration</em>'
                : `${errorPct(predicted, observed)}%`;
            const color = SLOT_COLORS[slot];
            return `
                <tr class="lepton-row" style="background: ${color}15">
                    <td><span class="dot" style="background:${color}"></span> ${name}</td>
                    <td>n=${s.f3_mode}</td>
                    <td>${s.morse_index} <span class="stab-tag stab-${s.stability}">${s.stability}</span></td>
                    <td>${s.E.toFixed(5)}</td>
                    <td>${predicted.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                    <td>${observed.toLocaleString()}</td>
                    <td>${errorCell}</td>
                    <td>${s.omega2.toFixed(4)}</td>
                    <td>ℓ=${s.ell}</td>
                    <td>${s.regime}</td>
                </tr>
            `;
        }).join('');

        const ratiosLabels = ['m₂/m₁', 'm₃/m₂', 'm₃/m₁'];
        const ratioRows = ratiosLabels.map((label, k) => {
            const m = triple.mass_ratios_model[k];
            const o = triple.mass_ratios_observed[k];
            return `<tr><td>${label}</td><td>${m.toFixed(4)}</td><td>${o.toFixed(4)}</td><td>${errorPct(m, o)}%</td></tr>`;
        }).join('');

        let weinbergRow = '';
        if (result.weinberg) {
            const w = result.weinberg;
            weinbergRow = `
                <h4 style="margin-top: 16px;">Weinberg angle (derived, not fitted)</h4>
                <table class="lepton-ratios-table">
                    <thead><tr><th>Quantity</th><th>MFT model</th><th>Observed</th><th>Error</th></tr></thead>
                    <tbody>
                        <tr><td>sin²θ_W = 1 − (E_W/E_Z)²</td>
                            <td>${w.sin2_theta_W_model.toFixed(4)}</td>
                            <td>${w.sin2_theta_W_observed.toFixed(4)}</td>
                            <td>${errorPct(w.sin2_theta_W_model, w.sin2_theta_W_observed)}%</td>
                        </tr>
                    </tbody>
                </table>
            `;
        }

        tripleContainer.innerHTML = `
            <h3>Identified ${sectorInfo.name.toLowerCase()} triple</h3>
            <table class="lepton-triple-table">
                <thead>
                    <tr>
                        <th>Particle</th>
                        <th>F3 mode</th>
                        <th>Morse</th>
                        <th>E (MFT)</th>
                        <th>Predicted (MeV)</th>
                        <th>Observed (MeV)</th>
                        <th>Error</th>
                        <th>ω²</th>
                        <th>ℓ</th>
                        <th>regime</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
            <h4 style="margin-top: 16px;">Mass ratios</h4>
            <table class="lepton-ratios-table">
                <thead><tr><th>Ratio</th><th>MFT model</th><th>Observed</th><th>Error</th></tr></thead>
                <tbody>${ratioRows}</tbody>
            </table>
            ${weinbergRow}
        `;
    } else {
        tripleContainer.innerHTML = `
            <p class="hint">No three-soliton triple matching observed ${sectorInfo.name.toLowerCase()} mass ratios was found in the computed spectrum. Try increasing the number of ω² scan points (currently 40), or check that potential parameters satisfy the silver-ratio condition.</p>
        `;
    }

    // === Potential plot ===
    const potCanvas = document.getElementById('potential-canvas');
    const pc = result.potential_curve;
    const markers = [
        { x: result.phi_barrier, color: '#e67e22', label: `φ_b=${result.phi_barrier.toFixed(3)}` },
        { x: result.phi_vacuum, color: '#7f8c8d', label: `φ_v=${result.phi_vacuum.toFixed(3)}` },
    ];
    if (triple) {
        const idxs = [triple.idx0, triple.idx1, triple.idx2];
        idxs.forEach((idx, slot) => {
            const s = result.spectrum[idx];
            markers.push({ x: s.phi_core, color: SLOT_COLORS[slot], label: triple.particles[slot] });
        });
    }
    plotCurves(potCanvas, [{ xs: pc.phi, ys: pc.V, color: '#333' }], {
        xlabel: 'φ', ylabel: 'V(φ)', markers,
    });

    // === Profile plot ===
    const profCanvas = document.getElementById('profile-canvas');
    const profHeading = document.getElementById('profile-heading');
    if (triple) {
        profHeading.textContent = `${sectorInfo.name} soliton profiles u(r)`;
        const idxs = [triple.idx0, triple.idx1, triple.idx2];
        const normalize = (u, r) => {
            const norm = Math.sqrt(u.reduce((sum, ui, i) => sum + ui * ui * (i > 0 ? r[i] - r[i - 1] : 0), 0));
            return norm > 0 ? u.map(ui => ui / norm) : u;
        };
        const curves = idxs.map((idx, slot) => {
            const s = result.spectrum[idx];
            return {
                xs: s.r,
                ys: normalize(s.u, s.r),
                color: SLOT_COLORS[slot],
                label: triple.particles[slot],
            };
        });
        plotCurves(profCanvas, curves, { xlabel: 'r', ylabel: 'u(r) / norm', xmax: 12 });
    } else {
        profHeading.textContent = 'Soliton profiles u(r)';
        clearCanvas(profCanvas);
        const ctx = profCanvas.getContext('2d');
        ctx.fillStyle = '#888';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No triple identified', profCanvas.width / 2, profCanvas.height / 2);
    }

    // === Full spectrum table ===
    const rows = result.spectrum.map((s, idx) => {
        const isParticle = s.particle ? 'lepton-row' : '';
        const particleLabel = s.particle
            ? `<span class="lepton-tag" style="background:${SLOT_COLORS[s.f3_mode]}">${s.particle}</span>`
            : '';
        const morseCell = (s.morse_index !== null && s.morse_index !== undefined)
            ? `${s.morse_index} <span class="stab-tag stab-${s.stability}">${s.stability}</span>`
            : '<span class="muted">—</span>';
        const f3Cell = (s.f3_mode !== null && s.f3_mode !== undefined) ? `n=${s.f3_mode}` : '';
        const rowStyle = s.particle ? `style="background: ${SLOT_COLORS[s.f3_mode]}15"` : '';
        return `
            <tr class="${isParticle}" ${rowStyle}>
                <td>${idx}${particleLabel}</td>
                <td>${s.E.toFixed(5)}</td>
                <td>${s.mass_MeV.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                <td>${s.omega2.toFixed(4)}</td>
                <td>${s.Q.toFixed(4)}</td>
                <td>${s.phi_core.toFixed(4)}</td>
                <td>${s.n_nodes}</td>
                <td>ℓ=${s.ell}</td>
                <td>${f3Cell}</td>
                <td>${morseCell}</td>
                <td>${s.regime}</td>
            </tr>
        `;
    }).join('');

    tableContainer.innerHTML = `
        <h3>Full discrete spectrum (${result.spectrum.length} solitons, sorted by energy)</h3>
        <p class="hint">
            ω² is an eigenvalue of each soliton, not a free parameter.
            ${triple
                ? `The identified ${sectorInfo.name.toLowerCase()} triple is highlighted; F3 mode and Morse index columns show the Family-of-Three Stability Theorem classification.`
                : 'No triple was identified for this configuration.'
            }
        </p>
        <div class="spectrum-table-wrapper">
            <table class="spectrum-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>E (MFT)</th>
                        <th>Mass (MeV)</th>
                        <th>ω²</th>
                        <th>Q</th>
                        <th>φ_core</th>
                        <th>nodes</th>
                        <th>ℓ</th>
                        <th>F3 mode</th>
                        <th>Morse</th>
                        <th>regime</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

// === Solver call ===

async function runSolver() {
    if (!solverReady) return;

    const solveBtn = document.getElementById('solve-btn');
    solveBtn.disabled = true;
    solveBtn.textContent = 'Computing spectrum… (this takes ~25 seconds)';

    // Clear all stale UI state up front so nothing from a previous sector lingers
    document.getElementById('lepton-triple-container').innerHTML =
        '<p class="hint" style="color:#888;">Computing spectrum…</p>';
    document.getElementById('spectrum-table-container').innerHTML = '';
    clearCanvas(document.getElementById('potential-canvas'));
    clearCanvas(document.getElementById('profile-canvas'));

    // Yield to the browser briefly so the button-disable rendering happens
    await new Promise(resolve => setTimeout(resolve, 50));

    const params = {
        m2: parseFloat(document.getElementById('m2').value),
        lam4: parseFloat(document.getElementById('lam4').value),
        lam6: parseFloat(document.getElementById('lam6').value),
        Z: parseFloat(document.getElementById('Z').value),
        a: parseFloat(document.getElementById('a').value),
        sector: activeSector,
    };

    pyodide.globals.set('js_params', pyodide.toPy(params));
    let result;
    try {
        const pyResult = pyodide.runPython('solve_spectrum(js_params)');
        result = pyResult.toJs({ dict_converter: Object.fromEntries });
        pyResult.destroy();
    } catch (err) {
        result = {
            success: false,
            message: `Python error: ${err.message}`,
            spectrum: [],
            triple: null,
            weinberg: null,
            potential_curve: null,
            phi_barrier: null,
            phi_vacuum: null,
            silver_ratio: null,
            mft_to_mev: null,
            sector: activeSector,
            sector_info: null,
        };
    }

    renderResults(result);

    solveBtn.disabled = false;
    solveBtn.textContent = 'Compute spectrum';
}

// === Preset handling ===

function loadPreset(name) {
    if (!solverReady) return;

    activeSector = name;
    pyodide.globals.set('preset_name', name);
    const presetPy = pyodide.runPython('get_preset(preset_name)');
    const preset = presetPy.toJs({ dict_converter: Object.fromEntries });
    presetPy.destroy();

    document.getElementById('m2').value = preset.m2;
    document.getElementById('lam4').value = preset.lam4;
    document.getElementById('lam6').value = preset.lam6;
    document.getElementById('Z').value = preset.Z;
    document.getElementById('a').value = preset.a;

    document.getElementById('preset-description').textContent = preset.description || '';

    // Mark active preset visually
    document.querySelectorAll('.preset').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.preset === name);
    });
}

// === Wiring ===

document.addEventListener('DOMContentLoaded', () => {
    initPyodide();

    document.getElementById('solve-btn').addEventListener('click', runSolver);

    document.querySelectorAll('.preset').forEach(btn => {
        btn.addEventListener('click', e => {
            const name = e.currentTarget.dataset.preset;
            loadPreset(name);
        });
    });
});
