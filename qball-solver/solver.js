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

const LEPTON_COLORS = {
    electron: '#27ae60',
    muon: '#2c5aa0',
    tau: '#c0392b',
};

const LEPTON_OBSERVED_MEV = {
    electron: 0.511,
    muon: 105.66,
    tau: 1776.86,
};

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

    // === Summary ===
    const sr = result.silver_ratio;
    const srBadge = sr.satisfied
        ? '<span class="badge ok">silver-ratio satisfied</span>'
        : `<span class="badge warn">silver-ratio violated (λ₄²=${sr.lam4_squared.toFixed(3)}, 8m₂λ₆=${sr.eight_m2_lam6.toFixed(3)})</span>`;

    const sectorIsLepton = result.sector === 'lepton_sector';
    summary.innerHTML = `
        <p>
            <strong>${result.spectrum.length}</strong> distinct solitons found in the spectrum.
            ${srBadge}
        </p>
        <p class="hint">
            Mass column uses MFT_TO_MEV = ${result.mft_to_mev.toFixed(2)},
            the global conversion factor anchored by the lepton ground-state at the canonical potential
            (m_e = 0.511 MeV). ${sectorIsLepton
                ? 'For the lepton sector this is the correct calibration.'
                : 'Non-lepton sectors apply the same conversion; matching to observed masses in those sectors may require additional MFT-physics adjustments (sector-specific calibration anchors, mass-extraction formulas, or back-reaction corrections) not yet implemented here.'
            }
            φ_barrier = ${result.phi_barrier.toFixed(4)}, φ_vacuum = ${result.phi_vacuum.toFixed(4)}.
        </p>
    `;

    // === Lepton triple ===
    // Only show the lepton-triple table when the user has loaded the lepton
    // sector preset; for quark/boson sectors, even if triple-matching ran,
    // the labels (electron/muon/tau) wouldn't make sense.
    const triple = (result.sector === 'lepton_sector') ? result.lepton_triple : null;
    if (triple) {
        const e = result.spectrum[triple.electron_idx];
        const mu = result.spectrum[triple.muon_idx];
        const ta = result.spectrum[triple.tau_idx];

        const errorPct = (model, observed) =>
            (100 * Math.abs(model - observed) / observed).toFixed(1);

        const r10_model = mu.E / e.E;
        const r21_model = ta.E / mu.E;
        const r20_model = ta.E / e.E;

        tripleContainer.innerHTML = `
            <h3>Identified lepton triple</h3>
            <table class="lepton-triple-table">
                <thead>
                    <tr>
                        <th>Family</th>
                        <th>F3 mode</th>
                        <th>Morse</th>
                        <th>E (MFT units)</th>
                        <th>Predicted (MeV)</th>
                        <th>Observed (MeV)</th>
                        <th>Error</th>
                        <th>ω²</th>
                        <th>regime</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="lepton-row electron">
                        <td><span class="dot" style="background:${LEPTON_COLORS.electron}"></span> electron</td>
                        <td>n=${e.f3_mode}</td>
                        <td>${e.morse_index} <span class="stab-tag stab-${e.stability}">${e.stability}</span></td>
                        <td>${e.E.toFixed(5)}</td>
                        <td>${e.mass_MeV.toFixed(2)}</td>
                        <td>${LEPTON_OBSERVED_MEV.electron}</td>
                        <td>calibration</td>
                        <td>${e.omega2.toFixed(4)}</td>
                        <td>${e.regime}</td>
                    </tr>
                    <tr class="lepton-row muon">
                        <td><span class="dot" style="background:${LEPTON_COLORS.muon}"></span> muon</td>
                        <td>n=${mu.f3_mode}</td>
                        <td>${mu.morse_index} <span class="stab-tag stab-${mu.stability}">${mu.stability}</span></td>
                        <td>${mu.E.toFixed(5)}</td>
                        <td>${mu.mass_MeV.toFixed(2)}</td>
                        <td>${LEPTON_OBSERVED_MEV.muon}</td>
                        <td>${errorPct(mu.mass_MeV, LEPTON_OBSERVED_MEV.muon)}%</td>
                        <td>${mu.omega2.toFixed(4)}</td>
                        <td>${mu.regime}</td>
                    </tr>
                    <tr class="lepton-row tau">
                        <td><span class="dot" style="background:${LEPTON_COLORS.tau}"></span> tau</td>
                        <td>n=${ta.f3_mode}</td>
                        <td>${ta.morse_index} <span class="stab-tag stab-${ta.stability}">${ta.stability}</span></td>
                        <td>${ta.E.toFixed(5)}</td>
                        <td>${ta.mass_MeV.toFixed(2)}</td>
                        <td>${LEPTON_OBSERVED_MEV.tau}</td>
                        <td>${errorPct(ta.mass_MeV, LEPTON_OBSERVED_MEV.tau)}%</td>
                        <td>${ta.omega2.toFixed(4)}</td>
                        <td>${ta.regime}</td>
                    </tr>
                </tbody>
            </table>
            <table class="lepton-ratios-table">
                <thead>
                    <tr><th>Mass ratio</th><th>MFT model</th><th>Observed</th><th>Error</th></tr>
                </thead>
                <tbody>
                    <tr><td>m_μ / m_e</td><td>${r10_model.toFixed(2)}</td><td>206.77</td><td>${errorPct(r10_model, 206.768)}%</td></tr>
                    <tr><td>m_τ / m_μ</td><td>${r21_model.toFixed(3)}</td><td>16.817</td><td>${errorPct(r21_model, 16.817)}%</td></tr>
                    <tr><td>m_τ / m_e</td><td>${r20_model.toFixed(1)}</td><td>3477.2</td><td>${errorPct(r20_model, 3477.2)}%</td></tr>
                </tbody>
            </table>
        `;
    } else {
        const sectorName = result.sector || 'this sector';
        const isLeptonSector = result.sector === 'lepton_sector';
        if (isLeptonSector) {
            tripleContainer.innerHTML = '<p class="hint">No three-soliton triple matching observed lepton ratios was found in the computed spectrum.</p>';
        } else {
            tripleContainer.innerHTML = `
                <h3>Spectrum for ${sectorName.replace('_', ' ')}</h3>
                <p class="hint">
                    The discrete soliton tower is shown below. Specific particle identification
                    (matching to observed quark or boson masses) is currently implemented only for
                    the charged lepton sector. The spectrum here uses Z = ${result.params.Z} from
                    the loaded preset; the same potential parameters as all other sectors.
                </p>
            `;
        }
    }

    // === Potential plot ===
    const potCanvas = document.getElementById('potential-canvas');
    const pc = result.potential_curve;
    const markers = [
        { x: result.phi_barrier, color: '#e67e22', label: `φ_b=${result.phi_barrier.toFixed(3)}` },
        { x: result.phi_vacuum, color: '#7f8c8d', label: `φ_v=${result.phi_vacuum.toFixed(3)}` },
    ];
    if (triple) {
        const e = result.spectrum[triple.electron_idx];
        const mu = result.spectrum[triple.muon_idx];
        const ta = result.spectrum[triple.tau_idx];
        markers.push({ x: e.phi_core, color: LEPTON_COLORS.electron, label: 'e⁻' });
        markers.push({ x: mu.phi_core, color: LEPTON_COLORS.muon, label: 'μ' });
        markers.push({ x: ta.phi_core, color: LEPTON_COLORS.tau, label: 'τ' });
    }
    plotCurves(potCanvas, [{ xs: pc.phi, ys: pc.V, color: '#333' }], {
        xlabel: 'φ', ylabel: 'V(φ)', markers,
    });

    // === Profile plot ===
    const profCanvas = document.getElementById('profile-canvas');
    const profHeading = document.getElementById('profile-heading');
    if (triple) {
        profHeading.textContent = 'Lepton soliton profiles u(r)';
        const e = result.spectrum[triple.electron_idx];
        const mu = result.spectrum[triple.muon_idx];
        const ta = result.spectrum[triple.tau_idx];
        // Normalize each profile for visual comparison
        const normalize = (u, r) => {
            const norm = Math.sqrt(u.reduce((sum, ui, i) => sum + ui * ui * (i > 0 ? r[i] - r[i - 1] : 0), 0));
            return norm > 0 ? u.map(ui => ui / norm) : u;
        };
        plotCurves(profCanvas, [
            { xs: e.r, ys: normalize(e.u, e.r), color: LEPTON_COLORS.electron, label: 'electron' },
            { xs: mu.r, ys: normalize(mu.u, mu.r), color: LEPTON_COLORS.muon, label: 'muon' },
            { xs: ta.r, ys: normalize(ta.u, ta.r), color: LEPTON_COLORS.tau, label: 'tau' },
        ], { xlabel: 'r', ylabel: 'u(r) / norm', xmax: 12 });
    } else {
        profHeading.textContent = 'Soliton profiles u(r)';
        clearCanvas(profCanvas);
        const ctx = profCanvas.getContext('2d');
        ctx.fillStyle = '#888';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        const sectorLabel = (result.sector || 'this sector').replace('_', ' ');
        ctx.fillText('Particle profile overlay not available for', profCanvas.width / 2, profCanvas.height / 2 - 10);
        ctx.fillText(sectorLabel + ' (yet)', profCanvas.width / 2, profCanvas.height / 2 + 10);
    }

    // === Full spectrum table ===
    const rows = result.spectrum.map((s, idx) => {
        const isLepton = s.lepton ? `lepton-row ${s.lepton}` : '';
        const leptonLabel = s.lepton ? `<span class="lepton-tag" style="background:${LEPTON_COLORS[s.lepton]}">${s.lepton}</span>` : '';
        const morseCell = (s.morse_index !== null && s.morse_index !== undefined)
            ? `${s.morse_index} <span class="stab-tag stab-${s.stability}">${s.stability}</span>`
            : '<span class="muted">—</span>';
        const f3Cell = (s.f3_mode !== null && s.f3_mode !== undefined) ? `n=${s.f3_mode}` : '';
        return `
            <tr class="${isLepton}">
                <td>${idx}${leptonLabel}</td>
                <td>${s.E.toFixed(5)}</td>
                <td>${s.mass_MeV.toFixed(2)}</td>
                <td>${s.omega2.toFixed(4)}</td>
                <td>${s.Q.toFixed(4)}</td>
                <td>${s.phi_core.toFixed(4)}</td>
                <td>${s.n_nodes}</td>
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
                ? 'The canonical lepton triple is highlighted; F3 mode and Morse index columns show the Family-of-Three Stability Theorem classification (e, μ stable; τ metastable; higher modes excluded as multiply unstable).'
                : 'Specific particle identification is not implemented for this sector yet — the full discrete tower is shown.'
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
                        <th>F3 mode</th>
                        <th>Morse index</th>
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
            lepton_triple: null,
            potential_curve: null,
            phi_barrier: null,
            phi_vacuum: null,
            silver_ratio: null,
            mft_to_mev: null,
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
