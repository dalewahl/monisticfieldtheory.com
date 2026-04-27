/* MFT Q-Ball Solver — JavaScript layer
 *
 * Loads Pyodide, fetches the Python solver module, wires up UI events,
 * runs the solver on demand, renders results to canvases.
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

function plotCurve(canvas, xs, ys, options = {}) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    const padding = { left: 50, right: 20, top: 20, bottom: 35 };
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;

    clearCanvas(canvas);

    if (xs.length === 0) {
        ctx.fillStyle = '#888';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data', W / 2, H / 2);
        return;
    }

    const xmin = options.xmin !== undefined ? options.xmin : Math.min(...xs);
    const xmax = options.xmax !== undefined ? options.xmax : Math.max(...xs);
    const ymin = options.ymin !== undefined ? options.ymin : Math.min(...ys);
    const ymax = options.ymax !== undefined ? options.ymax : Math.max(...ys);
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

    // Zero line if range crosses zero
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

    // Axis labels
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

    // Curve
    ctx.strokeStyle = options.color || '#2c5aa0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
        const px = xToPixel(xs[i]);
        const py = yToPixel(ys[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Vertical markers (e.g., φ_barrier, φ_vacuum, φ_core)
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
}

// === Render results ===

function renderResults(result) {
    const summary = document.getElementById('results-summary');
    const tableContainer = document.getElementById('solitons-table-container');

    if (!result.success) {
        summary.innerHTML = `<p class="error">${result.message}</p>`;
        clearCanvas(document.getElementById('potential-canvas'));
        clearCanvas(document.getElementById('profile-canvas'));
        tableContainer.innerHTML = '';
        return;
    }

    // Potential plot with markers
    const potCanvas = document.getElementById('potential-canvas');
    const pc = result.potential_curve;
    const markers = [];
    if (result.phi_barrier) {
        markers.push({ x: result.phi_barrier, color: '#e67e22', label: `φ_b=${result.phi_barrier.toFixed(3)}` });
    }
    if (result.phi_vacuum) {
        markers.push({ x: result.phi_vacuum, color: '#c0392b', label: `φ_v=${result.phi_vacuum.toFixed(3)}` });
    }
    for (const s of result.solitons) {
        markers.push({
            x: s.phi_core,
            color: '#2c5aa0',
            label: `φ_core=${s.phi_core.toFixed(3)}`,
        });
    }
    plotCurve(potCanvas, pc.phi, pc.V, {
        xlabel: 'φ',
        ylabel: 'V(φ)',
        markers,
    });

    // Profile plot — first soliton found, or empty
    const profCanvas = document.getElementById('profile-canvas');
    if (result.solitons.length > 0) {
        const s = result.solitons[0];
        plotCurve(profCanvas, s.r, s.u, {
            xlabel: 'r',
            ylabel: 'u(r)',
            color: '#2c5aa0',
        });
    } else {
        clearCanvas(profCanvas);
        const ctx = profCanvas.getContext('2d');
        ctx.fillStyle = '#888';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No soliton solution at this ω²', profCanvas.width / 2, profCanvas.height / 2);
    }

    // Summary
    if (result.solitons.length === 0) {
        summary.innerHTML = `
            <p>No soliton solutions found at ω² = ${result.params.omega2}.</p>
            <p class="hint">Try a different ω² value (range 0.05–0.99). For the canonical lepton calibration:
            ω² ≈ 0.96 → electron, ω² ≈ 0.55 → muon, ω² ≈ 0.16 → tau.</p>
        `;
    } else {
        const MFT_TO_MEV = 119.67;
        const rows = result.solitons.map((s, idx) => `
            <tr>
                <td>${idx}</td>
                <td>${s.E.toFixed(5)}</td>
                <td>${(s.E * MFT_TO_MEV).toFixed(2)}</td>
                <td>${s.Q.toFixed(4)}</td>
                <td>${s.A.toFixed(4)}</td>
                <td>${s.phi_core.toFixed(4)}</td>
                <td>${s.n_nodes}</td>
                <td>${s.regime}</td>
            </tr>
        `).join('');

        summary.innerHTML = `
            <p><strong>${result.solitons.length}</strong> soliton solution${result.solitons.length === 1 ? '' : 's'} found at ω² = ${result.params.omega2}.</p>
            <p class="hint">Mass shown in MeV uses MFT_TO_MEV = ${MFT_TO_MEV} (electron-calibrated, canonical parameters).</p>
        `;

        tableContainer.innerHTML = `
            <h3>Soliton spectrum</h3>
            <table class="solitons-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>E (MFT)</th>
                        <th>Mass (MeV)</th>
                        <th>Q</th>
                        <th>A</th>
                        <th>φ_core</th>
                        <th>nodes</th>
                        <th>regime</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    }
}

// === Solver call ===

async function runSolver() {
    if (!solverReady) return;

    const solveBtn = document.getElementById('solve-btn');
    solveBtn.disabled = true;
    solveBtn.textContent = 'Solving…';

    const params = {
        m2: parseFloat(document.getElementById('m2').value),
        lam4: parseFloat(document.getElementById('lam4').value),
        lam6: parseFloat(document.getElementById('lam6').value),
        Z: parseFloat(document.getElementById('Z').value),
        a: parseFloat(document.getElementById('a').value),
        omega2: parseFloat(document.getElementById('omega2').value),
    };

    // Pass params via Pyodide's globals, then call solve()
    pyodide.globals.set('js_params', pyodide.toPy(params));
    let result;
    try {
        const pyResult = pyodide.runPython('solve(js_params)');
        result = pyResult.toJs({ dict_converter: Object.fromEntries });
        pyResult.destroy();
    } catch (err) {
        result = {
            success: false,
            message: `Python error: ${err.message}`,
            solitons: [],
            potential_curve: null,
            phi_barrier: null,
            phi_vacuum: null,
        };
    }

    renderResults(result);

    solveBtn.disabled = false;
    solveBtn.textContent = 'Solve';
}

// === Preset handling ===

function loadPreset(name) {
    if (!solverReady) return;

    pyodide.globals.set('preset_name', name);
    const presetPy = pyodide.runPython('get_preset(preset_name)');
    const preset = presetPy.toJs({ dict_converter: Object.fromEntries });
    presetPy.destroy();

    document.getElementById('m2').value = preset.m2;
    document.getElementById('lam4').value = preset.lam4;
    document.getElementById('lam6').value = preset.lam6;
    document.getElementById('Z').value = preset.Z;
    document.getElementById('a').value = preset.a;
    document.getElementById('omega2').value = preset.omega2;

    document.getElementById('preset-description').textContent = preset.description || '';
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
