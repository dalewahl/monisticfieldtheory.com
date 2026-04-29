/* MFT Family-of-Three Visualizer — JavaScript only, no Pyodide.
 *
 * Loads the canonical F3 dataset (theorem_results.json from the published
 * P3 calculation) and visualizes:
 *   1. The four soliton profiles u_n(r) at canonical params
 *   2. The fluctuation eigenvalue spectrum for each mode
 *   3. The constrained Morse indices that prove the F3 theorem
 *
 * The dataset was computed at N=3000, Rmax=14, Z=1.1 — published F3
 * paper parameters. We just present it; we don't recompute.
 */

const PARTICLE_NAMES = ['electron', 'muon', 'tau', '(no fourth)'];
const PARTICLE_LABELS = ['e⁻', 'μ⁻', 'τ⁻', '—'];
const STABILITY = ['stable', 'stable', 'metastable', 'unstable'];
const MODE_COLORS = ['#27ae60', '#2c5aa0', '#b9770e', '#c0392b'];

let data = null;
let currentMode = 0;
let viewMode = 'constrained'; // 'raw' or 'constrained'

const profileCanvas = document.getElementById('profile-canvas');
const eigCanvas = document.getElementById('eig-canvas');

async function loadData() {
    try {
        const resp = await fetch('data/canonical.json');
        const arr = await resp.json();
        data = arr[0];
        document.getElementById('loading-msg').style.display = 'none';
        document.getElementById('main-content').style.display = 'block';
        buildModeSelector();
        renderAll();
    } catch (e) {
        document.getElementById('loading-msg').textContent = `Failed to load dataset: ${e.message}`;
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
        for (const y of c.ys) { if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
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

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotH);
    ctx.lineTo(padding.left + plotW, padding.top + plotH);
    ctx.stroke();

    // Zero-line
    if (ymin < 0 && ymax > 0) {
        const yz = yToPx(0);
        ctx.strokeStyle = '#aaa';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, yz);
        ctx.lineTo(padding.left + plotW, yz);
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

    for (const c of curves) {
        ctx.strokeStyle = c.color || '#2c5aa0';
        ctx.lineWidth = c.lineWidth || 2;
        if (c.dashed) ctx.setLineDash([5, 4]);
        ctx.globalAlpha = c.alpha !== undefined ? c.alpha : 1.0;
        ctx.beginPath();
        for (let i = 0; i < c.xs.length; i++) {
            const px = xToPx(c.xs[i]);
            const py = yToPx(c.ys[i]);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
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
            if (c.dashed) ctx.setLineDash([5, 4]);
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

// === Mode selector UI ===

function buildModeSelector() {
    const container = document.getElementById('mode-selector');
    container.innerHTML = '';

    for (let n = 0; n < 4; n++) {
        const s = data.states[String(n)];
        const stab = STABILITY[n];
        const stabClass = stab === 'stable' ? 'stab-stable' :
                         stab === 'metastable' ? 'stab-meta' :
                         'stab-unstable';

        const row = document.createElement('div');
        row.className = `mode-row ${n === currentMode ? 'active' : ''}`;
        row.dataset.mode = n;
        row.innerHTML = `
            <div class="mode-marker"></div>
            <div class="mode-label">
                u<sub>${n}</sub>
                <span class="mode-particle">${PARTICLE_NAMES[n]} (${PARTICLE_LABELS[n]})</span>
            </div>
            <span class="mode-stab-tag ${stabClass}">${stab}</span>
        `;
        row.addEventListener('click', () => {
            currentMode = n;
            buildModeSelector();
            renderAll();
        });
        container.appendChild(row);
    }
}

// === Main render ===

function renderAll() {
    renderProfilePlot();
    renderEigenvalueSpectrum();
    renderVerdict();
    renderSummaryTable();
}

function renderProfilePlot() {
    const r = data.r_plot;
    const showAll = document.getElementById('show-all-profiles')?.checked;

    const curves = [];
    if (showAll) {
        for (let n = 0; n < 4; n++) {
            const u = data.states[String(n)].u_plot;
            curves.push({
                xs: r,
                ys: u,
                color: MODE_COLORS[n],
                lineWidth: n === currentMode ? 2.5 : 1.4,
                alpha: n === currentMode ? 1.0 : 0.4,
                label: `u${n}: ${PARTICLE_NAMES[n]}`,
            });
        }
    } else {
        const u = data.states[String(currentMode)].u_plot;
        curves.push({
            xs: r, ys: u,
            color: MODE_COLORS[currentMode],
            lineWidth: 2.4,
            label: `u${currentMode}(r)`,
        });
    }

    plotCurves(profileCanvas, curves, {
        xlabel: 'r',
        ylabel: 'u(r)',
        xmin: 0, xmax: 14,
    });
}

function renderEigenvalueSpectrum() {
    const container = document.getElementById('eig-grid');
    container.innerHTML = '';

    // Render eigenvalue rows for all 4 modes, highlighting the current selection
    for (let n = 0; n < 4; n++) {
        const s = data.states[String(n)];
        const f = viewMode === 'constrained' ? s.fluct_constrained : s.fluct_raw;
        const isActive = n === currentMode;

        // Find max abs eigenvalue across all modes for consistent scaling
        let maxAbs = 0;
        for (let m = 0; m < 4; m++) {
            const sm = data.states[String(m)];
            const fm = viewMode === 'constrained' ? sm.fluct_constrained : sm.fluct_raw;
            for (const e of fm.eigvals) maxAbs = Math.max(maxAbs, Math.abs(e));
        }

        // Build bars
        const barsHTML = f.eigvals.map(ev => {
            const w = Math.max(2, (Math.abs(ev) / maxAbs) * 80);
            const cls = ev < 0 ? 'negative' : 'positive';
            return `<div class="eig-bar ${cls}" style="width:${w}px"
                       title="${ev.toFixed(4)}"></div>`;
        }).join('');

        const row = document.createElement('div');
        row.className = 'eig-row';
        row.style.opacity = isActive ? '1.0' : '0.5';
        row.style.borderLeft = isActive ? `3px solid ${MODE_COLORS[n]}` : '1px solid #eaecef';
        row.innerHTML = `
            <span class="eig-mode-label">u<sub>${n}</sub> (${PARTICLE_LABELS[n]})</span>
            <div class="eig-bars">${barsHTML}</div>
            <span class="eig-summary">
                neg: <strong>${f.neg_count}</strong>
            </span>
        `;
        row.addEventListener('click', () => {
            currentMode = n;
            buildModeSelector();
            renderAll();
        });
        row.style.cursor = 'pointer';
        container.appendChild(row);
    }
}

function renderVerdict() {
    const container = document.getElementById('verdict-block');
    const s = data.states[String(currentMode)];
    const constrained_neg = s.fluct_constrained.neg_count;
    const expected = Math.max(0, currentMode - 1);
    const matches = constrained_neg === expected;

    container.innerHTML = `
        <h3>Mode u${currentMode}: ${PARTICLE_NAMES[currentMode]}</h3>
        <p>
            Energy <code>E = ${s.E.toFixed(6)}</code>,
            nodes = <code>${s.nodes}</code>,
            virial value = <code>${s.virial_value.toFixed(4)}</code>
        </p>
        <p>
            Raw Morse index (full Hilbert space): <strong>${s.fluct_raw.neg_count}</strong> negative eigenvalues.
            <br>
            Constrained Morse index (after charge constraint + dilation projection): <strong>${constrained_neg}</strong>.
        </p>
        <p>
            Theorem prediction: <code>m_phys(u${currentMode}) = max(0, ${currentMode}-1) = ${expected}</code>.
            ${matches ? '<strong style="color:#1e7e34;">✓ matches</strong>' : '<strong style="color:#922b21;">✗ does not match</strong>'}
        </p>
    `;
}

function renderSummaryTable() {
    const container = document.getElementById('summary-table-container');
    const tbody = data.states;
    const rowClasses = ['electron', 'muon', 'tau', 'excluded'];

    let html = `
        <h3>The full F3 verdict</h3>
        <table class="f3-summary-table">
            <thead>
                <tr>
                    <th>Mode</th>
                    <th>Identification</th>
                    <th>Nodes</th>
                    <th>Raw m_un</th>
                    <th>Constrained m_phys</th>
                    <th>Theorem: max(0,n-1)</th>
                    <th>Energy E</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>`;

    for (let n = 0; n < 4; n++) {
        const s = tbody[String(n)];
        const expected = Math.max(0, n - 1);
        const constrained = s.fluct_constrained.neg_count;
        const matches = constrained === expected;
        const stab = STABILITY[n];
        const stabClass = stab === 'stable' ? 'stab-stable' :
                         stab === 'metastable' ? 'stab-meta' :
                         'stab-unstable';

        html += `
            <tr class="${rowClasses[n]}">
                <td>u<sub>${n}</sub></td>
                <td>${PARTICLE_NAMES[n]} (${PARTICLE_LABELS[n]})</td>
                <td>${s.nodes}</td>
                <td>${s.fluct_raw.neg_count}</td>
                <td><strong>${constrained}</strong></td>
                <td>${expected}</td>
                <td>${s.E.toFixed(4)}</td>
                <td>
                    <span class="mode-stab-tag ${stabClass}">${stab}</span>
                    ${matches ? ' ✓' : ' ✗'}
                </td>
            </tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

// === Wire up ===

document.addEventListener('DOMContentLoaded', () => {
    loadData();

    // Toolbar: switch raw/constrained
    document.querySelectorAll('.eig-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            viewMode = btn.dataset.view;
            document.querySelectorAll('.eig-toggle').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderEigenvalueSpectrum();
        });
    });

    document.getElementById('show-all-profiles')?.addEventListener('change', renderProfilePlot);
});
