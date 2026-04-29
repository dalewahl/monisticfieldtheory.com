/* MFT Energy Landscape Visualizer — pure JavaScript, no Pyodide.
 *
 * Visualizes how the sextic potential V₆(φ) = m²φ²/2 - λ₄φ⁴/4 + λ₆φ⁶/6
 * topology depends on the dimensionless ratio ρ = λ₄²/(m² λ₆).
 *
 * The silver-ratio condition ρ = 8 produces the canonical MFT double-well
 * with field-space ratio φ_v/φ_b = δ = 1+√2.
 *
 * For ρ < 4 there is no barrier (single-well, monotonic).
 * For ρ = 4 there is a degenerate inflection (barrier just appearing).
 * For 4 < ρ < 8 the barrier exists but is asymmetric.
 * For ρ = 8 we have the silver-ratio condition.
 * For ρ > 8 the barrier-to-vacuum asymmetry grows further.
 */

const DELTA = 1.0 + Math.sqrt(2.0);  // silver ratio

// Hold m² and λ₆ fixed at canonical values; vary λ₄ via ρ
const M2 = 1.0;
const LAM6 = 0.5;

let currentRho = 8.0;  // start at silver-ratio condition

// === Math ===

function lam4FromRho(rho) {
    // ρ = λ₄² / (m² λ₆) → λ₄ = sqrt(ρ m² λ₆)
    return Math.sqrt(rho * M2 * LAM6);
}

function V(phi, m2, lam4, lam6) {
    return 0.5 * m2 * phi * phi
         - 0.25 * lam4 * Math.pow(phi, 4)
         + (1.0 / 6.0) * lam6 * Math.pow(phi, 6);
}

function Vprime(phi, m2, lam4, lam6) {
    return m2 * phi - lam4 * Math.pow(phi, 3) + lam6 * Math.pow(phi, 5);
}

function Vpp(phi, m2, lam4, lam6) {
    return m2 - 3.0 * lam4 * phi * phi + 5.0 * lam6 * Math.pow(phi, 4);
}

function criticalPoints(m2, lam4, lam6) {
    // V'(φ) = φ (m² - λ₄ φ² + λ₆ φ⁴) = 0
    // Quadratic in φ²: λ₆ φ⁴ - λ₄ φ² + m² = 0
    // φ² = (λ₄ ± √(λ₄² - 4 m² λ₆)) / (2 λ₆)
    const disc = lam4 * lam4 - 4.0 * m2 * lam6;
    if (disc < 0) {
        return { phi_b: null, phi_v: null, has_barrier: false, degenerate: false };
    }
    if (Math.abs(disc) < 1e-10) {
        // Degenerate case: phi_b = phi_v
        const phi_eq = Math.sqrt(lam4 / (2 * lam6));
        return { phi_b: phi_eq, phi_v: phi_eq, has_barrier: true, degenerate: true };
    }
    const sqrt_disc = Math.sqrt(disc);
    const phi_b_sq = (lam4 - sqrt_disc) / (2 * lam6);
    const phi_v_sq = (lam4 + sqrt_disc) / (2 * lam6);
    if (phi_b_sq <= 0 || phi_v_sq <= 0) {
        return { phi_b: null, phi_v: null, has_barrier: false, degenerate: false };
    }
    return {
        phi_b: Math.sqrt(phi_b_sq),
        phi_v: Math.sqrt(phi_v_sq),
        has_barrier: true,
        degenerate: false,
    };
}

function classifyRegime(rho) {
    if (rho < 4.0) return 'no-barrier';
    if (Math.abs(rho - 4.0) < 0.05) return 'degenerate';
    if (rho < 7.7) return 'asymmetric';
    if (rho >= 7.7 && rho <= 8.3) return 'silver';
    return 'skewed';
}

const REGIME_LABELS = {
    'no-barrier': 'no barrier',
    'degenerate': 'barrier just emerging',
    'asymmetric': 'barrier present, asymmetric',
    'silver': 'silver-ratio condition',
    'skewed': 'skewed (vacuum dominant)',
};

const REGIME_DESCRIPTIONS = {
    'no-barrier': (
        'When ρ < 4, the potential is a single-well: V(φ) is monotonic for φ > 0. ' +
        'There is no nonlinear vacuum and no soliton solutions exist. The Symmetric ' +
        'Back-Reaction Theorem requires a double-well, so this regime is incompatible ' +
        'with MFT.'
    ),
    'degenerate': (
        'At exactly ρ = 4, the barrier and nonlinear vacuum collapse to a single ' +
        'inflection point φ_b = φ_v = √(λ₄/2λ₆). The discriminant λ₄² − 4m²λ₆ vanishes. ' +
        'This is the boundary between "no barrier" and "double well".'
    ),
    'asymmetric': (
        'For 4 < ρ < 8, the double-well exists but is asymmetric: the back-reaction ' +
        'strengths Σ(φ_b) and Σ(φ_v) differ. The Symmetric Back-Reaction theorem is ' +
        'violated. The hierarchy ratio Δm²₃₂/Δm²₂₁ deviates from δ⁴ - 1.'
    ),
    'silver': (
        'At ρ = 8 exactly, the Symmetric Back-Reaction theorem is satisfied: ' +
        'Σ(φ_b) = Σ(φ_v). The field-space ratio φ_v/φ_b = δ = 1+√2 (silver ratio). ' +
        'V(φ_b) = +1/(3δ), V(φ_v) = −δ/3. This is canonical MFT.'
    ),
    'skewed': (
        'For ρ > 8, the barrier shrinks while the nonlinear vacuum deepens. The ' +
        'symmetry of the back-reaction is broken in the opposite direction from the ' +
        'asymmetric case. The hierarchy ratio overshoots δ⁴ - 1.'
    ),
};

// === Plot helpers ===

function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function plotV(canvas, rho) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const padding = { left: 60, right: 30, top: 30, bottom: 50 };
    const plotW = W - padding.left - padding.right;
    const plotH = H - padding.top - padding.bottom;

    clearCanvas(canvas);

    const lam4 = lam4FromRho(rho);
    const cp = criticalPoints(M2, lam4, LAM6);

    // Determine plot range
    // x: phi from -0.2 to ~ phi_v * 1.2 if barrier exists, else 2.5
    let phi_min = -0.2;
    let phi_max = cp.has_barrier && cp.phi_v !== null
        ? Math.max(2.4, cp.phi_v * 1.25)
        : 2.5;

    // Sample V at many points
    const N = 400;
    const phi_arr = [], V_arr = [];
    for (let i = 0; i <= N; i++) {
        const phi = phi_min + (phi_max - phi_min) * i / N;
        phi_arr.push(phi);
        V_arr.push(V(phi, M2, lam4, LAM6));
    }

    // y range: include all V values + 10% padding
    let ymin = Math.min(...V_arr);
    let ymax = Math.max(...V_arr);
    // Lock the visible y-range to a stable window so the plot doesn't jump too much
    // Reasonable defaults at silver ratio: V_v = -δ/3 ≈ -0.805, V_b = 1/(3δ) ≈ 0.138
    ymin = Math.min(ymin, -1.0);
    ymax = Math.max(ymax, 0.4);
    // Pad
    const yrange = ymax - ymin;
    ymin -= yrange * 0.05;
    ymax += yrange * 0.05;
    const xrange = phi_max - phi_min;

    const xToPx = phi => padding.left + ((phi - phi_min) / xrange) * plotW;
    const yToPx = v => padding.top + plotH - ((v - ymin) / (ymax - ymin)) * plotH;

    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(padding.left, padding.top, plotW, plotH);

    // Subtle grid
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 0.5;
    for (let v = Math.ceil(ymin * 4) / 4; v <= ymax; v += 0.25) {
        const py = yToPx(v);
        ctx.beginPath();
        ctx.moveTo(padding.left, py);
        ctx.lineTo(padding.left + plotW, py);
        ctx.stroke();
    }
    for (let phi = 0; phi <= phi_max; phi += 0.5) {
        const px = xToPx(phi);
        if (px < padding.left) continue;
        ctx.beginPath();
        ctx.moveTo(px, padding.top);
        ctx.lineTo(px, padding.top + plotH);
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
        ctx.lineWidth = 1;
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
    [ymin, (ymin + ymax) / 2, ymax].forEach(v => {
        ctx.fillText(v.toFixed(2), padding.left - 5, yToPx(v) + 4);
    });
    ctx.textAlign = 'center';
    [0, 0.5, 1.0, 1.5, 2.0, 2.5].forEach(phi => {
        if (phi >= phi_min && phi <= phi_max) {
            ctx.fillText(phi.toFixed(1), xToPx(phi), padding.top + plotH + 14);
        }
    });

    // Axis labels
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.fillText('φ', padding.left + plotW / 2, H - 10);
    ctx.save();
    ctx.translate(15, padding.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('V(φ)', 0, 0);
    ctx.restore();

    // V(φ) curve
    const regime = classifyRegime(rho);
    const curveColor = {
        'no-barrier': '#c0392b',
        'degenerate': '#bc6c25',
        'asymmetric': '#b9770e',
        'silver': '#234a85',
        'skewed': '#6a4c93',
    }[regime];

    ctx.strokeStyle = curveColor;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < phi_arr.length; i++) {
        const px = xToPx(phi_arr[i]);
        const py = yToPx(V_arr[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Critical points
    const points = [{ phi: 0, color: '#27ae60', label: 'φ=0' }];
    if (cp.has_barrier) {
        if (cp.degenerate) {
            points.push({ phi: cp.phi_b, color: '#bc6c25', label: 'φ_b = φ_v (inflection)' });
        } else {
            points.push({ phi: cp.phi_b, color: '#d4a017', label: 'φ_b' });
            points.push({ phi: cp.phi_v, color: '#c0392b', label: 'φ_v' });
        }
    }

    for (const p of points) {
        const px = xToPx(p.phi);
        const py = yToPx(V(p.phi, M2, lam4, LAM6));
        if (px < padding.left || px > padding.left + plotW) continue;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label below the point (or above if y is near top)
        ctx.fillStyle = '#222';
        ctx.font = 'bold 11px sans-serif';
        const labelY = py < padding.top + plotH * 0.3 ? py + 22 : py - 12;
        ctx.textAlign = 'center';
        ctx.fillText(p.label, px, labelY);
    }
}

// === UI updates ===

function updateAll(rho) {
    currentRho = rho;
    const lam4 = lam4FromRho(rho);
    const cp = criticalPoints(M2, lam4, LAM6);
    const regime = classifyRegime(rho);

    // Slider container border
    const sliderContainer = document.getElementById('rho-slider-container');
    sliderContainer.classList.toggle('silver-active', regime === 'silver');

    // Rho display
    document.getElementById('rho-value').textContent = rho.toFixed(3);
    const tag = document.getElementById('silver-tag');
    if (regime === 'silver') {
        tag.className = 'silver-tag is-silver';
        tag.textContent = '✦ silver-ratio condition';
    } else if (!cp.has_barrier) {
        tag.className = 'silver-tag no-barrier';
        tag.textContent = 'no barrier';
    } else {
        tag.className = 'silver-tag has-barrier';
        const offset = ((rho - 8.0) / 8.0 * 100);
        tag.textContent = `${offset > 0 ? '+' : ''}${offset.toFixed(1)}% from silver`;
    }

    // Critical points block
    const cpBlock = document.getElementById('crit-points');
    if (cp.has_barrier) {
        const phi_b_str = cp.phi_b !== null ? cp.phi_b.toFixed(4) : '—';
        const phi_v_str = cp.phi_v !== null ? cp.phi_v.toFixed(4) : '—';
        const V_b_str = cp.phi_b !== null ? V(cp.phi_b, M2, lam4, LAM6).toFixed(4) : '—';
        const V_v_str = cp.phi_v !== null ? V(cp.phi_v, M2, lam4, LAM6).toFixed(4) : '—';
        cpBlock.innerHTML = `
            <h4>Critical points</h4>
            <div class="crit-row cp-zero">
                <div class="crit-dot"></div>
                <span class="crit-name">φ = 0</span>
                <span class="crit-vals">V = <strong>0.0000</strong></span>
            </div>
            <div class="crit-row cp-barrier">
                <div class="crit-dot"></div>
                <span class="crit-name">φ_b ${cp.degenerate ? '(= φ_v)' : '(barrier)'}</span>
                <span class="crit-vals">φ = <strong>${phi_b_str}</strong>, V = <strong>${V_b_str}</strong></span>
            </div>
            ${cp.degenerate ? '' : `
            <div class="crit-row cp-vacuum">
                <div class="crit-dot"></div>
                <span class="crit-name">φ_v (vacuum)</span>
                <span class="crit-vals">φ = <strong>${phi_v_str}</strong>, V = <strong>${V_v_str}</strong></span>
            </div>
            `}
        `;
    } else {
        cpBlock.innerHTML = `
            <h4>Critical points</h4>
            <p style="color:#922b21; font-size:13px; margin: 4px 0;">
                Only φ = 0 (no barrier exists at ρ = ${rho.toFixed(2)} < 4).
            </p>
        `;
    }

    // Silver-ratio comparison block
    const silverBlock = document.getElementById('silver-comparison');
    if (cp.has_barrier && !cp.degenerate) {
        const fieldRatio = cp.phi_v / cp.phi_b;
        const fieldRatioErr = (fieldRatio - DELTA) / DELTA * 100;
        const V_b = V(cp.phi_b, M2, lam4, LAM6);
        const V_v = V(cp.phi_v, M2, lam4, LAM6);
        const V_v_silver = -DELTA / 3.0;
        const V_b_silver = 1.0 / (3.0 * DELTA);

        silverBlock.innerHTML = `
            <h4>Silver-ratio targets</h4>
            <div class="ratio-row">
                <span class="ratio-name">φ_v/φ_b</span>
                <span class="ratio-val"><strong>${fieldRatio.toFixed(4)}</strong></span>
                <span class="target">target δ = ${DELTA.toFixed(4)}<br>${fieldRatioErr >= 0 ? '+' : ''}${fieldRatioErr.toFixed(2)}%</span>
            </div>
            <div class="ratio-row">
                <span class="ratio-name">V(φ_b)</span>
                <span class="ratio-val"><strong>${V_b.toFixed(4)}</strong></span>
                <span class="target">target 1/(3δ) = ${V_b_silver.toFixed(4)}</span>
            </div>
            <div class="ratio-row">
                <span class="ratio-name">V(φ_v)</span>
                <span class="ratio-val"><strong>${V_v.toFixed(4)}</strong></span>
                <span class="target">target −δ/3 = ${V_v_silver.toFixed(4)}</span>
            </div>
        `;
    } else {
        silverBlock.innerHTML = `
            <h4>Silver-ratio targets</h4>
            <p style="color:#888; font-size: 12.5px; margin: 4px 0;">
                Silver-ratio comparison requires a non-degenerate double-well (ρ > 4).
            </p>
        `;
    }

    // Regime tag in plot title
    const regimeTag = document.getElementById('regime-tag');
    regimeTag.className = `regime-tag regime-${regime}`;
    regimeTag.textContent = REGIME_LABELS[regime];

    // Regime description
    document.getElementById('regime-description').textContent = REGIME_DESCRIPTIONS[regime];

    // Redraw plot
    plotV(document.getElementById('v-canvas'), rho);
}

// === Wire up ===

const PRESETS = {
    'no-barrier':  { rho: 2.5, label: 'No barrier (ρ=2.5)' },
    'degenerate':  { rho: 4.0, label: 'Degenerate (ρ=4)' },
    'asymmetric':  { rho: 6.0, label: 'Asymmetric (ρ=6)' },
    'silver':      { rho: 8.0, label: 'Silver ratio (ρ=8)' },
    'skewed':      { rho: 12.0, label: 'Skewed (ρ=12)' },
};

document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('rho-slider');

    slider.addEventListener('input', (e) => {
        const rho = parseFloat(e.target.value);
        updateAll(rho);
    });

    document.querySelectorAll('.preset[data-regime]').forEach(btn => {
        btn.addEventListener('click', () => {
            const regime = btn.dataset.regime;
            const rho = PRESETS[regime].rho;
            slider.value = rho;
            updateAll(rho);
            // Mark active
            document.querySelectorAll('.preset[data-regime]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Initial render
    slider.value = currentRho;
    updateAll(currentRho);
    // Mark silver as active by default
    document.querySelector('.preset[data-regime="silver"]')?.classList.add('active');
});
