# MFT Q-Ball Solver — Deployment

This folder contains a self-contained interactive solver for the Q-ball soliton
equation in Monistic Field Theory. Everything runs in the visitor's browser
via Pyodide; no server-side code.

## Files

- `index.html` — The page
- `solver.js` — JavaScript layer (UI events, Pyodide orchestration, plotting)
- `solver.py` — Python module (browser-adapted Q-ball solver)
- `style.css` — Page styling
- `README.md` — This file

## Local testing

You cannot just open `index.html` directly in your browser — Pyodide and `fetch()`
require a real HTTP context. Use a simple local server:

```bash
cd qball-solver
python3 -m http.server 8000
# then visit http://localhost:8000
```

Or with Node:

```bash
npx serve .
```

The page will load Pyodide (~10 MB, ~5-10 second first-load), then NumPy and SciPy,
then `solver.py`, then become interactive.

## Deployment to GitHub Pages

### Option A: docs/ subfolder of an existing repo

In your repo `dalewahl/mft`:

```
dalewahl/mft/
├── docs/
│   └── qball-solver/
│       ├── index.html
│       ├── solver.js
│       ├── solver.py
│       ├── style.css
│       └── README.md
└── (rest of repo...)
```

In GitHub repo settings → Pages:
- Source: Deploy from a branch
- Branch: main, folder: /docs

The page will be live at `https://dalewahl.github.io/mft/qball-solver/`.

### Option B: Dedicated site repo

Create a new repo (e.g., `dalewahl/monisticfieldtheory-site`) with the
`qball-solver/` folder at the root, plus eventually a top-level `index.html`
for the landing page.

In repo settings → Pages, deploy from main branch root folder.

### Custom domain (monisticfieldtheory.com)

In repo settings → Pages → Custom domain: `monisticfieldtheory.com`

Add a `CNAME` file at the repo root containing exactly:
```
monisticfieldtheory.com
```

At your domain registrar, configure DNS:

For the apex domain (`monisticfieldtheory.com`):
```
A  @  185.199.108.153
A  @  185.199.109.153
A  @  185.199.110.153
A  @  185.199.111.153
```

For the www subdomain:
```
CNAME  www  dalewahl.github.io
```

GitHub will auto-issue an HTTPS certificate within a few hours after DNS
propagates.

## Performance notes

- First load: ~5-10 seconds (downloading Pyodide + NumPy + SciPy, ~15 MB compressed)
- Subsequent visits: instant (browser cache)
- Each `Solve` call: <1 second for typical parameters

## Adding more solvers

This page is the template. To add another solver:

1. Copy `qball-solver/` to a new folder, e.g. `f3-morse-solver/`
2. Replace `solver.py` with the new physics module (must export a `solve(params)` function)
3. Adjust `index.html` parameter inputs and labels
4. Adjust `solver.js` rendering for new output format if needed
5. Update site nav to link to the new solver

The Pyodide loading, plotting helpers, and overall structure can stay the same.

## Troubleshooting

**Page loads but solver never becomes ready:**
- Check browser console for errors
- Verify Pyodide CDN is reachable (firewall, network)
- Verify `solver.py` is being served (check Network tab)

**"Solve" button does nothing:**
- Open browser console; Python errors are caught and displayed in the results pane
- Check that input values are valid numbers

**Plots don't render:**
- Verify the canvases have non-zero size in the DOM
- Check that `result.solitons` is non-empty and contains expected fields
