# Deriv Multi-Symbol Rise/Fall Bot - Deployment

## Files
- `deriv_multisymbol_bot.py` - the entire bot, single file
- `requirements.txt` - dependencies (websockets, numpy, scipy, statsmodels, arch, hmmlearn)
- `Procfile` - tells Railway to run this as a worker process
- `railway.json` - explicit start command + restart policy (backup to Procfile)
- `runtime.txt` - pins Python 3.12 so Railway's Nixpacks builder grabs matching
  compiled wheels for `arch` and `hmmlearn` (these have C extensions; mismatched
  Python versions can force a slow source build or fail)

## Environment variables (set these in Railway's dashboard, or locally before running)
- `DERIV_API_TOKEN` (required) - your Deriv API token
- `DERIV_APP_ID` (optional) - your registered app_id; defaults to the public demo app id `1089`

## Running on your PC first
```bash
pip install -r requirements.txt
export DERIV_API_TOKEN=your_token_here      # Windows (PowerShell): $env:DERIV_API_TOKEN="your_token_here"
python deriv_multisymbol_bot.py
```

## Running on Railway
1. Push these files to a GitHub repo (or use `railway up` from this folder directly).
2. In Railway: New Project -> Deploy from repo (or CLI).
3. Set `DERIV_API_TOKEN` (and `DERIV_APP_ID` if applicable) under Variables.
4. Railway will install `requirements.txt` and run the `Procfile`'s worker command automatically.

## Resource notes
The full-power version fits real statistical models (HMM via Baum-Welch, GARCH
via MLE, Hawkes via MLE optimization) for every symbol during calibration -
this is CPU-bound and briefly memory-heavier than a typical lightweight bot.
On Railway's free/hobby tier this should still run fine for a synthetic-index
universe of ~10-15 symbols, but if you notice calibration taking unusually
long or the service restarting under memory pressure, bump up to a plan with
more RAM/CPU rather than trimming the model fitting itself.

## Before going live
Run on a Deriv **demo account token** first. The initial calibration alone
will take real time (it bootstraps ~3000 ticks per symbol and fits 4 models
per symbol across the whole universe) - watch the console output to confirm
calibration completes cleanly and symbols start getting selected before you
ever point this at a real-money token.
