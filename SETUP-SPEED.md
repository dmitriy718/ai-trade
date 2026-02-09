# Faster setup and build (10–15 min → target ~3–5 min)

Setup on a Raspberry Pi can feel heavy because of **pip install** (pandas, numpy, cryptography, etc.), not because of `pyproject.toml`. This doc explains where time goes and how to speed it up.

## Where the time goes

| Step | What runs | Typical time on Pi |
|------|-----------|---------------------|
| `git pull` | Network | ~5–30 s |
| `python3 -m venv venv` | First-time only | ~10–20 s |
| `pip install --upgrade pip` | Network + unpack | ~30–60 s |
| `pip install -r requirements*.txt` | **Main cost**: download + build/wheels | **5–15 min** |

**Important:** `pyproject.toml` is **not** used for installing dependencies in `SuperStart.sh`. Only `requirements.txt` / `requirements-pi.txt` are used. If something “reads pyproject.toml” slowly, it’s likely a **dev** step (e.g. `pip install -e .`, ruff, mypy, pytest), not the normal Pi run. For “just run the bot” you don’t need to install from `pyproject.toml`.

## What we already do

- **requirements-pi.txt** (no TensorFlow) is used when present → avoids 5–10+ min of TensorFlow install on Pi.
- **SuperStart.sh** now uses:
  - `PIP_CACHE_DIR` (default `~/.cache/pip`) so repeat installs reuse downloaded wheels.
  - `pip install --prefer-binary` to avoid building from source when a wheel exists.
  - **FAST_SETUP=1**: skips `git pull` and `pip install --upgrade pip` for quicker restarts when you only need to start the bot.

## Quick wins (already in SuperStart)

```bash
# Faster “just start” run (no git pull, no pip upgrade)
FAST_SETUP=1 ./SuperStart.sh
```

Use this when you haven’t changed code or dependencies and just want the bot (and dashboard) started quickly.

## More ideas to cut the 10–15 min

### 1. Use Pi-specific requirements only (done if you have `requirements-pi.txt`)

- Always use **requirements-pi.txt** on Pi. SuperStart already prefers it. Never use full **requirements.txt** with TensorFlow on Pi unless you really need it.
- Saves **several minutes** and avoids heavy builds.

### 2. Reuse pip cache (done)

- SuperStart sets `PIP_CACHE_DIR`. First run is still slow; **subsequent** installs (e.g. after adding one dependency) are much faster.

### 3. Skip pip upgrade on “run only” (done with FAST_SETUP=1)

- `FAST_SETUP=1` skips `pip install --upgrade pip`. Saves ~30–60 s when you’re not changing the environment.

### 4. Don’t install the project as editable on Pi

- On Pi, **don’t** run `pip install -e .` (no need for it to run `main.py`). That would involve `pyproject.toml` and the build backend and can add time and complexity.
- Installing from **requirements-pi.txt** only is enough and avoids any `pyproject.toml` build.

### 5. Pre-build a venv once, then only reinstall when deps change

- First time: run full SuperStart (or a one-time “setup” script) to create `venv` and install deps.
- Later: if you only change code (no new packages), **don’t** run the “install” step; just restart the bot, e.g.:
  - `pkill -f "python main.py" ; nohup venv/bin/python main.py >> logs/bot_output.log 2>&1 &`
- Re-run the install step only when you change `requirements-pi.txt` or add a new dependency.

### 6. Faster installer: `uv` (optional)

- [uv](https://github.com/astral-sh/uv) can install from `requirements-pi.txt` much faster than pip on Pi.
- One-time: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then `uv venv` and `uv pip install -r requirements-pi.txt`.
- Can replace the “install” block in SuperStart with uv if you want maximum install speed (often 2–5× faster).

### 7. Docker: build once, run many times

- If you use Docker, the **image build** (which runs pip install) is the slow part; **starting** the container is fast.
- Build on a more powerful machine or during off-hours; on Pi only `docker compose up` so startup stays quick.

### 8. Lazy / optional dev tools

- `pyproject.toml` is used by **pytest**, **mypy**, **ruff** — not by the bot at runtime. Don’t run tests/lint on the Pi as part of “start the bot”; run them on your dev machine or in CI. That avoids any perceived “pyproject.toml slowness” during Pi startup.

## Summary

- **Bot runtime startup** is already fast (~5–10 s from `main.py` to “WebSocket connected”).
- The **10–15 min** is almost entirely **first-time (or rare) environment setup**: venv + pip install.
- Use **requirements-pi.txt**, **FAST_SETUP=1** for quick restarts, and **pip cache**; avoid `pip install -e .` and TensorFlow on Pi. Optionally use **uv** or a **pre-built venv** to cut install time further.
