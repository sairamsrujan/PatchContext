# Running PatchContext locally with a free public URL (tunnel)

This runs the app on your own machine (Mac or Windows) and exposes it at a
free public `https://…` URL via a **Cloudflare quick tunnel** — no account, no
card. The URL is live while the app and tunnel are running (e.g. 9–6, off at
night). Because it uses your machine's full CPU, queries are **fast (~20–30s)**,
unlike the single-vCPU cloud version.

> The app needs ~4 GB RAM free. An 8 GB machine is enough if you close other
> heavy apps. On first run it downloads ~2.4 GB of models (once, then cached).

---

## A. macOS

Your Mac is already set up. From the `patchcontext/` folder:

```bash
# 1. one-time: install cloudflared (if not already)
brew install cloudflared

# 2. make sure .env has your LLM keys (LLM_API_KEY / LLM_FALLBACK_API_KEY)
#    (already done during the build)

# 3a. Easiest — the helper script runs the Docker image + tunnel and prints the URL:
./scripts/serve_public.sh

#    …OR 3b. run without Docker (uses the .venv directly):
source .venv/bin/activate
python scripts/fix_macos_libomp.py         # one-time, macOS only
streamlit run app.py --server.port 8501 --server.fileWatcherType none &
cloudflared tunnel --url http://localhost:8501
```

The public URL (`https://<random>.trycloudflare.com`) prints in the
`cloudflared` output. Share that. Press `Ctrl-C` to stop.

---

## B. Windows (e.g. the 8 GB laptop)

Do this once to set up, then step 6 each time you want it live.

**1. Install Python 3.11**
Download from <https://www.python.org/downloads/release/python-3119/> →
"Windows installer (64-bit)". During install, **tick "Add python.exe to PATH"**.

**2. Get the project onto the Windows machine**
Copy the whole `patchcontext` folder over (USB, or push to GitHub and clone).
Make sure it includes `data/index/` (the `embeddings.npy` + `metadata.parquet`
files — the app needs them). You do **not** need the `.venv` folder; you'll
make a fresh one below.

**3. Open PowerShell in the folder** (Shift-right-click the folder → "Open
PowerShell window here") and create the environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

(If PowerShell blocks the activate script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.)

**4. Create the `.env` file with your LLM keys**

```powershell
copy .env.example .env
notepad .env
```

In Notepad, set at least these two lines to your real keys, then save:
```
LLM_API_KEY=nvapi-...your NVIDIA key...
LLM_FALLBACK_API_KEY=sk-or-...your OpenRouter key...
```
(Windows has no GPU, so leave `MODEL_DEVICE=cpu`.)

**5. Install cloudflared**
Easiest: `winget install --id Cloudflare.cloudflared`
(or download `cloudflared-windows-amd64.exe` from
<https://github.com/cloudflare/cloudflared/releases>, rename it to
`cloudflared.exe`, and put it in the project folder).

**6. Run it (this is the everyday command)**
In one PowerShell window (with `.venv` activated):
```powershell
streamlit run app.py --server.port 8501 --server.fileWatcherType none
```
Wait until it says "You can now view your Streamlit app". Then open a **second**
PowerShell window in the same folder and start the tunnel:
```powershell
cloudflared tunnel --url http://localhost:8501
```
The public URL (`https://<random>.trycloudflare.com`) appears in the second
window. Share it. First query loads the models (~1–2 min); after that it's fast.

**To stop:** press `Ctrl-C` in both windows (or just close them).

---

## Getting a stable URL (optional)

The quick-tunnel URL is **random every time you start it**, so a bookmarked
link breaks the next day. If you want the **same URL each run** (e.g. to give
your mentor one link), use `localtunnel` instead of cloudflared:

```bash
# needs Node.js installed (nodejs.org). Same on Mac and Windows:
npx localtunnel --port 8501 --subdomain patchcontext
```
→ `https://patchcontext.loca.lt` (same each time, if the name is free).
Note: localtunnel shows visitors a one-time "click to continue" page.

---

## Troubleshooting

- **"This app has gone over its resource limits" / very slow / freezes:** close
  other apps to free RAM; the app needs ~4 GB. On 8 GB, run nothing else heavy.
- **First query hangs ~2 min:** normal — it's loading the models. Later queries
  are fast.
- **Answers say "not configured" or fail:** the `.env` keys are missing or
  wrong. Re-check `LLM_API_KEY` / `LLM_FALLBACK_API_KEY`.
- **`cloudflared` not found:** re-do step 5; on Windows make sure the `.exe` is
  in the folder or installed via winget.
