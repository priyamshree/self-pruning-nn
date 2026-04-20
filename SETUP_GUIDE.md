# Setup & Submission Guide (Priyam's copy)

This file is for **you**, not the reviewer. Once you're done with submission you can delete it from the repo (or keep it — doesn't hurt). Every command below is copy-paste.

---

## The one-paragraph plan

Develop the code in **VS Code on your HP Pavilion**. Push it to **GitHub** (private or public, your call). Open the Colab notebook, point it at your repo, run the GPU training there (~5 minutes). Download the `results/` folder Colab produced, drop the numbers + plot into `report.md`, commit, push again. Share the GitHub link with Tredence.

You don't need a GPU on your laptop for any of this.

---

## Step 0 — One-time installs on Windows 11

If you haven't already:

1. **Install Git**: https://git-scm.com/download/win (accept defaults).
2. **Install VS Code**: https://code.visualstudio.com/ (accept defaults).
3. **Install Python 3.11**: https://www.python.org/downloads/windows/ — ✅ tick *"Add python.exe to PATH"* during setup.
4. Open VS Code → Extensions sidebar → install:
   - `Python` (Microsoft)
   - `Jupyter` (Microsoft)
   - `GitLens` (optional, nice to have)
5. **Create a GitHub account** if you don't have one: https://github.com/signup

Verify in a **new** PowerShell window:

```powershell
git --version
python --version
code --version
```

All three should print a version number.

---

## Step 1 — Drop the project into VS Code

1. Create a folder anywhere you like, e.g. `C:\Users\Priyam\code\self-pruning-nn`.
2. Copy **every file from this repo** (the one I built) into that folder. Layout:
   ```
   self-pruning-nn/
   ├── self_pruning_nn.py
   ├── notebooks/run_on_colab.ipynb
   ├── report.md
   ├── README.md
   ├── requirements.txt
   ├── .gitignore
   └── SETUP_GUIDE.md        (this file)
   ```
3. In VS Code: `File -> Open Folder...` and select the folder.

---

## Step 2 — Create a local Python virtual environment (optional but clean)

In VS Code's integrated terminal (`` Ctrl+` ``):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation with an execution-policy error, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

then `.venv\Scripts\activate` again.

VS Code will prompt you to select the `.venv` interpreter — accept.

---

## Step 3 — Local smoke test (optional, 2–5 minutes on CPU)

```powershell
python self_pruning_nn.py --quick
```

`--quick` runs 1 epoch on 1 lambda. If you see the training log print and a CIFAR-10 download happen, everything is wired up. You do **not** need to do the full training locally — that's what Colab is for.

---

## Step 4 — Initialize Git and make the first commit

In the terminal, inside the project folder:

```powershell
git init
git add .
git config user.name  "Priyam"
git config user.email "shree.priyam.1953@gmail.com"
git commit -m "Initial commit: self-pruning neural network scaffold"
```

---

## Step 5 — Push to GitHub

### Easiest path: GitHub CLI (`gh`)

Install once from https://cli.github.com/, then:

```powershell
gh auth login                         # pick GitHub.com, HTTPS, browser login
gh repo create self-pruning-nn --public --source=. --push
```

Done — that creates the repo and pushes in one go.

### Alternative: do it in the browser

1. Go to https://github.com/new → repository name `self-pruning-nn` → Public → **do not** tick "Initialize with README" → Create.
2. GitHub shows a page with commands. Run these in VS Code's terminal (GitHub will give you the exact URL):
   ```powershell
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/self-pruning-nn.git
   git push -u origin main
   ```
3. You'll be prompted for login — use the browser flow or a Personal Access Token.

Open the repo page in a browser and confirm all files are visible.

---

## Step 6 — Train on Colab (the actual heavy work)

1. Go to https://colab.research.google.com/.
2. `File -> Open notebook -> GitHub` tab.
3. Paste your repo URL (e.g. `https://github.com/YOUR_USERNAME/self-pruning-nn`).
4. Open `notebooks/run_on_colab.ipynb`.
5. **`Runtime -> Change runtime type -> T4 GPU -> Save`.**
6. In the second code cell, uncomment the two `git clone` lines and edit the URL to your repo:
   ```python
   !git clone https://github.com/YOUR_USERNAME/self-pruning-nn.git
   %cd self-pruning-nn
   ```
7. `Runtime -> Run all`.

The full sweep (3 lambdas × 15 epochs) takes about 5 minutes on a T4. The last cell will prompt to download `results.zip` to your laptop.

---

## Step 7 — Commit the results

Unzip `results.zip` into your project folder so the `results/` directory has:

```
results/
├── results.csv
├── results_table.md
├── runs.json
└── gate_distribution_best.png
```

Open `report.md` in VS Code and:

1. Copy the numbers from `results/results_table.md` into the table in **Section 2** of `report.md`.
2. Skim the "Reading the trade-off" paragraph — if your actual numbers contradict what the paragraph claims, rewrite the relevant sentence in your own words. A reviewer will spot a mismatched description faster than a mismatched number.

Then commit and push:

```powershell
git add results/ report.md
git commit -m "Add training results and gate distribution plot"
git push
```

---

## Step 8 — Submit

Reply to the Tredence email with:

1. **Your resume (PDF)** — attach.
2. **GitHub / portfolio links** — include your profile and this repo URL.
3. **Case study** — the repo URL:
   `https://github.com/YOUR_USERNAME/self-pruning-nn`

Template you can paste and edit:

> Hi Tredence Studio — AI Agents Engineering Team,
>
> Please find my application for the AI Engineering Internship — 2025 cohort below.
>
> **Resume:** attached.
> **GitHub profile:** https://github.com/YOUR_USERNAME
> **Case study — The Self-Pruning Neural Network:** https://github.com/YOUR_USERNAME/self-pruning-nn
>
> The repo contains the single training script, a short report (`report.md`), a Colab notebook used for the actual GPU training, and the results (accuracy + sparsity for three lambda values plus a gate-distribution plot).
>
> Happy to walk through any part of the implementation in an interview.
>
> Regards,
> Priyam

---

## Sanity checklist before you hit send

- [ ] Repo is public (or you've added Tredence as a collaborator if private).
- [ ] `self_pruning_nn.py` is at the repo root.
- [ ] `report.md` has **real numbers** in the lambda table, not the `_fill after run_` placeholders.
- [ ] `results/gate_distribution_best.png` is committed and shows a spike near 0 + cluster near 1.
- [ ] `README.md` renders cleanly on GitHub.
- [ ] You can `git clone` your own repo into a brand-new folder and run `python self_pruning_nn.py --quick` without editing anything.

---

## Troubleshooting

**"CIFAR-10 download is stuck" on Colab.** Rare. Restart the runtime and re-run. The dataset is ~170 MB.

**"CUDA out of memory" on Colab.** Shouldn't happen with an MLP this small. If it does, drop `--batch-size` to 64.

**"Accuracy is ~10%" (random).** Almost always a bug in gate initialization. Verify `gate_init=2.0` in `PrunableLinear.__init__` so initial gates aren't near zero.

**"Sparsity is 0%".** Lambda is too small — try the next step up, or train for more epochs.

**"Sparsity is 100% and accuracy is 10%".** Lambda is too big — drop it an order of magnitude.

**Git push asks for a password.** GitHub removed password auth. Use `gh auth login`, or generate a Personal Access Token at https://github.com/settings/tokens and paste it in place of the password.

---

You've got this. The problem is well-defined, your code is clean, and the grading criteria reward exactly what this repo does: a correct custom layer, a working sparsity loss, and a clear trade-off analysis.