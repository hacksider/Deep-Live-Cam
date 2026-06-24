# Contributing to Deep-Live-Cam Remote

Thanks for your interest in contributing. This repository is a downstream fork of [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) focused on Google Colab batch processing and a desktop remote controller app.

## Branch model

- `main` is this fork's product/release branch.
- `upstream-main` is a clean sync branch from `hacksider/main`.
- Use short-lived feature branches for changes, for example:
  - `feat/colab-batch-option`
  - `fix/remote-api-cancel`
  - `docs/readme-cleanup`
- For changes intended for upstream `hacksider/Deep-Live-Cam`, branch from `upstream-main` and keep the patch small.

## What belongs in this fork

Good fits:

- Colab notebook and runtime setup improvements.
- Batch photo/video processing fixes.
- Desktop remote controller UI/API work.
- Tailscale/private remote workflow improvements.
- Notebook round-trip tooling and documentation.
- Small upstream-compatible bug fixes, when kept isolated.

Avoid mixing unrelated scopes in one PR. For example, do not combine a notebook setup change, a desktop UI redesign, and a core face-swapper refactor in the same branch.

## Local setup

Windows PowerShell example:

```powershell
git clone https://github.com/djebaz/Deep-Live-Cam-Remote.git
Set-Location .\Deep-Live-Cam-Remote
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the current desktop launcher:

```powershell
.\run_windows_remote_app.ps1
```

Colab users should start from:

```text
google-colab/Deep_Live_Cam_Remote_Batch.ipynb
```

## Notebook edits

The Colab notebook is generated from a markerized Python source file.

Edit this file:

```text
google-colab/Deep_Live_Cam_Remote_Batch.py
```

Then rebuild the notebook from the repo root:

```powershell
python scripts/py_to_ipynb.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  --eol auto
```

If you changed the `.ipynb` directly in Colab, export it back:

```powershell
python scripts/ipynb_to_py.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  --eol auto
```

Preserve cell IDs, marker lines, `meta_b64`, `NOTEBOOK_META_B64`, and markdown/raw sentinels.

## Validation expectations

For code changes, at minimum check syntax for files you edited:

```powershell
python -m py_compile .\colab_batch.py .\colab_api.py
```

Focused tests, when intentionally run:

```powershell
python -m pytest .\tests -q
```

For UI or Colab changes, include manual validation notes in the PR because GPU, Drive, Tailscale, and desktop environments vary.

## Pull request checklist

Before opening a PR, include:

- What changed and why.
- Which workflow is affected: Colab batch, Colab API, desktop app, upstream local app, docs, or notebook tooling.
- Validation performed, or a clear note if validation was deferred.
- Screenshots for visible desktop UI changes when practical.
- Confirmation that generated/cache/model files were not committed.

## Files not to commit

Do not commit:

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- downloaded model files
- local app state files
- temporary round-trip notebooks or scratch exports
- personal Google Drive paths, tokens, or Tailscale secrets

## License and attribution

This fork keeps the upstream license in `LICENSE`. Contributions must be compatible with that license and with the licenses of model/dependency projects used by Deep-Live-Cam.

When contributing changes derived from another project or PR, credit the source in the PR description.

## Responsible use

Do not contribute features whose primary purpose is abuse, deception, evasion, credential theft, or bypassing platform safeguards. Use and distribute this software only with appropriate rights, consent, and legal compliance.
