#!/usr/bin/env python
"""
Convert markerized .py -> .ipynb, preserving cell ids and Colab collapsed sections.
Expects markers `# %% [markdown] cell=N id=<id>` or `# %% [code] cell=N id=<id>`.
"""

import argparse
import base64
import fnmatch
import json
import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional


MARKER_RE = re.compile(r"# %% \[(markdown|code|raw)\] cell=\d+(?: id=([^\s]+))?(?: meta_b64=([^\s]+))?")
NOTEBOOK_META_RE = re.compile(r"# NOTEBOOK_META_B64=([A-Za-z0-9+/=]+)")
EOL_CHOICES = {"crlf", "lf", "auto"}
EMBED_B64_DIRECTIVE_RE = re.compile(
    r"^(?P<indent>\s*)# IPYNB_EMBED_B64_FROM_CELL "
    r"target=(?P<target>[A-Za-z_]\w*) source_id=(?P<source_id>[^\s]+) codec=zlib\s*$"
)


def _first_heading_line(source: List[str]) -> Optional[str]:
    import re

    heading_re = re.compile(r"^\s*#{1,6}\s+")
    for ln in source:
        stripped = ln.strip()
        if not stripped:
            continue
        if heading_re.match(stripped):
            return stripped
    return None


def _with_newlines(lines: List[str]) -> List[str]:
    if not lines:
        return []
    if len(lines) == 1:
        return [lines[0]]
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _with_newlines_all(lines: List[str]) -> List[str]:
    return [ln + "\n" for ln in lines] if lines else []


def _decode_b64_meta(raw: Optional[str]) -> Dict:
    if not raw:
        return {}
    data = base64.b64decode(raw.encode("ascii"))
    return json.loads(data.decode("utf-8"))


def _merge_notebook_meta(default: Dict, decoded: Dict, collapsed: List[str]) -> Dict:
    meta = {}

    # Kernelspec
    ks_default = default.get("kernelspec", {}) or {}
    ks_decoded = decoded.get("kernelspec", {}) or {}
    ks = {**ks_default, **ks_decoded}
    if ks:
        meta["kernelspec"] = ks

    # Language info
    li_default = default.get("language_info", {}) or {}
    li_decoded = decoded.get("language_info", {}) or {}
    li = {**li_default, **li_decoded}
    if li:
        meta["language_info"] = li

    colab = decoded.get("colab", {}).copy() if decoded.get("colab") else {}
    if not colab and default.get("colab"):
        colab = default["colab"].copy()

    # collapsed_sections: prefer decoded; fallback to computed
    if "collapsed_sections" not in colab:
        colab["collapsed_sections"] = collapsed
    if colab:
        meta["colab"] = colab

    # Other top-level fields from decoded only
    for key in ("name",):
        if key in decoded:
            meta[key] = decoded[key]

    return meta


def _detect_eol(target: Path, mode: str) -> str:
    if mode in ("crlf", "lf"):
        return mode
    gitattrs = target.parent / ".gitattributes"
    if gitattrs.exists():
        name = target.name
        for line in gitattrs.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pattern, attrs = parts[0], parts[1:]
            if not fnmatch.fnmatch(name, pattern):
                continue
            for attr in attrs:
                if attr.startswith("eol="):
                    eol_val = attr.split("=", 1)[1].lower()
                    if eol_val in ("lf", "crlf"):
                        return eol_val
    return "crlf"


def _apply_eol(text: str, eol: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if eol == "crlf":
        text = text.replace("\n", "\r\n")
    return text


def _materialize_b64_embeds(cells: List[Dict]) -> None:
    cells_by_id = {cell.get("id"): cell for cell in cells if cell.get("id")}
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        output: List[str] = []
        for raw_line in cell.get("source", []):
            line = raw_line.rstrip("\r\n")
            match = EMBED_B64_DIRECTIVE_RE.match(line)
            if not match:
                output.append(raw_line)
                continue
            source_id = match.group("source_id")
            source_cell = cells_by_id.get(source_id)
            if not source_cell or source_cell.get("cell_type") != "code":
                raise ValueError(f"Embedded source cell not found or not code: {source_id}")
            source = "".join(source_cell.get("source", []))
            payload = base64.b64encode(zlib.compress(source.encode("utf-8"), level=9)).decode("ascii")
            indent = match.group("indent")
            target = match.group("target")
            output.append(
                f"{indent}# IPYNB_GENERATED_B64_FROM_CELL target={target} "
                f"source_id={source_id} codec=zlib\n"
            )
            output.append(f'{indent}{target} = "{payload}"\n')
        cell["source"] = output


def py_to_notebook(py_path: Path, ipynb_path: Path, eol: str = "crlf") -> None:
    lines = py_path.read_text(encoding="utf-8").splitlines()
    cells = []
    notebook_meta: Dict = {}
    eol_mode = _detect_eol(ipynb_path, eol)
    seen_first_marker = False

    i = 0
    while i < len(lines):
        line = lines[i]
        mmeta = NOTEBOOK_META_RE.match(line)
        if mmeta and not seen_first_marker:
            notebook_meta = _decode_b64_meta(mmeta.group(1))
            i += 1
            continue

        m = MARKER_RE.match(line)
        if not m:
            if line.startswith("# %%"):
                raise ValueError(f"Unsupported cell marker format at line {i+1}: {line}")
            i += 1
            continue

        seen_first_marker = True
        kind, cid, meta_b64 = m.groups()
        i += 1

        if kind == "markdown":
            if i >= len(lines) or lines[i].strip() != '"""MARKDOWN':
                raise ValueError(f"Missing MARKDOWN opener after cell marker at line {i}")
            i += 1
            md = []
            while i < len(lines) and lines[i].strip() != '"""ENDMARKDOWN':
                md.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError("Unterminated MARKDOWN block")
            i += 1  # skip closing
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": _with_newlines(md),
            }
        elif kind == "raw":
            if i >= len(lines) or lines[i].strip() != '"""RAW':
                raise ValueError(f"Missing RAW opener after cell marker at line {i}")
            i += 1
            md = []
            while i < len(lines) and lines[i].strip() != '"""ENDRAW':
                md.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError("Unterminated RAW block")
                i += 1
            cell = {
                "cell_type": "raw",
                "metadata": {},
                "source": _with_newlines(md),
            }
        else:
            if i < len(lines) and lines[i].startswith('"""CELL:'):
                i += 1  # drop the cell title docstring
            code = []
            while i < len(lines):
                if MARKER_RE.match(lines[i]):
                    break
                code.append(lines[i])
                i += 1
            # Drop one trailing empty string used as cell separator, but preserve intentional blanks
            if code and code[-1] == "":
                code.pop()
            cell = {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": _with_newlines_all(code),
            }

        if cid:
            cell["id"] = cid
            cell["metadata"]["id"] = cid
        cell_meta = _decode_b64_meta(meta_b64)
        if cell_meta:
            cell["metadata"].update(cell_meta)
        cells.append(cell)
        continue

    collapsed = []
    _materialize_b64_embeds(cells)
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        heading = _first_heading_line(c["source"])
        if heading and heading.lstrip().startswith("#") and c.get("id"):
            collapsed.append(c["id"])

    default_meta = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"collapsed_sections": collapsed},
    }

    merged_meta = _merge_notebook_meta(default_meta, notebook_meta, collapsed)

    nb = {
        "cells": cells,
        "metadata": merged_meta,
        "nbformat": notebook_meta.get("nbformat", 4),
        "nbformat_minor": notebook_meta.get("nbformat_minor", 5),
    }

    text = json.dumps(nb, indent=2, ensure_ascii=False)
    text = _apply_eol(text, eol_mode)
    ipynb_path.write_text(text, encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert markerized .py to .ipynb")
    parser.add_argument("py", type=Path, help="Input markerized python file")
    parser.add_argument("ipynb", type=Path, help="Output notebook file")
    parser.add_argument("--eol", choices=sorted(EOL_CHOICES), default="crlf", help="Line endings: crlf|lf|auto (default crlf)")
    args = parser.parse_args()
    py_to_notebook(args.py, args.ipynb, eol=args.eol)


if __name__ == "__main__":
    main()
