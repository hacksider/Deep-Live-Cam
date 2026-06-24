#!/usr/bin/env python
"""
Export .ipynb -> markerized .py with cell ids and docstring headers.
Conventions: UTF-8 no BOM, CRLF; markers `# %% [markdown] cell=N id=<id>`.
"""

import argparse
import base64
import fnmatch
import json
import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional


SAFE_CELL_META_KEYS = {"cellView", "colab_type", "editable", "deletable"}
SAFE_COLAB_META_KEYS = {"name", "toc_visible", "include_colab_link", "collapsed_sections", "provenance"}
SAFE_KERNELSPEC_KEYS = {"name", "display_name", "language"}
SAFE_LANGUAGE_INFO_KEYS = {"name"}  # omit version by default to reduce churn
SUPPORTED_CELL_TYPES = {"markdown", "code", "raw"}
EOL_CHOICES = {"crlf", "lf", "auto"}
GENERATED_B64_RE = re.compile(
    r"^(?P<indent>\s*)# IPYNB_GENERATED_B64_FROM_CELL "
    r"target=(?P<target>[A-Za-z_]\w*) source_id=(?P<source_id>[^\s]+) codec=zlib\s*$"
)


def _infer_title(code_lines: List[str], idx: int) -> str:
    for ln in code_lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("# @title"):
            # Colab title convention
            return stripped[len("# @title") :].strip() or f"code-cell-{idx}"
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        return stripped or f"code-cell-{idx}"
    return f"code-cell-{idx}"


def _encode_b64(payload: Dict) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(data).decode("ascii")


def _extract_cell_meta(cell_meta: Dict[str, object]) -> Optional[str]:
    safe = {k: cell_meta[k] for k in SAFE_CELL_META_KEYS if k in cell_meta}
    if not safe:
        return None
    return _encode_b64(safe)


def _extract_notebook_meta(nb: Dict) -> Dict:
    meta = nb.get("metadata", {}) or {}
    out: Dict[str, object] = {}

    colab = meta.get("colab", {}) or {}
    safe_colab = {k: colab[k] for k in SAFE_COLAB_META_KEYS if k in colab}
    if safe_colab:
        out["colab"] = safe_colab

    kernelspec = meta.get("kernelspec", {}) or {}
    safe_kernel = {k: kernelspec[k] for k in SAFE_KERNELSPEC_KEYS if k in kernelspec}
    if safe_kernel:
        out["kernelspec"] = safe_kernel

    langinfo = meta.get("language_info", {}) or {}
    safe_lang = {k: langinfo[k] for k in SAFE_LANGUAGE_INFO_KEYS if k in langinfo}
    if safe_lang:
        out["language_info"] = safe_lang

    if "nbformat" in nb:
        out["nbformat"] = nb["nbformat"]
    if "nbformat_minor" in nb:
        out["nbformat_minor"] = nb["nbformat_minor"]

    return out


def _detect_eol(target: Path, mode: str) -> str:
    if mode in ("crlf", "lf"):
        return mode
    # auto: parse .gitattributes for explicit eol on matching globs
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


def _restore_b64_embed_directives(cells: List[Dict]) -> None:
    cells_by_id = {
        (cell.get("id") or (cell.get("metadata", {}) or {}).get("id")): cell
        for cell in cells
        if cell.get("id") or (cell.get("metadata", {}) or {}).get("id")
    }
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source_lines = [line.rstrip("\r\n") for line in cell.get("source", [])]
        output: List[str] = []
        index = 0
        while index < len(source_lines):
            match = GENERATED_B64_RE.match(source_lines[index])
            if not match:
                output.append(source_lines[index])
                index += 1
                continue
            if index + 1 >= len(source_lines):
                raise ValueError("Generated base64 marker is missing its assignment")
            indent = match.group("indent")
            target = match.group("target")
            source_id = match.group("source_id")
            assignment_re = re.compile(
                rf'^{re.escape(indent)}{re.escape(target)} = "([A-Za-z0-9+/=]+)"$'
            )
            assignment = assignment_re.match(source_lines[index + 1])
            if not assignment:
                raise ValueError(f"Generated base64 assignment is malformed for target {target}")
            source_cell = cells_by_id.get(source_id)
            if not source_cell or source_cell.get("cell_type") != "code":
                raise ValueError(f"Embedded source cell not found or not code: {source_id}")
            expected_source = "".join(source_cell.get("source", []))
            try:
                actual_source = zlib.decompress(base64.b64decode(assignment.group(1))).decode("utf-8")
            except Exception as exc:
                raise ValueError(f"Invalid generated base64 payload for target {target}") from exc
            if actual_source != expected_source:
                raise ValueError(
                    f"Generated base64 payload for {target} is stale relative to cell {source_id}; "
                    "rebuild the notebook from the markerized .py"
                )
            output.append(
                f"{indent}# IPYNB_EMBED_B64_FROM_CELL target={target} "
                f"source_id={source_id} codec=zlib"
            )
            index += 2
        cell["source"] = [line + "\n" for line in output]


def notebook_to_py(ipynb_path: Path, py_path: Path, eol: str = "crlf") -> None:
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    _restore_b64_embed_directives(cells)
    nb_meta = _extract_notebook_meta(nb)
    eol_mode = _detect_eol(py_path, eol)

    lines: List[str] = []
    lines.append("# Auto-generated from notebook; keep markers for round-trip")
    lines.append("# Markers + docstring headers are required for ipynb reconstruction")
    if nb_meta:
        lines.append(f"# NOTEBOOK_META_B64={_encode_b64(nb_meta)}")
    lines.append("")

    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        original_ctype = ctype
        if ctype not in SUPPORTED_CELL_TYPES:
            ctype = "raw"
        cid = cell.get("id") or cell.get("metadata", {}).get("id")
        meta_b64 = _extract_cell_meta(cell.get("metadata", {}) or {})
        marker = f"# %% [{ctype}] cell={idx}"
        if cid:
            marker += f" id={cid}"
        if meta_b64:
            marker += f" meta_b64={meta_b64}"
        lines.append(marker)

        if ctype == "markdown":
            lines.append('"""MARKDOWN')
            for ln in cell.get("source", []):
                lines.append(ln.rstrip("\r\n"))
            lines.append('"""ENDMARKDOWN')
        elif ctype == "raw":
            lines.append('"""RAW')
            if original_ctype and original_ctype != "raw":
                lines.append(f"# Original cell_type={original_ctype}")
            for ln in cell.get("source", []):
                lines.append(ln.rstrip("\r\n"))
            lines.append('"""ENDRAW')
        elif ctype == "code":
            code_lines = [ln.rstrip("\r\n") for ln in cell.get("source", [])]
            title = _infer_title(code_lines, idx)
            lines.append(f'"""CELL: {title}"""')
            lines.extend(code_lines)
        else:
            lines.append(f"# Unsupported cell type: {ctype}")

        lines.append("")  # blank line between cells

    text = "\n".join(lines).rstrip("\n") + "\n"
    text = _apply_eol(text, eol_mode)
    py_path.write_text(text, encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .ipynb to markerized .py")
    parser.add_argument("ipynb", type=Path, help="Input notebook")
    parser.add_argument("py", type=Path, help="Output markerized python file")
    parser.add_argument("--eol", choices=sorted(EOL_CHOICES), default="crlf", help="Line endings: crlf|lf|auto (default crlf)")
    args = parser.parse_args()
    notebook_to_py(args.ipynb, args.py, eol=args.eol)


if __name__ == "__main__":
    main()
