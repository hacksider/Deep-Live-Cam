import argparse
import ipaddress
import re
import subprocess
from pathlib import Path


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".ps1",
    ".bat",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".qss",
    ".svg",
    ".csv",
    ".ipynb",
}

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "build",
    "dist",
    "tmp",
    "tmp_files",
    "__pycache__",
}

WINDOWS_PATH_RE = re.compile(r"(?i)\b[A-Z]:[\\/](?:Users|Documents and Settings|home|tmp|Temp|Program Files)[\\/][^\s\"'<>|]+")
UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9])/(?:Users|home|var|tmp|opt|mnt|srv)/[^\s\"'<>|]+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"(?i)\b(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}\b")
ADDRESS_RE = re.compile(r"(?i)\b\d{1,6}\s+[A-Za-z0-9.\-'\s]{2,}\b(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|court|ct)\b")
POSTAL_RE = re.compile(
    r"(?ix)\b(?:zip(?:\s+code)?|postal(?:\s+code)?)\s*[:#-]\s*(?:\d{5}(?:-\d{4})?|[A-Z]\d[A-Z][ -]?\d[A-Z]\d)\b"
)
DOB_CONTEXT_RE = re.compile(r"(?i)\b(?:dob|date\s*of\s*birth|birth(?:day|date)?)\b.{0,20}\b(?:\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})\b")
US_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
US_EIN_RE = re.compile(r"\b\d{2}-\d{7}\b")
URL_QUERY_PII_RE = re.compile(r"(?i)[?&](?:email|phone|user(?:name|id)?|session(?:_?token)?|auth(?:orization)?|token|csrf|address|zip|postal|dob|birthdate)=[^&\s\"']+")
SESSION_ARTIFACT_RE = re.compile(r"(?i)\b(?:authorization|bearer|cookie|set-cookie|session[_-]?token|csrf(?:token)?|x-api-key|api[_-]?key|access[_-]?token|refresh[_-]?token)\b")
PROFILE_HANDLE_RE = re.compile(r"(?i)\b(?:user(?:name|_?login|_?handle)?|profile(?:_?name)?|account(?:_?name)?)\s*[:=]\s*[\"']?[A-Za-z0-9_-]{4,}[\"']?")
METADATA_RE = re.compile(r"(?i)\b(?:exif|xmp|iptc|creatortool)\b")
METADATA_KEYED_RE = re.compile(r"(?i)\b(?:creator|author|owner|byline|software)\b\s*[:=]\s*['\"][^'\"]{2,}['\"]")

PLACEHOLDER_EMAILS = {
    "redacted@example.invalid",
}
PLACEHOLDER_HANDLES = {
    "example_username",
    "example_user",
    "test_user",
    "test_username",
    "redacted_user",
    "redacted_username",
}


def run_git(args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def should_scan(path: Path, any_extension: bool = False) -> bool:
    if not any_extension and path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    parts = set(path.parts)
    return not any(part in parts for part in EXCLUDED_PARTS)


def changed_files(base_ref: str, head_ref: str) -> list[Path]:
    output = run_git(["diff", "--name-only", "--diff-filter=ACMR", base_ref, head_ref])
    return [Path(line.strip()) for line in output.splitlines() if line.strip()]


def tracked_files() -> list[Path]:
    output = run_git(["ls-files"])
    return [Path(line.strip()) for line in output.splitlines() if line.strip()]


def scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    except OSError:
        return []
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        findings.extend(scan_line(path, lineno, line))
    return findings


def _is_public_ip(token: str) -> bool:
    try:
        ip = ipaddress.ip_address(token)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved)


def _check_absolute_paths(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for regex in (WINDOWS_PATH_RE, UNIX_PATH_RE):
        for match in regex.finditer(line):
            out.append(f"[absolute_path] {path}:{lineno}: {match.group(0)}")
    return out


def _check_email(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for match in EMAIL_RE.finditer(line):
        email = match.group(0).lower()
        if email in PLACEHOLDER_EMAILS or email.endswith("@example.com") or email.endswith("@example.org") or email.endswith("@example.net"):
            continue
        out.append(f"[email] {path}:{lineno}: {match.group(0)}")
    return out


def _check_ip(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for regex in (IPV4_RE, IPV6_RE):
        for match in regex.finditer(line):
            token = match.group(0)
            if _is_public_ip(token):
                out.append(f"[ip] {path}:{lineno}: {token}")
    return out


def _check_address(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for regex in (ADDRESS_RE, POSTAL_RE):
        for match in regex.finditer(line):
            out.append(f"[address] {path}:{lineno}: {match.group(0)}")
    return out


def _check_national_id(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for regex in (US_SSN_RE, US_EIN_RE):
        for match in regex.finditer(line):
            out.append(f"[national_id] {path}:{lineno}: {match.group(0)}")
    return out


def _check_birthdate(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for match in DOB_CONTEXT_RE.finditer(line):
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        if digits and set(digits) == {"0"}:
            continue
        out.append(f"[birthdate] {path}:{lineno}: {value}")
    return out


def _check_url_query(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for match in URL_QUERY_PII_RE.finditer(line):
        out.append(f"[url_query] {path}:{lineno}: {match.group(0)}")
    return out


def _check_session_artifacts(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    if SESSION_ARTIFACT_RE.search(line):
        out.append(f"[session_artifact] {path}:{lineno}: {line.strip()[:200]}")
    return out


def _check_profile_handle(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    for match in PROFILE_HANDLE_RE.finditer(line):
        value = match.group(0)
        handle_match = re.search(r"[:=]\s*[\"']?([A-Za-z0-9_-]{4,})[\"']?$", value)
        if handle_match and handle_match.group(1).lower() in PLACEHOLDER_HANDLES:
            continue
        out.append(f"[profile_handle] {path}:{lineno}: {value}")
    return out


def _check_metadata(path: Path, lineno: int, line: str) -> list[str]:
    out: list[str] = []
    if "re.compile(" in line and "METADATA_RE" in line:
        return out
    if METADATA_RE.search(line) or METADATA_KEYED_RE.search(line):
        out.append(f"[metadata] {path}:{lineno}: {line.strip()[:200]}")
    return out


CHECK_HANDLERS = {
    "absolute_paths": _check_absolute_paths,
    "email": _check_email,
    "ip": _check_ip,
    "address": _check_address,
    "national_id": _check_national_id,
    "birthdate": _check_birthdate,
    "url_query": _check_url_query,
    "session_artifacts": _check_session_artifacts,
    "profile_handle": _check_profile_handle,
    "metadata": _check_metadata,
}

CHECK_PRESETS = {
    "absolute_only": ["absolute_paths"],
    "personal_basic": ["absolute_paths", "email", "ip"],
    "personal_full": [
        "absolute_paths",
        "email",
        "ip",
        "address",
        "national_id",
        "birthdate",
        "profile_handle",
        "metadata",
    ],
}


ACTIVE_CHECKS: list[str] = []


def scan_line(path: Path, lineno: int, line: str) -> list[str]:
    findings: list[str] = []
    for check in ACTIVE_CHECKS:
        findings.extend(CHECK_HANDLERS[check](path, lineno, line))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail on local path/PII leak patterns in tracked text files.")
    parser.add_argument("--mode", choices=("changed", "all"), default="changed")
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--any-extension", action="store_true", help="Scan all tracked file extensions (not just TEXT_EXTENSIONS).")
    parser.add_argument("--checks", default="absolute_only", help="Comma-separated checks or preset name.")
    args = parser.parse_args()

    checks_arg = args.checks.strip()
    if checks_arg in CHECK_PRESETS:
        checks = CHECK_PRESETS[checks_arg]
    else:
        checks = [c.strip() for c in checks_arg.split(",") if c.strip()]
        unknown = [c for c in checks if c not in CHECK_HANDLERS]
        if unknown:
            raise RuntimeError(f"unknown check(s): {', '.join(unknown)}")
    if not checks:
        raise RuntimeError("no checks selected")

    global ACTIVE_CHECKS
    ACTIVE_CHECKS = checks

    repo_root = Path.cwd()
    if args.mode == "all":
        files = tracked_files()
    else:
        base_ref = args.base_ref.strip()
        if not base_ref:
            merge_base = run_git(["merge-base", "origin/main", args.head_ref]).strip()
            if not merge_base:
                raise RuntimeError("unable to resolve merge-base against origin/main")
            base_ref = merge_base
        files = changed_files(base_ref, args.head_ref)

    findings: list[str] = []
    for rel_path in files:
        full_path = repo_root / rel_path
        if not should_scan(rel_path, any_extension=args.any_extension):
            continue
        findings.extend(scan_file(full_path))

    if findings:
        print("Leak pattern(s) detected:")
        for item in findings:
            print(item)
        return 1

    print("No leak patterns detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
