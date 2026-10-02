from pathlib import Path

from dotenv import load_dotenv

BIND_HOST = "127.0.0.1"


def load_web_env(path: Path) -> None:
    """Load .env without replacing variables already set in the process."""
    load_dotenv(path, override=False)


def require_password(value: str | None) -> str:
    if not value:
        raise SystemExit("WEB_PASSWORD is required")
    return value


def web_port(value: str | None) -> int:
    if not value:
        return 8000
    return int(value)


def base_path(value: str | None) -> str:
    """Public URL prefix. Empty means the site root.

    Nginx should strip this prefix before proxying. The app routes stay at /.
    """
    if not value or value.strip() in ("", "/"):
        return ""
    path = value.strip()
    if not path.startswith("/"):
        path = "/" + path
    path = path.rstrip("/")
    if path in ("", "/"):
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-")
    if ".." in path or "//" in path or any(char not in allowed for char in path):
        raise SystemExit("WEB_BASE_PATH is invalid")
    return path


def choose_providers(available: list[str]) -> list[str]:
    for name in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if name in available:
            return [name]
    return available[:1]


def apply_swap_defaults(globals_module) -> None:
    globals_module.mouth_mask = False
    globals_module.opacity = 1.0
    globals_module.poisson_blend = False


def ensure_swapper(pre_start) -> None:
    if not pre_start():
        raise SystemExit(1)
