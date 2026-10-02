BIND_HOST = "127.0.0.1"


def require_password(value: str | None) -> str:
    if not value:
        raise SystemExit("WEB_PASSWORD is required")
    return value


def web_port(value: str | None) -> int:
    if not value:
        return 8000
    return int(value)


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
