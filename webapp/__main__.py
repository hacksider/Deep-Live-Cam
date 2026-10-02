import os
from pathlib import Path

import onnxruntime
import uvicorn

import modules.globals
from modules.face_analyser import detect_one_face_fast, get_one_face
from modules.processors.frame.face_swapper import pre_start, swap_face
from webapp.engine import LiveEngine
from webapp.images import decode_image, encode_jpeg
from webapp.runtime import (
    BIND_HOST,
    apply_swap_defaults,
    base_path,
    choose_providers,
    ensure_swapper,
    load_web_env,
    require_password,
    web_port,
)
from webapp.server import create_app


def main() -> None:
    load_web_env(Path(__file__).resolve().parent.parent / ".env")
    password = require_password(os.environ.get("WEB_PASSWORD"))
    port = web_port(os.environ.get("WEB_PORT"))
    apply_swap_defaults(modules.globals)
    modules.globals.execution_providers = choose_providers(
        onnxruntime.get_available_providers()
    )
    ensure_swapper(pre_start)
    engine = LiveEngine(
        get_one_face=get_one_face,
        detect_one_face=detect_one_face_fast,
        swap_face=swap_face,
        decode_image=decode_image,
        encode_jpeg=encode_jpeg,
    )
    secure_cookie = os.environ.get("WEB_SECURE_COOKIE") != "0"
    app = create_app(
        engine,
        password,
        secure_cookie=secure_cookie,
        base_path=base_path(os.environ.get("WEB_BASE_PATH")),
    )
    uvicorn.run(app, host=BIND_HOST, port=port)


if __name__ == "__main__":
    main()
