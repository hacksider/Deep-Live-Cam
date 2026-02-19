from modules.utilities import has_image_extension


def analyze_target(start, root):
    import modules.globals
    from modules.ui import (
        POPUP,
        update_status,
        create_source_target_popup,
        select_output_path,
    )
    from modules.utilities import is_image, is_video
    from modules.face_analyser import (
        get_unique_faces_from_target_image,
        get_unique_faces_from_target_video,
    )

    if POPUP is not None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if modules.globals.map_faces:
        with modules.globals.MAP_LOCK:
            modules.globals.source_target_map = []

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_video()

        if len(modules.globals.source_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.source_target_map)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def check_and_ignore_nsfw(target, destroy=None):
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame
    from modules.ui import update_status

    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False
