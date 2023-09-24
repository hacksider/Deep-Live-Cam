import numpy
import opennsfw2
from PIL import Image

from modules.typing import Frame

MAX_PROBABILITY = 0.85


def predict_frame(target_frame: Frame) -> bool:
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    model = opennsfw2.make_open_nsfw_model()
    views = numpy.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
