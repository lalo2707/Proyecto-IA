import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from utils import get_keypoints, insert_keypoints_sequence, create_folder, get_actions
from config import DATA_PATH, FRAME_ACTIONS_PATH


def create_keypoints(frames_path, save_path):
    data = pd.DataFrame([])

    with Holistic() as model_holistic:
        for n_sample, sample_name in enumerate(os.listdir(frames_path), 1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)

    create_folder(os.path.dirname(save_path))
    data.to_hdf(save_path, key="data", mode="w")


if __name__ == "__main__":
    words_path = FRAME_ACTIONS_PATH

    for word_name in os.listdir(words_path):
        word_path = os.path.join(words_path, word_name)
        hdf_path = os.path.join(DATA_PATH, f"{word_name}.h5")
        print(f'Creando keypoints de "{word_name}"...')
        create_keypoints(word_path, hdf_path)
        print(f"Keypoints creados!")
