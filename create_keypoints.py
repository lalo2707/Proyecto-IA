import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from utils import get_keypoints, insert_keypoints_sequence, create_folder
from config import DATA_PATH, FRAME_ACTIONS_PATH

def create_keypoints(frames_path, save_path):
    data = pd.DataFrame([])

    with Holistic() as model_holistic:
        for n_sample, sample_name in enumerate(os.listdir(frames_path), 1):
            sample_path = os.path.join(frames_path, sample_name)
            if not os.path.isdir(sample_path):
                continue  # Saltar archivos que no son carpetas

            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)

    create_folder(os.path.dirname(save_path))
    data.to_hdf(save_path, key="data", mode="w")

if __name__ == "__main__":
    for etiqueta in os.listdir(FRAME_ACTIONS_PATH):
        etiqueta_path = os.path.join(FRAME_ACTIONS_PATH, etiqueta)
        if not os.path.isdir(etiqueta_path):
            continue

        hdf_path = os.path.join(DATA_PATH, f"{etiqueta}.h5")
        print(f'Creando keypoints de "{etiqueta}"...')
        create_keypoints(etiqueta_path, hdf_path)
        print("✅ Keypoints creados.")
