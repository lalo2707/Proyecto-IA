import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from model import get_model
from utils import get_actions, get_sequences_and_labels, create_folder
from config import MAX_LENGTH_FRAMES, MODEL_NAME, NUM_EPOCH, DATA_PATH, MODELS_PATH

def training_model(data_path, model_path):
    actions = get_actions(data_path)
    sequences, labels = get_sequences_and_labels(actions, data_path)

    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    model = get_model(len(actions))
    model.fit(X, y, epochs=NUM_EPOCH)
    model.summary()
    create_folder(os.path.dirname(model_path))
    model.save(model_path)

if __name__ == "__main__":
    data_path = DATA_PATH
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)

    training_model(data_path, model_path)
