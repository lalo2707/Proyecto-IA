from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from config import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

def get_model(output_length: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))  # Solo la última LSTM con return_sequences=False
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



#Mismo modelo pero con mas capas con BatchNormalization para el overftting
""" 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from config import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from config import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

def get_model(output_length: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 """