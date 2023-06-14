import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
%matplotlib inline

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)

create_pngs_from_wavs('audio/noisy_speech/SNR_-3dB', 'Spectrograms/noisy_speech/SNR_-3dB')
create_pngs_from_wavs('audio/noisy_speech/SNR_-6dB', 'Spectrograms/noisy_speech/SNR_-6dB')
create_pngs_from_wavs('audio/noisy_speech/SNR_-9dB', 'Spectrograms/noisy_speech/SNR_-9dB')
create_pngs_from_wavs('audio/test_speech', 'Spectrograms/speech')

from keras.utils import load_img, img_to_array

def load_images_from_path(path):
    images = []

    for file in os.listdir(path):
        images.append(img_to_array(load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        
    return images

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)
        
x = []
y = []

images = load_images_from_path('Spectrograms/noisy_speech/SNR_-3dB')
show_images(images)
    
x += images

images = load_images_from_path('Spectrograms/noisy_speech/SNR_-6dB')
show_images(images)
    
x += images

images = load_images_from_path('Spectrograms/noisy_speech/SNR_-9dB')
show_images(images)
    
x += images

images = load_images_from_path('Spectrograms/noisy_speech/SNR_-9dB')
show_images(images)
    
y += images
y += images
y += images

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_norm = np.array(y_train) / 255
y_test_norm = np.array(y_test) / 255

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

# Build and compile the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(224 * 224 * 3, activation='linear'))  # Output layer for regression

# Reshape target output
y_train_reshaped = np.reshape(y_train_norm, (y_train_norm.shape[0], 224 * 224 * 3))
y_test_reshaped = np.reshape(y_test_norm, (y_test_norm.shape[0], 224 * 224 * 3))

# Train the model
hist = model.fit(x_train_norm, y_train_reshaped, validation_data=(x_test_norm, y_test_reshaped), batch_size=10, epochs=10)

# Reshape predicted output
y_pred_reshaped = np.reshape(model.predict(x_test_norm), (y_test_norm.shape[0], 224, 224, 3))
