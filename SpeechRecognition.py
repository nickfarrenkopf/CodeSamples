import os
import time
import wave
import librosa
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from scipy.io import wavfile
from scipy.signal import spectrogram
from pynput.keyboard import Key as K

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.Computer import Keyboard


### RECORD ###

def record_single_timed(base_path, word, record_time=2):
    """ records audio and saves it to file """
    print('Recording for {}'.format(word))
    frames = get_frames_timed(record_time)
    filepath = get_next_filepath(base_path, word)
    write_to_audio_file(filepath, frames)

def record_until_done(base_path, word):
    """ record audio until special key is pressed """
    print('Recording for {}'.format(word))
    get_frames_keys(base_path, word)

def get_frames_timed(record_time):
    """ records frames for specified interval """
    p, stream = open_stream()
    frames = []
    for i in range(record_time * rate // chunk):
        if i % (rate // chunk) == 0:
            time_left = record_time - np.round(i * chunk / rate)
            print(' - recording for {}'.format(time_left))
        frames.append(stream.read(chunk))
    close_stream(p, stream)
    return frames

def get_frames_keys(base_path, word, initial_wait_time=1):
    """ record audio constantly, stopping or saving on specific key """
    # start thread to listen to keys
    global key, key_pressed
    key, key_pressed = '', False
    t = Thread(target=key_listener_for_get_frames, args=(initial_wait_time,))
    t.start()
    # start listening to audio
    print('Listening to audio and keys...')
    p, stream = open_stream()
    frames = []
    while not is_letter(key, 'q'):
        # record frames is space is clicked
        if key_pressed and key == K.space:
            frames.append(stream.read(chunk))
        # save frames on s key
        if key_pressed and is_letter(key, 's'):
            frames, key = save_and_clear_frames(base_path, word, frames)
        # save data on space control
        if not key_pressed and key == K.space and len(frames) > 0:
            frames, key = save_and_clear_frames(base_path, word, frames)
    close_stream(p, stream)
    print('Finished listening to audio and keys...')


### AUDIO ###

def open_stream():
    """ open pyaudio stream """
    p = pyaudio.PyAudio()
    stream = p.open(format=fmt, channels=channels, rate=rate, input=True,
                    frames_per_buffer=chunk)
    return p, stream

def close_stream(p, stream):
    """ close pyaudio stream """
    stream.stop_stream()
    stream.close()
    p.terminate()

def save_and_clear_frames(base_path, word, frames):
    """ save frame data to file and reset """
    filepath = get_next_filepath(base_path, word)
    write_to_audio_file(filepath, frames)
    return [], ''


### IMAGE ###

def audio_to_image(filepath, pad=4):
    """ load audio file and convert to mfcc data, saving as image """
    mfcc = file_to_mfcc(filepath)
    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
    mfcc = np.pad(mfcc, ((0, 0), (pad, pad)),
                  mode='constant', constant_values=0)
    DT.save_data_image(mfcc, filepath.replace('.wav', '.png'))

def audio_to_image_all(filepaths):
    """ load audio files and convert to images """
    paths = [file for file in filepaths if '.wav' in file]
    _ = [audio_to_image(p) for p in paths]
  

### KEYBOARD ###

def key_listener_for_get_frames(initial_wait_time):
    """ repeat until q key pressed """
    global key, key_pressed
    time.sleep(initial_wait_time)
    while not is_letter(key, 'q'):
        key, key_pressed = Keyboard.get_key()

def is_letter(key, letter):
    """ check for type of key pressed """
    if key is '' or 'char' not in dir(key):
        return False
    return key.char == letter
    

### FILE ###

def get_next_filepath(base_path, word):
    """ get next file number in iter sequence """
    folder = os.path.join(base_path, word)
    if not os.path.exists(folder):
        os.makedirs(folder)
    idx = len(os.listdir(folder))
    filepath = os.path.join(folder, '{}_{}.wav'.format(word, idx))
    return filepath

def write_to_audio_file(filepath, frames):
    """ write recorded frames to wav file """
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_size)
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print('Saved to {}'.format(filepath))


### TRANSFORM ###

def file_to_mfcc(filepath):
    """ creates mfcc data given filename """
    y, sr = librosa.load(filepath)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)
    mfcc = np.array([m / (m.max() - m.min()) for m in mfcc])
    if len(mfcc[0]) > n_steps:
        start = len(mfcc[0]) // 2
        if start + n_steps > len(mfcc[0]):
            start = len(mfcc[0]) - n_steps
        mfcc = mfcc[:, start:start+n_steps]
    val1 = (n_steps - len(mfcc[0])) // 2
    val2 = n_steps - len(mfcc[0]) - val1
    mfcc = np.pad(mfcc, ((0,0),(val1,val2)), mode='constant',
                  constant_values=mfcc.min())
    mfcc = mfcc.reshape((n_steps, n_mfcc))
    return mfcc

def file_to_specgram(filepath):
    """ creates spectrogram data given filename """
    sr, y = wavfile.read(filepath)
    y = np.array([np.mean(d) for d in y])
    f, t, sxx = spectrogram(y)
    sxx = sxx / sxx.max()
    print(sxx.shape)
    return sxx


### HELPER ###

def plot_many(data, names, n_x=4, n_y=4, figure_size=(12, 6), save_path=False):
    """ color plots output array of shape (100, 100) """
    fig = plt.figure(figsize=figure_size)
    # subplots
    for i in range(n_x * n_y):
        ax = fig.add_subplot(n_x, n_y, i + 1)
        ax.imshow(data[i])
        ax.set_aspect('equal')
        ax.set_xlabel(names[i])
    # other
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig.set_tight_layout(True)
    _ = [fig.savefig(save_path) if save_path else fig.show() for i in range(1)]


### PARAMS ###

# default audio params
fmt = pyaudio.paInt16
chunk = 1024
channels = 2
rate = 44100
sample_size = 2

# transform params
n_mfcc = 64
n_steps = 64

# key listener params
key = ''
key_pressed = False


### PROGRAM ###

if __name__ == '__main__':

    # record and save audio
    #record_single(filepath)
    #write_to_image_path(filepath)
    #record_until_done('', 'test')
    
    # plot test data
    if 0:
        filepath = 'test.wav'
        data = file_to_specgram(filepath)
        print(data.shape)
        plot_many([data], n_x=1, n_y=1)


