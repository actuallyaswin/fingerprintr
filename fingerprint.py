# -*- coding: utf-8 -*-
# @Author: Aswin Sivaraman
# @Date:   2018-01-30 02:50:47
# @Last Modified by:   Aswin Sivaraman
# @Last Modified time: 2018-02-02 03:05:31

from heapq import nlargest
from sys import byteorder
from array import array
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import librosa
import os

MAX_DURATION = 10 # seconds
CHUNK_SIZE = 1024
SAMPLE_RATE = 44100
THRESHOLD = 500
FFT_SIZE = 1024

PLT_START = 5
PLT_END = 15

TZ_SIZE = 5 # constellation points
TZ_OFFSET = 3 # offset of the anchor point

def build_catalog(path):

    # List all the files in the path that are WAV files
    files = [x for x in os.listdir(path) if x.endswith('.wav')]

    for file in files:

        # Load the raw audio data
        y, sr = librosa.load(os.path.join(path,file), sr=SAMPLE_RATE, mono=True)

        # For each file, convert the audio to a spectrogram
        D = librosa.stft(y, n_fft=FFT_SIZE)[:200,:]
        num_frames = D.shape[1]

        # Plot the power spectrogram
        P = librosa.amplitude_to_db(D, ref=np.max)
        plt.imshow(P, origin='lower')

        # Create a constellation map
        C = np.zeros_like(P)
        label = 0
        for i in range(0,num_frames,32):
            bins = []
            bins.append((np.max(P[0:20,i]), 0+np.argmax(P[0:20,i])))
            bins.append((np.max(P[20:40,i]), 20+np.argmax(P[20:40,i])))
            bins.append((np.max(P[40:80,i]), 40+np.argmax(P[40:80,i])))
            bins.append((np.max(P[80:160,i]), 80+np.argmax(P[80:160,i])))
            bins.append((np.max(P[160:,i]), 160+np.argmax(P[160:,i])))
            avg = np.mean([bins[i][0] for i in range(len(bins))])
            peaks = [bins[i][1] for i in range(len(bins)) if bins[i][0] > avg]
            for peak in peaks:
                label += 1
                C[peak,i] = label

        # Plot the constellation map
        y, x = np.argwhere(C > 0).T
        plt.scatter(x,y,marker='x',c='white')
        for i in range(len(x)):
            plt.annotate(int(C[y[i],x[i]]), (x[i],y[i]), color='white')

        # Show the plot
        plt.xlim(librosa.core.time_to_frames(PLT_START, sr=SAMPLE_RATE, n_fft=FFT_SIZE),
                librosa.core.time_to_frames(PLT_END, sr=SAMPLE_RATE, n_fft=FFT_SIZE))
        plt.ylim(0,200)
        plt.tight_layout()
        plt.show()

    return

def is_silent(data_chunk):
    return max(data_chunk) < THRESHOLD

def normalize(snd_data):
    MAXIMUM = 16384
    scale = MAXIMUM/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*scale))
    return r

def trim(snd_data):

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def listen_to_microphone():

    # Set up microphone stream
    p = pyaudio.PyAudio()
    stream = p.open(input=True,channels=1,rate=SAMPLE_RATE,frames_per_buffer=CHUNK_SIZE,format=pyaudio.paInt16)
    chunks_per_second = SAMPLE_RATE / CHUNK_SIZE

    # Keep listening to the microphone until it is non-silent,
    # then start recording until it goes silent again or until
    # the recording exceeds MAX_DURATION
    data = array('h')
    started = False
    num_silent = 0
    num_sound = 0
    while True:
        # Listen to a chunk of samples from the microphone
        chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            chunk.byteswap()
        data.extend(chunk)

        # Check for silence
        silent = is_silent(chunk)

        if not silent and not started:
            print("Recording...")
            started = True
        elif started:
            if not silent:
                num_sound += 1
            else:
                num_silent += 1
        if num_sound > (MAX_DURATION * chunks_per_second):
            print("3 second recording done.")
            break
        if num_silent > (3 * chunks_per_second):
            print("Too quiet!")
            break

    # Close the microphone stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Trim and normalize the recording
    data = trim(data)
    data = normalize(data)

    # Return the recording as a NumPy array
    return np.array(data)

def main():

    catalog = build_catalog('data')
    # catalog = load_catalog('catalog.npy')

    # data = listen_to_microphone()


if __name__ == '__main__':
    main()