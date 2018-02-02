# -*- coding: utf-8 -*-
# @Author: Aswin Sivaraman
# @Date:   2018-01-30 02:50:47
# @Last Modified by:   Aswin Sivaraman
# @Last Modified time: 2018-02-02 05:18:05

from sys import byteorder
from array import array
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import librosa
import json
import os

MAX_DURATION = 5 # seconds
CHUNK_SIZE = 1024
SAMPLE_RATE = 44100
THRESHOLD = 500
FFT_SIZE = 1024

PLT_STARTTIME = 5 # seconds
PLT_DURATION =  5 # seconds

TZ_SIZE = 5 # constellation points
TZ_OFFSET = 3 # offset of the anchor point

CONFIDENCE = 50

def plot_constellation(power,constellation,start=PLT_STARTTIME,duration=PLT_DURATION):
    # Plots the power spectrogram
    plt.imshow(power, origin='lower')

    # Scatter plot the constellation / landmarks
    y, x = np.argwhere(constellation > 0).T
    plt.scatter(x,y,marker='x',c='white')
    for i in range(len(x)):
        plt.annotate(int(constellation[y[i],x[i]]), (x[i],y[i]), color='white')

    # Set axis limits based on top-level parameters
    f_start = librosa.core.time_to_frames(start, sr=SAMPLE_RATE, n_fft=FFT_SIZE)
    f_finish = librosa.core.time_to_frames(start+duration, sr=SAMPLE_RATE, n_fft=FFT_SIZE)
    plt.xlim(f_start, f_finish)
    plt.ylim(0,200)
    plt.tight_layout()
    plt.show()

def pick_best_match(matches):
    # Pick the song with the most matches
    result = max(matches, key=matches.get)

    # Check if the number of matches meets the confidence minimum
    if matches[result] < CONFIDENCE:
        print("Unable to find a matching song!")
    else:
        print("You are listening to",result,"!")

def match_fingerprint(landmarks, catalog):
    # Prepare a histogram for matches
    M = {}

    # Loop through each landmark and generate an address for it
    for i in range(TZ_OFFSET, len(landmarks)-TZ_SIZE):
        anchor = i-TZ_OFFSET
        for j in range(i, i+TZ_SIZE):
            address = str(landmarks[anchor][0]) +\
                        ";" + str(landmarks[j][0]) +\
                        ";" + str(landmarks[j][1]-landmarks[anchor][1])

            # Try finding this address in the catalog
            if address in catalog:
                song_id = catalog[address].split(';')[1]
                if song_id in M:
                    M[song_id] += 1
                else:
                    M[song_id] = 1

    return M

def compute_fingerprint(snd_data):
    # Convert the audio to a spectrogram
    D = librosa.stft(snd_data, n_fft=FFT_SIZE)[:200,:]
    P = librosa.amplitude_to_db(D, ref=np.max)
    num_frames = D.shape[1]

    # Create a map of landmarks
    L = {}
    C = np.zeros_like(P)
    label = 0
    for frame in range(0,num_frames,32):
        bins = []
        bins.append((np.max(P[0:20,frame]), 0+np.argmax(P[0:20,frame]))) # Find a peak in FFT bins 0:20
        bins.append((np.max(P[20:40,frame]), 20+np.argmax(P[20:40,frame]))) # Find a peak in FFT bins 20:40
        bins.append((np.max(P[40:80,frame]), 40+np.argmax(P[40:80,frame]))) # Find a peak in FFT bins 40:80
        bins.append((np.max(P[80:160,frame]), 80+np.argmax(P[80:160,frame]))) # Find a peak in FFT bins 80:160
        bins.append((np.max(P[160:,frame]), 160+np.argmax(P[160:,frame]))) # Find a peak in FFT bins 160:end
        avg = np.mean([bins[i][0] for i in range(len(bins))])
        peaks = [bins[i][1] for i in range(len(bins)) if bins[i][0] > avg] # Only keep the peaks larger than the average
        for frequency in peaks:
            L[label] = (frequency, frame)
            label += 1
            C[frequency,frame] = label

    return L, C, P

def load_catalog(filepath):
    with open(filepath, 'r') as fp:
        r = json.load(fp)
    print("Loaded "+filepath+"!")
    return r

def build_catalog(path):

    # List all the files in the path that are WAV files
    files = [x for x in os.listdir(path) if x.endswith('.wav')]

    # Define a hashmap for the catalog
    A = {}

    for file in files:

        # Set the song ID
        song_id = os.path.splitext(file)[0]

        # Load the raw audio data
        y, sr = librosa.load(os.path.join(path,file), sr=SAMPLE_RATE, mono=True)

        # Run the fingerprint algorithm to get the landmarks
        L, C, P = compute_fingerprint(y)

        # Loop through each landmark, determine the target zones, and save the addresses
        for i in range(TZ_OFFSET, len(L)-TZ_SIZE):
            anchor = i-TZ_OFFSET
            for j in range(i, i+TZ_SIZE):
                # address = [frequency of anchor;frequency of point;delta time between anchor and point]
                address = str(L[anchor][0]) + ";" + str(L[j][0]) + ";" + str(L[j][1]-L[anchor][1])

                # Couple the address with the identifier
                # identifier = [absolute time of the anchor in the song;id of the song]
                A[address] = str(L[anchor][1]) + ";" + str(song_id)

        # plot_constellation(P,C)

    with open('catalog.json', 'w') as fp:
        json.dump(A, fp, sort_keys=True, indent=4, separators=(',', ': '))

    print("Created catalog.json!")

    return A

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
            print(MAX_DURATION,"second recording done.")
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

    # catalog = build_catalog('data')
    catalog = load_catalog('catalog.json')
    data = listen_to_microphone()
    landmarks = compute_fingerprint(data)[0]
    matches = match_fingerprint(landmarks, catalog)
    result = pick_best_match(matches)

if __name__ == '__main__':
    main()