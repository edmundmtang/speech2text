# Speech to Text

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pyaudio
import wave
import speech_recognition as sr
from glob import glob

# Choose device to run model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Import model
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# Audio settings
chunk = 1024
sample_format = pyaudio.paInt16 # 16 bits per sample
channels = 1 # Number of audio channels
fs = 44100 # Record at 44100 samples per second, sample rate
time_in_seconds = 3
filename = "soundsample.wav"

p = pyaudio.PyAudio() # Create an interface to PortAudio

print('-----Now Recording-----')
#Open a Streaeem with the value we just defined
stream = p.open(format=sample_format,
               channels = channels,
               rate = fs,
               frames_per_buffer = chunk,
               input = True)

frames = [] # Initialize Array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * time_in_seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the Stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()

print('-----Finished Recording-----')


# Open and Set the data of the WAV file
file = wave.open(filename, 'wb')
file.setnchannels(channels)
file.setsampwidth(p.get_sample_size(sample_format))
file.setframerate(fs)
 
#Write and Close the File
file.writeframes(b''.join(frames))
file.close()

print('-----Saved Audio Recording-----')

test_files = glob(filename)

batches = split_into_batches(test_files, batch_size=10)

input = prepare_model_input(read_batch(batches[0]),
                            device=device)

def quicktest(input) -> None:
    output = model(input)
    for example in output:
        print(decoder(example.cpu()))


