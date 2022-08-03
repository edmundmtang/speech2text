# Speech to Text

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import io
import torch
import pyaudio # Soundcard audio I/O access library
import wave # Python 3 module for reading / writing simple .wav files
import speech_recognition as sr
import torchaudio


import time

print("Setting up...")
# Choose device to run model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Import model
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# Initialize microphone
r = sr.Recognizer()
mic = sr.Microphone()

print("Ready!")

def record():
    print("Recording...")
    with mic as source:
        audio = r.listen(source)
    return prepare_model_input([read_audio(io.BytesIO(audio.get_wav_data()))], device=device)


def transcribe(input) -> None:
    print("Transcribing...")
    output = model(input)[0]
    print(decoder(output.cuda()))
    

def recordAndTranscribe() -> None:
    while True:
        input = record()
        transcribe(input)
