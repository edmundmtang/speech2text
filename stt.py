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
    tic = time.perf_counter()
    with mic as source:
        audio = r.listen(source)
    input = prepare_model_input([read_audio(io.BytesIO(audio.get_wav_data()))], device=device)
    toc = time.perf_counter()
    print(f"Time spent recording: {toc-tic:0.4f} seconds")
    return input


def transcribe(input) -> None:
    print("Transcribing...")
    tic = time.perf_counter()
    output = model(input)[0]
    result = decoder(output.cuda())
    type(result)
    print(result)
    toc = time.perf_counter()
    print(f"Time spent transcribing: {toc-tic:0.4f} seconds")
    

def recordAndTranscribe() -> None:
    while True:
        input = record()
        transcribe(input)
