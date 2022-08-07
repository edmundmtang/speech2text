# Speech to Text

import os
import multiprocessing as mp
from threading import Thread


import io

import pyaudio # Soundcard audio I/O access library
import wave # Python 3 module for reading / writing simple .wav files

import torch
import torchaudio

import speech_recognition as sr

import keyboard

import time




def record(queue: mp.Queue, ns):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while ns.is_recording:
        print("Recording...")
        with mic as source:
            audio = recognizer.listen(source, phrase_time_limit = 3)
        queue.put(audio)

def prepare(audio: sr.AudioData):
    return prepare_model_input([read_audio(io.BytesIO(audio.get_wav_data()))], device=device)
    
def transcribe(input: torch.Tensor) -> str:
    output = model(input)[0]
    result = decoder(output.cuda())
    if result:
        return result
    return ""

def prepareAndTranscribe(queue, ns):
    
    while ns.is_recording:
        print("Transcribing...")
        time.sleep(0.1)
        if queue.empty():
            continue
        audio = queue.get()
        input = prepare(audio)
        result = transcribe(input)
        print(result)
    

# Create hotkey for recording

def toggleRecording(ns) -> None:
    print("FLIP")
    ns.is_recording ^= True # flip

        

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Choose device to run model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("Preparing model...")
    
    # Import model
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language='en', # also available 'de', 'es'
                                           device=device)
    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils  # see function signature for details

    print("Model prepared!")
    
    print("Preparing manager...")
    
    mgr = mp.Manager()
    ns = mgr.Namespace()
    ns.is_recording = False

    print("Manager prepared!")


    audio_q = mp.Queue() # audio to be transcribed

    record_process = mp.Process(target=record, args=(audio_q, ns,))
    transcribe_thread = Thread(target=prepareAndTranscribe, args=(audio_q, ns,))

    record_process.start()
    transcribe_thread.start()
    
    keyboard.add_hotkey("ctrl+plus", lambda: toggleRecording(ns))

    print(record_process)
    print(transcribe_thread)
    input("Press Enter to continue.\n")

    ns.is_recording = False

    while not audio_q.empty():
        print("OH")
        audio_q.get()
    print("AH")
    record_process.join()
    print("1")
    transcribe_thread.join()
    print("2")
