# Speech to Text

import os
import multiprocessing as mp
from threading import Thread

import io
import time
import keyboard

import torch
import torchaudio

import speech_recognition as sr

def record(queue: mp.Queue, ns) -> None:
    # Start microphone
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Microphone process initialized.")
    while True:
        time.sleep(0.5)
        if ns.is_recording:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=ns.phrase_time_limit)
            queue.put(audio)

def prepare(audio: sr.AudioData):
    return prepare_model_input([read_audio(io.BytesIO(audio.get_wav_data()))], device=device)

def transcribe(input: torch.Tensor) -> str:
    output = model(input)[0]
    result = decoder(output.cuda())
    return result if result else ""

def prepareAndTranscribe(queue: mp.Queue, ns):
    print("Transcription thread initialized")
    while True:
        time.sleep(0.5)
        if not queue.empty():
            audio = queue.get() # this will empty out the queue if is_recording = False
            if ns.is_recording:
                input = prepare(audio)
                result = transcribe(input)
                if result != "":
                    print(result)

def toggleRecord(ns) -> None:
    print("Toggle")
    ns.is_recording ^= True

if __name__ == "__main__":
    print("Starting up...")

    # Choose device to run model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('...')
    # Import model
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language='en', # also available 'de', 'es'
                                           device=device)

    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils  # see function signature for details
    print('...')
    mgr = mp.Manager()
    ns = mgr.Namespace()
    ns.is_recording = False
    ns.phrase_time_limit = 3
    print('...')
    audio_q = mp.Queue() # audio to be transcribed
    record_process = mp.Process(target=record, args=(audio_q, ns,))
    record_process.daemon = True
    transcribe_thread = Thread(target=prepareAndTranscribe, args=(audio_q, ns,))
    transcribe_thread.daemon = True
    print('...')
    record_process.start()
    transcribe_thread.start()
    keyboard.add_hotkey("ctrl+plus", lambda: toggleRecord(ns))

    print("Ready!")

    input("Press Enter to continue.\n")
