# Speech to Text

import os
import multiprocessing as mp
from threading import Thread

import io
import time
import keyboard

import torch
#import torchaudio
import librosa
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import speech_recognition as sr

def record(queue: mp.Queue, ns) -> None:
    # Start microphone
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=2)
    print("Microphone process initialized.")
    while True:
        if ns.is_recording:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit = ns.phrase_time_limit)
            queue.put(audio)

def prepare(audio: sr.AudioData, sampling_rate: int):
    flac_data, sampling_rate = librosa.load(io.BytesIO(audio.get_flac_data()), sr=sampling_rate)
    return processor(flac_data, sampling_rate=sampling_rate, return_tensors="pt")

def transcribe(input):
    generated_ids = model.generate(input["input_features"], attention_mask=input["attention_mask"])
    result = processor.batch_decode(generated_ids)[0]
    return result if result else ""

def prepareAndTranscribe(queue: mp.Queue, ns):
    print("Transcription thread initialized.")
    while True:
        time.sleep(0.5)
        if not queue.empty():
            audio = queue.get() # this will empty out the queue if is_recording = False
            if ns.is_recording:
                input = prepare(audio, ns.sampling_rate)
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("...")
    # Import model
    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
    print("...")
    mgr = mp.Manager()
    ns = mgr.Namespace()
    ns.is_recording = False
    ns.phrase_time_limit = 3
    ns.sampling_rate = 16000
    print("...")
    audio_q = mp.Queue() # audio to be transcribed
    record_process = mp.Process(target=record, args=(audio_q, ns,))
    record_process.daemon = True
    transcribe_thread = Thread(target=prepareAndTranscribe, args=(audio_q, ns,))
    transcribe_thread.daemon = True
    print('...')
    record_process.start()
    transcribe_thread.start()
    keyboard.add_hotkey("ctrl+backslash", lambda: toggleRecord(ns))

    time.sleep(5)

    input("Press Enter to exit.\n")
    
