import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

transcription = processor.batch_decode(generated_ids)
print(transcription)

import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    print("Recording...")
    audio = recognizer.listen(source)

print("Recorded!")

flac_data = audio.get_flac_data()



import librosa
import io

sampling_rate = 16000
flac_data, sr = librosa.load(io.BytesIO(audio.get_flac_data()), sr=16000)
model_input = processor(flac_data, sampling_rate=sampling_rate, return_tensors="pt")

gen_ids = model.generate(model_input["input_features"], attention_mask=model_input["attention_mask"])

transcription = processor.batch_decode(gen_ids)
print(transcription)
