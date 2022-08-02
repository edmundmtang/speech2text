# Speech to Text

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pyaudio
import speech_recognition as sr

# Choose device to run model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Import model
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details


mic = sr.Microphone()

with mic as source:
    # need some io.BytesIO?
    waveform, sample_rate = torch.load(mic)
