# Speech to Text

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import io
import torch
import pyaudio
import speech_recognition as sr
from time import sleep

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

class Recorder():

    sampling_rate = 44100
    num_channels = 1
    sample_width = 4 # The width of each sample in bytes. Each group of ``sample_width`` bytes represents a single audio sample. 

    def pyaudio_stream_callback(self, in_data, frame_count, time_info, status):
        self.raw_audio_bytes_array.extend(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):

        self.raw_audio_bytes_array = bytearray()

        pa = pyaudio.PyAudio()
        self.pyaudio_stream = pa.open(format=pyaudio.paInt16,
                                      channels=self.num_channels,
                                      rate=self.sampling_rate,
                                      input=True,
                                      stream_callback=self.pyaudio_stream_callback)

        self.pyaudio_stream.start_stream()

    def stop_recording(self):

        self.pyaudio_stream.stop_stream()
        self.pyaudio_stream.close()

        speech_recognition_audio_data = sr.AudioData(self.raw_audio_bytes_array,
                                                                     self.sampling_rate,
                                                                     self.sample_width)
        return speech_recognition_audio_data


recorder = Recorder()

# start recording
recorder.start_recording()
print("-----Start Recording-----")

# say something interesting...
sleep(3)

# stop recording
speech_recognition_audio_data = recorder.stop_recording()
print("-----Stop Recording-----")

# convert the audio represented by the ``AudioData`` object to
# a byte string representing the contents of a WAV file
wav_data = speech_recognition_audio_data.get_wav_data()
