# trying to pickle torch model
import os
import multiprocessing as mp

import torch
import torchaudio

import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


print("Preparing model...")

# Choose device to run model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Import model
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

print("Model prepared!")

class Foo:
    def __getstate__(self):
        self.a = torch.tensor([1,2,3])
        return self.a

    def __setstate__(self, state):
        print('lalala')
    
f = Foo()

f_str = pickle.dumps(f)

f2 = pickle.loads(f_str)
