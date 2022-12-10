import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
import utilities


class LogMelSpec(nn.Module):
    
    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
                            sample_rate=sample_rate, n_mels=n_mels,
                            win_length=win_length, hop_length=hop_length)
        
    def forward(self, x):
        x = self.transform(x) # mel spectrogram
        x = np.log(x + 1e-14) # logarithm, add small value to avoid divergence
        return x

class DataSet(torch.utils.data.Dataset):
        
    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, specaug_policy, time_mask, freq_mask,
                 valid=False, shuffle=True, text_to_int=True, log_ex=True, verbose=False):
        self.log_ex = log_ex
        self.text_process = utilities.TextProcess()

        if verbose:
            print("Loading data json file from", json_path)

        self.data = pd.read_json(json_path, lines=True)
        
        if valid: # 
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
            
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                #SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask) # To-do: Add spec augment
            ) 
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        try:
            file_path = self.data.key.iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            label = self.text_process.text_to_int_seq(self.data['text2'].iloc[idx])
            spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)
            if spec_len < label_len:
                raise Exception('spectrogram len is bigger then label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s'%file_path)
            #if spectrogram.shape[2] > 1650:
            #    raise Exception('spectrogram too big. size %s'%spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s'%file_path)
        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)  
                
        return spectrogram, label, spec_len, label_len
   
    def describe(self):
        return self.data.describe()
    
def collate_fn_padd(data):
    '''
    Padds batch of variable length
    
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitary tensors
    '''
    spectrograms = []
    labels = []
    input_lens = []
    label_lens = []
    for (spectrogram, label, input_len, label_len) in data:
        if spectrogram is None:
            continue
        spectrograms.append(spectrogram.squeeze(0).transpose(0,1))
        labels.append(torch.Tensor(label))
        input_lens.append(input_len)
        label_lens.append(label_len)
        
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2,3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lens = input_lens
    label_lens = label_lens
    
    return spectrograms, labels, input_lens, label_lens
