import data
from torch.utils.data import DataLoader
from model import AcousticModel

class params():
    # ====== Hyper Parameters ====== #
    num_classes = 28
    # n_feats = 81 # decided earlier based on how mel spectrograms were generated
    dropout = 0.1
    dense_size = 128
    hidden_size = 1024
    num_layers = 1

    max_spec_len = 4096
    max_label_len = 256

    batch_size = 16
    learning_rate = 0.001
    num_workers = 2
    # ============================== #

    # ===== Sample Parameters ====== #
    sample_rate = 8000
    n_feats = 81
    specaug_rate = 0.5 # To-do: specaug_rate, specaug_policy, time_mask, and freq_mask do nothing until spec augment is implemented
    specaug_policy = 3
    time_mask = 70
    freq_mask = 15
    shuffle = True
    pin_memory = True
    train_path = 'F:/cv-corpus-11.0-2022-09-21/en/train.json'
    test_path = 'F:/cv-corpus-11.0-2022-09-21/en/test.json'
    # ============================== #

# Load Data Sets
train_dataset = data.DataSet(params.train_path,
                             params.sample_rate,
                             params.n_feats,
                             params.specaug_rate,
                             params.specaug_policy,
                             params.time_mask,
                             params.freq_mask,
                             shuffle=params.shuffle)
val_dataset = data.DataSet(params.test_path,
                           params.sample_rate,
                           params.n_feats,
                           params.specaug_rate,
                           params.specaug_policy,
                           params.time_mask,
                           params.freq_mask,
                           shuffle=params.shuffle)

# Construct Data Loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=params.batch_size,
                          num_workers=params.num_workers,
                          collate_fn=data.collate_fn_padd,
                          pin_memory = params.pin_memory)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=params.batch_size,
                          num_workers=params.num_workers,
                          collate_fn=data.collate_fn_padd,
                          pin_memory = params.pin_memory)
# Build model

    


