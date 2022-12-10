import torch.nn as nn
import torch.optim as optim

class trainingModule():

    def __init__(self, model, data_loader, params):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = nn.CTCLoss(blank=27, zero_infinity=params.zero_infinity)
        self.optimizer = optim.AdamW(model.parameters(), params.learning_rate)
        #self.scheduler = optim.lr_scheduler...

    def step(self, batch):
        spectrograms = labels, input_lens, label_lens = batch
        bs = spectrograms.shape[0]
        hidden = model._init_hidden(bs)
        hn, c0 = hidden[0], hidden[1]

        # zero gradients each batch
        optimizer.zero_grad()

        # make predictions for this batch
        outputs, _ = model(spectrograms, (hn, c0))

        # compute loss and its gradients
        loss = self.loss_fn(F.log_softmax(outputs, dim=2), labels, input_lens, label_lens)
        loss.backward()

        # adjust learning weights
        optimizer.step()

        return outputs, loss.item()

    def train_num_batches(model, num, debugging=True)
        
    
    def train_one_epoch(model, epoch_index, batch_list, loss_list, batches=100, debugging=False):
        pass
        
        

