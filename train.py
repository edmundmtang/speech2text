import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from decoder import GreedyDecode
from time import time

from utilities import TextProcess
textprocess = TextProcess()

class TrainingModule():

    def __init__(self, model, data_loader, params, device=torch.device('cpu')):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_fn = nn.CTCLoss(blank=27, zero_infinity=params.zero_infinity)
        self.optimizer = optim.AdamW(model.parameters())
        self.device = device
        self.scheduler = optim.lr_scheduler.CyclicLR(
            self.optimizer,
            mode = params.scheduler_mode,
            base_lr = params.base_lr,
            max_lr = params.max_lr,
            step_size_up = params.step_size_up,
            cycle_momentum=False,
            )
            

    def step(self, batch):
        spectrograms, labels, input_lens, label_lens = batch
        spectrograms = spectrograms.to(self.device)
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device) , hidden[1].to(self.device)

        # zero gradients each batch
        self.optimizer.zero_grad()

        # make predictions for this batch
        outputs, _ = self.model(spectrograms, (hn, c0))

        # compute loss and its gradients
        loss = self.loss_fn(F.log_softmax(outputs, dim=2), labels, input_lens, label_lens)
        loss.backward()

        # adjust learning weights
        self.optimizer.step()
        self.scheduler.step()

        return outputs, loss.item()

    def train_num_batches(self, num, debugging=True):
        start_time = time()
        for i, batch in enumerate(self.data_loader):
            outputs, loss = self.step(batch)
            
            current_time = time()
            average_time = (current_time - start_time)/(i+1)
            if average_time > 1:
                rate_str = ' | {:.2f} s/it'.format(average_time)
            else:
                rate_str = ' | {:.2f} it/s'.format(1/average_time)

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            lr_str = ' | lr: {:.5f}'.format(lr)
            
            progress_str = 'Batch {}/{} loss: {}'.format(i + 1, num, loss) + rate_str + lr_str
            
            if debugging:
                print('Target string: ', textprocess.int_to_text_seq(batch[1][0].tolist()))
                print('Output string: ', GreedyDecode(outputs[:,0,:], dim=1, collapse_repeated=False))
                print('--------------------------------')
                print(progress_str)
                print('================================')
            else:
                print(progress_str)
            if i+1 == num:
                return average_time # average time per batch
    
    def train_one_epoch(self, epoch_index, batch_list, loss_list, batches=100, debugging=False):
        pass
        
        

