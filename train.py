import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from decoder import GreedyDecode
from datetime import date
from tqdm import tqdm
import shutil

from utilities import TextProcess
textprocess = TextProcess()

class TrainingModule():

    def __init__(self, model, data_loader, params, device=torch.device('cpu')):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_fn = nn.CTCLoss(blank=27, zero_infinity=False)
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

        self.verbose = params.verbose
        self.print_freq = params.print_freq
        self.sample_freq = params.sample_freq
        
        self.save_freq = params.save_freq
        self.save_dir = params.save_dir
        self.loss_list = []
        self.batch_list = []
        self.lr_list = []
        
        if params.resume:
            print('Loading previous state.')
            self.load_checkpoint(params.load_path)

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
            
    def train(self, verbose=False):
        lowest_loss = float('inf')
        if self.batch_list:
            old_index = self.batch_list[-1]
        else:
            old_index = 0
        for i, batch in tqdm(enumerate(self.data_loader), total = int(len(self.data_loader))):
            outputs, loss = self.step(batch)
            full_index = i+old_index+1
            if verbose and i % self.print_freq == 0:
                print('\nLoss: ', loss)
                print('Target string: ', textprocess.int_to_text_seq(batch[1][0].tolist()))
                print('Output string: ', GreedyDecode(outputs[:,0,:], dim=1, collapse_repeated=False))
                print('================================')
            if full_index % self.sample_freq == 1:
                self.batch_list.append(full_index)
                self.loss_list.append(loss)
                self.lr_list.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            if (full_index) % self.save_freq == 1:
                if loss < lowest_loss:
                    lowest_loss = loss
                    is_best = True
                save_name = 'ckp_' + str(date.today()) + '_b-' + str(full_index).zfill(7) + '.pth'
                self.save_checkpoint(is_best, save_name)
                is_best = False
        if loss < lowest_loss:
            lowest_loss = loss
            is_best = True
        save_name = 'ckp_' + str(date.today()) + '_b-' + str(full_index).zfill(7) + '.pth'
        self.save_checkpoint(is_best, save_name)
                
    def save_checkpoint(self, is_best, save_name):
        checkpoint = {
            'loss_list': self.loss_list,
            'batch_list': self.batch_list,
            'batch_count': int(len(self.data_loader)),
            'lr_list': self.lr_list,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            }
        save_path = self.save_dir + '/checkpoints/' + save_name
        torch.save(checkpoint, save_path)
        if is_best:
            best_path = self.save_dir + '/best/' + save_name
            shutil.copyfile(save_path, best_path)

    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)
        self.loss_list = checkpoint['loss_list']
        self.batch_list = checkpoint['batch_list']
        self.lr_list = checkpoint['lr_list']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
