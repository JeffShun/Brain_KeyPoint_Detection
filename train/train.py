import warnings
warnings.filterwarnings('ignore')
import os
from config.detect_keypoint_config import network_cfg
import torch
from torch import optim
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import time

def train():
    net = network_cfg.network
    net.weight_init()
    if network_cfg.load_from is not None:
        net.load_state_dict(torch.load(network_cfg.load_from))
    net.train()
    train_dataset = network_cfg.train_dataset
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=network_cfg.batchsize, 
                                  shuffle=network_cfg.shuffle,
                                  num_workers=network_cfg.num_workers, 
                                  drop_last=network_cfg.drop_last)
    valid_dataset = network_cfg.valid_dataset
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=network_cfg.batchsize, 
                                  shuffle=False,
                                  num_workers=network_cfg.num_workers, 
                                  drop_last=network_cfg.drop_last)
    
    optimizer = optim.Adam(params=net.parameters(), lr=network_cfg.lr, weight_decay=network_cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=network_cfg.lr_scheduler_step, 
                                          gamma=network_cfg.lr_scheduler_gamma)
    time_start=time.time()
    for epoch in range(network_cfg.total_epochs):
        train_loss = dict()
        valid_loss = dict()
        print("Epoch: {}".format(epoch+1))
        #Training Step!
        for ii, (train_data, train_label) in enumerate(train_dataloader):
            print('\rTraining Step:    {}{}/{}'.format('▉'*((ii+1)*40//len(train_dataset)),ii+1,len(train_dataset)),end='')
            train_data = V(train_data.float()).cuda()
            train_label = V(train_label.float()).cuda()
            optimizer.zero_grad()
            t_loss = net(train_data, train_label)
            loss_all = V(torch.zeros(1)).cuda()
            for loss_item, loss_val in t_loss:
                if loss_item not in train_loss:
                    train_loss[loss_item] = loss_val
                else:
                    train_loss[loss_item] += loss_val
                loss_all += loss_val
            loss_all.backward()
            optimizer.step()
            scheduler.step()
        for loss_item, loss_val in train_loss:
            train_loss[loss_item] /= (ii+1)

        # Valid Step!
        print('')
        if (epoch+1) // network_cfg.valid_interval == 0:
            for ii, (valid_data,valid_label) in enumerate(valid_dataloader):
                print('\rValidating Step:  {}{}/{}'.format('▉'*((ii+1)*40//len(valid_dataset)),ii+1,len(valid_dataset)),end='')
                valid_data = V(valid_data.float()).cuda()
                valid_label = V(valid_label.float()).cuda()
                v_loss = net.valid_forward(valid_data, valid_label)
                loss_all = V(torch.zeros(1)).cuda()
                for loss_item, loss_val in v_loss:
                    if loss_item not in valid_loss:
                        valid_loss[loss_item] = loss_val
                    else:
                        valid_loss[loss_item] += loss_val                
            for loss_item, loss_val in valid_loss:
                valid_loss[loss_item] /= (ii+1)

        time_end=time.time()
        time_consuming = (time_end-time_start)//60
        print('')
        print("Training: {},   Validating: {},    Time Cost: {}".format(train_loss, valid_loss, time_consuming))
        if (epoch+1) // network_cfg.checkpoint_save_interval == 0:
            torch.save(net.state_dict(), network_cfg.checkpoints_dir+"/{}.pth".format(epoch+1))


if __name__ == '__main__':
	torch.cuda.set_device(network_cfg.gpu)
	train()
