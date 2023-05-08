import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.model.backbones.ResUnet_refine import *
from custom.model.model_head import *
from custom.model.model_network import *
from custom.utils.common_tools import *
from custom.dataset.dataset import MyDataset

class network_cfg:
    # img
    patch_size = (64, 160, 160)
    win_clip = None

    # network
    network = Model_Network(
        backbone = ResUnet_refine(in_ch=1,channels=32, blocks=3),
        head = Model_Head(in_channels=32,scale_factor=(2.0, 2.0, 2.0),num_class=5),
        apply_sync_batchnorm=False,
    )

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.lst",
        constant_shift = 2,
        shift_range = 0,
        transforms = Compose([
            to_tensor(),
            normlize(win_clip=win_clip),
            random_rotate3d(prob=0.5,
                            x_theta_range=[-30,30],
                            y_theta_range=[-30,30],
                            z_theta_range=[-30,30]),
            resize(patch_size)
            ])
        )
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.lst",
        constant_shift = 2,
        shift_range = 0,
        transforms = Compose([
            to_tensor(),           
            normlize(win_clip=win_clip),
            resize(patch_size)
            ])
        )
    
    # dataloader
    batchsize = 4
    shuffle = True
    num_workers = 8
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 1e-4

    # scheduler
    milestones = [50,100]
    gamma = 0.1
    warmup_factor = 0.1
    warmup_iters = 50
    warmup_method = "linear"
    last_epoch = -1

    # debug
    valid_interval = 5
    log_dir = work_dir + "/Logs"
    checkpoints_dir = work_dir + '/checkpoints/v1'
    checkpoint_save_interval = 2
    total_epochs = 10
    load_from = ''

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
