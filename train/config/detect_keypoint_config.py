import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.model.backbones.Cascaded_ResUnet_SPP import *
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
        backbone = Cascaded_ResUnet_SPP(in_ch=1,channels=12, blocks=3),
        head = Model_Head(in_channels=12, num_class=5),
        apply_sync_batchnorm=False,
    )

    # loss function
    loss_f = MergeLoss(point_radius=[3, 1])

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.lst",
        constant_shift = 2,
        shift_range = 0,
        transforms = Compose([
            to_tensor(),
            normlize(win_clip=win_clip),
            random_rotate3d(x_theta_range=[-20,20],
                            y_theta_range=[-20,20],
                            z_theta_range=[-20,20],
                            prob=0.5),
            random_gamma_transform(gamma_range=[0.7,1.3], prob=0.5),
            random_apply_mosaic(prob=0.2, mosaic_size=40, mosaic_num=4),
            random_add_gaussian_noise(prob=0.2, mean=0, std=0.02),
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
    batchsize = 3
    shuffle = True
    num_workers = 8
    drop_last = False

    # optimizer
    lr = 1e-3
    weight_decay = 5e-4

    # scheduler
    milestones = [50,80]
    gamma = 0.1
    warmup_factor = 0.1
    warmup_iters = 50
    warmup_method = "linear"
    last_epoch = -1

    # debug
    valid_interval = 2
    log_dir = work_dir + "/Logs"
    checkpoints_dir = work_dir + '/checkpoints/v5'
    checkpoint_save_interval = 2
    total_epochs = 150
    load_from = work_dir + '/checkpoints/v1/40.pth'

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
