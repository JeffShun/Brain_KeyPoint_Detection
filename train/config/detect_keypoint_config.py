from ..custom.model.backbones.ResUnet_refine import *
from ..custom.model.model_head import *
from ..custom.model.model_network import *
from ..custom.utils.common_tools import *
from ..custom.dataset.dataset import MyDataset

class network_cfg:
    # img
    patch_size = [128, 128, 128]
    win_clip = None

    # network
    network = Model_Network(
        backbone = ResUnet_refine(in_ch=1,channels=32, blocks=3),
        head = Model_Head(in_channels=32,scale_factor=(2.0, 2.0, 2.0),num_class=5),
        apply_sync_batchnorm=True,
    )

    # dataset
    train_dataset = MyDataset(
        dst_list_file = "./train_data/processed_data/train.lst",
        patch_size = patch_size,
        shift_range = 5,
        transform = Compose([
            to_tensor(),
            normlize(win_clip=win_clip),
            random_rotate3d(prob=0.5,
                            x_theta_range=[-30,30],
                            y_theta_range=[-30,30],
                            z_theta_range=[-30,30])
            ])
        )
    valid_dataset = MyDataset(
        dst_list_file = "./train_data/processed_data/valid.lst",
        patch_size = patch_size,
        shift_range = 5,
        transform = Compose([
            to_tensor(),
            normlize(win_clip=win_clip)
            ])
        )
    
    # dataloader
    batchsize = 4
    shuffle = True
    num_workers = 0
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 1e-4
    lr_scheduler_step = 50,
    lr_scheduler_gamma = 0.1

    # debug
    valid_interval = 5
    debug_file = "./Logs/debug.log"

    # others
    checkpoints_dir = './checkpoints/v1'
    checkpoint_save_interval = 2
    gpu = 0
    total_epochs = 100
    load_from = None

