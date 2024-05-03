from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 1
config.TRAIN.lr_init = 1e-6
config.TRAIN.beta1 = 0.9


## initialize G
config.TRAIN.n_epoch_init = 1
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 4
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2) #2

## train set location
config.TRAIN.hr_img_path = '/home/nyma/PycharmProjects/JointSSD/Vedai_Augmented/'
config.TRAIN.lr_img_path = '/home/nyma/PycharmProjects/JointSSD/Vedai_Augmented(128x128)/'
config.TRAIN.hr2_img_path = '/home/nyma/PycharmProjects/JointSSD/Vedai_Augmented(256x256)/'
config.TRAIN.new_img_path ='/home/nyma/PycharmProjects/JointSSD/NewImage'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = '/home/nyma/PycharmProjects/JointSSD/Vedai_Test(512x512)'
config.VALID.lr_img_path = '/home/nyma/PycharmProjects/JointSSD/Vedai_Test_LR(128x128)'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
