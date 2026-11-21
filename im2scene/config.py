import argparse

import yaml
import torch
import torch.optim as optim
from im2scene import data
from im2scene import gan2d, giraffe, sdf
from im2scene.checkpoints import CheckpointIO
from im2scene.encoder.ranger import Ranger
from im2scene.sdf.models.sdf_utils import MultiResolutionDataset
import logging
import os
from torchvision import transforms

# method directory; for this project we only use giraffe
method_dict = {
    'gan2d': gan2d,
    'giraffe': giraffe,
    'sdf': sdf
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, len_dataset=0, args=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
        i_embed (int): set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical
        i_embed_views (int): set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical
    '''
    method = cfg['method']  # 默认使用giraffe方法
    # method_dict[]指向giraffe包
    model = method_dict[method].config.get_model(
        cfg, device=device, len_dataset=len_dataset, args=args)
    return model


def set_logger(cfg):
    logfile = os.path.join(cfg['training']['out_dir'],
                           cfg['training']['logfile'])
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logfile,
        filemode='a',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)


# Trainer
def get_trainer(model, optimizer_e, optimizer, optimizer_d, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer_e (optimizer): encoder optimizer object
        optimizer (optimizer): pytorch optimizer
        optimizer_d (optimizer): discriminator optimizer object
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    set_logger(cfg)
    # 跳转至  im2scene/giraffe/config/get_trainer
    trainer = method_dict[method].config.get_trainer(
        model, optimizer_e, optimizer, optimizer_d, cfg, device)
    return trainer


# Renderer
def get_renderer(model, cfg, device):
    ''' Returns a render instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    renderer = method_dict[method].config.get_renderer(model, cfg, device)
    return renderer


def get_dataset(cfg, args, **kwargs):
    ''' Returns a dataset instance.

    Args:
        cfg (dict): config dictionary
        mode (string): which mode is used (train / val /test / render)
        return_idx (bool): whether to return model index
        return_category (bool): whether to return model category
    '''
    # Get fields with cfg
    dataset_name = cfg['data']['dataset_name']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']
    img_size = cfg['data']['img_size']
    data_method = cfg['method']

    if dataset_name == 'lsun':
        dataset = data.LSUNClass(dataset_folder, categories, size=img_size,
                                 random_crop=cfg['data']['random_crop'],
                                 use_tanh_range=cfg['data']['use_tanh_range'],
                                 )
    elif data_method == 'sdf':
        opt = args.sdf_opt
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
        dataset = MultiResolutionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                         opt.model.renderer_spatial_output_dim)
    else:
        dataset = data.ImagesDataset(
            dataset_folder, size=img_size,
            use_tanh_range=cfg['data']['use_tanh_range'],  # 是否将如下缩放到[-1,1]
            celebA_center_crop=cfg['data']['celebA_center_crop'],  # 是否对celeba和celeba_hq进行数据裁剪
            random_crop=cfg['data']['random_crop'],  # 是否随机裁剪
        )
    return dataset


def get_optimizer(cfg, model, args):
    # giraffe使用RMSprop，sdf使用adam
    op = optim.RMSprop if cfg['training']['optimizer'] == 'RMSprop' else optim.Adam
    optimizer_kwargs = cfg['training']['optimizer_kwargs']
    lr = cfg['training']['learning_rate']  # 生成器学习率
    lr_d = cfg['training']['learning_rate_d']  # 判别器学习率

    # hasattr  判断对象是否有某个属性
    # 获取模型的参数，定义优化器，默认使用RMSprop
    if hasattr(model, "encoder") and model.encoder is not None:
        parameters_e = model.encoder.parameters()
        if args.vae == 1:
            optimizer_e = op(parameters_e, lr=0.0005)
        else:
            optimizer_e = Ranger(parameters_e, lr=0.0001)
    else:
        optimizer_e = None

    if args.sdf == 1:
        if args.need_train_vol_render:
            # 训练体渲染器的优化器
            optimizer = optim.Adam(model.generator.parameters(), lr=2e-5, betas=(0, 0.9))
            optimizer_d = optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0, 0.9))
        else:
            # 训练stylegan的优化器
            g_reg_ratio = args.sdf_opt.training.g_reg_every / (
                        args.sdf_opt.training.g_reg_every + 1) if args.sdf_opt.training.g_reg_every > 0 else 1
            d_reg_ratio = args.sdf_opt.training.d_reg_every / (args.sdf_opt.training.d_reg_every + 1)
            params_g = []
            params_dict_g = dict(model.generator.named_parameters())
            for key, value in params_dict_g.items():
                decoder_cond = ('decoder' in key)
                if decoder_cond:
                    params_g += [{'params': [value], 'lr': args.sdf_opt.training.lr * g_reg_ratio}]

            optimizer = optim.Adam(params_g,  # generator.parameters(),
                                 lr=args.sdf_opt.training.lr * g_reg_ratio,
                                 betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
            optimizer_d = optim.Adam(model.discriminator.parameters(),
                                 lr=args.sdf_opt.training.lr * d_reg_ratio,  # * g_d_ratio,
                                 betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

    else:
        if hasattr(model, "generator") and model.generator is not None:
            parameters_g = model.generator.parameters()
        else:
            parameters_g = list(model.decoder.parameters())
        optimizer = op(parameters_g, lr=lr, **optimizer_kwargs)  # lr = 0.0005

        if hasattr(model, "discriminator") and model.discriminator is not None:
            parameters_d = model.discriminator.parameters()
            optimizer_d = op(parameters_d, lr=lr_d)  # lr_d = 0.0001
        else:
            optimizer_d = None

    return optimizer_e, optimizer, optimizer_d


def load_model(model, optimizer_e, optimizer, optimizer_d, cfg, use_sdf, opt):

    out_dir = cfg['training']['out_dir']

    if use_sdf:
        # 使用sdf
        from im2scene.sdf.models.sdf_utils import get_rank
        # 读取checkpoint
        if opt.experiment.continue_training and opt.experiment.ckpt is not None:

            if get_rank() == 0:
                print("load model:", opt.experiment.ckpt)
            ckpt_path = os.path.join(opt.training.checkpoints_dir,
                                     opt.experiment.expname,
                                     'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

            try:
                opt.training.start_iter = int(opt.experiment.ckpt) + 1

            except ValueError:
                pass

            model.generator.load_state_dict(ckpt["g"])
            model.discriminator.load_state_dict(ckpt["d"])
            model.g_ema.load_state_dict(ckpt["g_ema"])
            # if "g_optim" in ckpt.keys():
            #     g_optim.load_state_dict(ckpt["g_optim"])
            #     d_optim.load_state_dict(ckpt["d_optim"])

        sphere_init_path = './pretrained_renderer/sphere_init.pt'
        if opt.training.no_sphere_init:
            # 不使用 sphere SDF 初始化渲染器
            opt.training.sphere_init = False
        elif not opt.experiment.continue_training and opt.training.with_sdf and os.path.isfile(sphere_init_path):
            # 如果是第一次训练，且使用SDF，且 sphere SDF 的文件存在
            if get_rank() == 0:
                print("loading sphere initialized model")
            ckpt = torch.load(sphere_init_path, map_location=lambda storage, loc: storage)
            model.generator.load_state_dict(ckpt["g"])
            model.discriminator.load_state_dict(ckpt["d"])
            model.g_ema.load_state_dict(ckpt["g_ema"])
            opt.training.sphere_init = False
        else:
            opt.training.sphere_init = True
    else:
        # 使用原版
        checkpoint_io = CheckpointIO(out_dir, model=model,
                                     optimizer_e=optimizer_e,
                                     optimizer=optimizer,
                                     optimizer_d=optimizer_d)

        # 尝试读取本地模型
        try:
            load_dict = checkpoint_io.load('model.pt')
            print("Loaded model checkpoint.")
        except FileExistsError:
            load_dict = dict()
            print("No model checkpoint found.")


def get_params():
    parser = argparse.ArgumentParser(
        description='Evaluate a GIRAFFE model.'
    )
    parser.add_argument('--config', type=str, default="configs/256res/ffhq_256_sdf.yaml",
                        help='Path to config file.')  # 配置文件
    # 设置位置x和d编码方式
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=0,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    # hash编码参数
    parser.add_argument("--finest_res", type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size", type=int, default=19,
                        help='log2 of hashmap size')
    # 是否使用小型网络
    parser.add_argument("--small_net", type=int, default=0,
                        help='set 1 for small net, 0 for default')
    # 是否使用vae
    parser.add_argument("--vae", type=int, default=0,
                        help='set 1 for suse vae, 0 for default')
    # 是否使用sdf网络
    parser.add_argument("--sdf", type=int, default=1,
                        help='set 1 for use sdf net, 0 for default')
    parser.add_argument("--ngp", type=int, default=0,
                        help='set 1 for use ngp net, 0 for default')

    # 消融实验，使用普通线性层
    parser.add_argument("--fc", type=int, default=0,
                        help='set 1 for use ngp net, 0 for default')
    # 消融实验，不使用额外监督
    parser.add_argument("--wod", type=int, default=0,
                        help='set 1 for do not use Dvol')

    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')  # 不使用gpu
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of '
                             'seconds with exit code 2.')

    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    return args, cfg