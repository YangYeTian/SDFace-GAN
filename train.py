import torch
import numpy as np
import os
import argparse
import logging
from im2scene import config
from im2scene.sdf.models.sdf_utils import data_sampler
from im2scene.training_utils import train_giraffe, train_vol_render, get_vol_render_opt, train_full_pipeline, \
    train_encoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    """
    1.读取模型各项参数
    """

    logger_py = logging.getLogger(__name__)
    np.random.seed(0)
    torch.manual_seed(0)

    # Arguments
    parser = argparse.ArgumentParser(description='Train a GIRAFFE model.')
    # parser.add_argument('--config', type=str, default="configs/64res/celeba_64_sdf.yaml", help='Path to config file.')  # 配置文件
    parser.add_argument('--config', type=str, default="configs/256res/ffhq_256_sdf.yaml", help='Path to config file.')  # 配置文件
    # 设置位置x和d编码方式
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=0,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    # hash编码参数
    parser.add_argument("--finest_res",   type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19,
                        help='log2 of hashmap size')
    # 是否使用小型网络
    parser.add_argument("--small_net", type=int, default=0,
                        help='set 1 for small net, 0 for default')
    # 是否使用vae
    parser.add_argument("--vae", type=int, default=0,
                        help='set 1 for suse vae, 0 for default')
    # 是否使用sdf网络
    parser.add_argument("--sdf", type=int, default=0,
                        help='set 1 for use sdf net, 0 for default')
    parser.add_argument("--ngp", type=int, default=0,
                        help='set 1 for use ngp net, 0 for default')

    # 消融实验，使用普通线性层
    parser.add_argument("--fc", type=int, default=0,
                        help='set 1 for use ngp net, 0 for default')
    # 消融实验，不使用额外监督
    parser.add_argument("--wod", type=int, default=0,
                        help='set 1 for do not use Dvol')
    parser.add_argument("--psp", type=int, default=0,
                        help='set 1 for use psp encoder net, 0 for default')

    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')  # 不使用gpu
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of '
                             'seconds with exit code 2.')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')  # 读取训练初始配置文件
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    if args.sdf == 1:
        expname = cfg['training']['out_dir'].split("/")[1]
        # 体渲染模型路径 判断体渲染器是否训练完毕
        vol_render_ckpt = os.path.join("./out", expname, 'vol_renderer.pt')
        full_pipeline_ckpt = os.path.join("./out", expname, 'full_pipeline.pt')
        if not os.path.exists(vol_render_ckpt):
            need_train_vol_render = True
        else:
            need_train_vol_render = False
        if not os.path.exists(full_pipeline_ckpt):
            need_train_full_pipeline = True
        else:
            need_train_full_pipeline = False
        if args.wod:
            need_train_vol_render = False
            need_train_full_pipeline = True

        args.need_train_vol_render = need_train_vol_render
        args.sdf_opt = get_vol_render_opt(expname, need_train_vol_render, args)
        args.sdf_opt.training.wod = args.wod
        cfg['method'] = "sdf"


    # Shorthands
    exit_after = args.exit_after
    batch_size = cfg['training']['batch_size'] if args.sdf == 0 else args.sdf_opt.training.batch
    n_workers = cfg['training']['n_workers']

    # 读取模型
    # 跳转到 im2scene/config  根据config参数创建giraffe模型或其他模型
    # model = config.get_model(cfg, device=device, len_dataset=len(train_dataset), args=args)
    model = config.get_model(cfg, device=device, args=args)

    # Initialize training
    # 创建优化器
    optimizer_e, optimizer, optimizer_d = config.get_optimizer(cfg, model, args)

    # Print model  打印模型
    nparameters = sum(p.numel() for p in model.parameters())
    logger_py.info(model)
    logger_py.info('Total number of parameters: %d' % nparameters)

    if hasattr(model, "discriminator") and model.discriminator is not None:
        nparameters_d = sum(p.numel() for p in model.discriminator.parameters())
        logger_py.info(
            'Total number of discriminator parameters: %d' % nparameters_d)
    if hasattr(model, "generator") and model.generator is not None:
        nparameters_g = sum(p.numel() for p in model.generator.parameters())
        logger_py.info('Total number of generator parameters: %d' % nparameters_g)

    # 读取数据集
    train_dataset = config.get_dataset(cfg, args)
    if args.sdf:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=data_sampler(train_dataset, shuffle=True, distributed=args.sdf_opt.training.distributed),
            drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True,
            pin_memory=True, drop_last=True,
        )

    if args.sdf == 1:
        if need_train_vol_render:
            train_vol_render(args.sdf_opt.training, args.sdf_opt.experiment, train_loader, model, optimizer_e, optimizer, optimizer_d, args.psp, device)
            need_train_vol_render = False
            args.need_train_vol_render = need_train_vol_render
            args.sdf_opt = get_vol_render_opt(expname, need_train_vol_render, args)
            model = config.get_model(cfg, device=device, args=args)
        if need_train_full_pipeline:
            train_full_pipeline(args.sdf_opt.training, args.sdf_opt.experiment, train_loader, model, optimizer_e, optimizer, optimizer_d, args.psp, device)
        if args.psp != 0 or args.vae != 0:
            os.makedirs(os.path.join(cfg['training']['out_dir'], "encoder", "samples"), exist_ok=True)
            train_encoder(args.sdf_opt.training, args.sdf_opt.experiment, train_loader, model, optimizer_e, args.vae, device)
    else:
        train_giraffe(cfg, logger_py, train_loader, model, optimizer_e, optimizer, optimizer_d, args.vae, exit_after, device)
