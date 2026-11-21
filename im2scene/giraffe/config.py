import os
from im2scene.discriminator import discriminator_dict
from im2scene.giraffe import models, training, rendering
from im2scene.giraffe.models import hash_encoding, autoencoder
import torch
from copy import deepcopy
import numpy as np


def get_model(cfg, device=None, len_dataset=0, args=None, **kwargs):
    ''' Returns the giraffe model.  获取giraffe模型

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        len_dataset (int): length of dataset
    '''
    # 读取参数
    decoder = cfg['model']['decoder']  # MLP网络  默认simple
    discriminator = cfg['model']['discriminator']  # 判别器  默认dc
    generator = cfg['model']['generator']  # 生成器  默认simple
    background_generator = cfg['model']['background_generator']  # 背景生成器 默认simple
    decoder_kwargs = cfg['model']['decoder_kwargs']
    discriminator_kwargs = cfg['model']['discriminator_kwargs']
    generator_kwargs = cfg['model']['generator_kwargs']
    background_generator_kwargs = \
        cfg['model']['background_generator_kwargs']

    # 边界框生成器  默认simple
    bounding_box_generator = cfg['model']['bounding_box_generator']
    # 边界框生成器参数  包括旋转平移缩放等参数，且都为一个最大值和一个最小值
    bounding_box_generator_kwargs = \
        cfg['model']['bounding_box_generator_kwargs']
    neural_renderer = cfg['model']['neural_renderer']
    neural_renderer_kwargs = cfg['model']['neural_renderer_kwargs']
    # 潜在编码z的维度
    z_dim = cfg['model']['z_dim']
    z_dim_bg = cfg['model']['z_dim_bg']
    img_size = cfg['data']['img_size']

    if args.vae==1:
        encoder = autoencoder.Encoder(img_size=img_size, channel_in=3, z_size=2*z_dim)
    else:
        encoder = None

    # 定义MLP网络，包括对象生成器和背景生成器
    if args.i_embed==0 and args.i_embed_views==0:
        # 使用原始位置编码和原始MLP网络层
        # MLP网络  hθ[1,N-1]  转到im2scene/giraffe/models/decoder.py
        decoder = models.decoder_dict[decoder](
            z_dim=z_dim, **decoder_kwargs
        )

        # 定义背景生成器  hθ[N]  转到im2scene/giraffe/models/decoder.py
        if background_generator is not None:
            background_generator = \
                models.background_generator_dict[background_generator](
                    z_dim=z_dim_bg, **background_generator_kwargs)
    else:
        # 使用hash编码

        # 小型MLP网络
        # 后续封装到别处去
        bounding_box = (torch.tensor([-1.5373, -1.3903, -1.0001]).to(device), torch.tensor([1.5373, 1.3903, 1.0001]).to(device))
        # bounding_box = (torch.tensor([-1.5373, -1.3903, -1.0001]), torch.tensor([1.5373, 1.3903, 1.0001]))
        # finest_res = 512
        # log2_hashmap_size = 19

        # x的编码器  返回两个对象，前者为编码器，后者为维度
        embed_fn, input_ch = hash_encoding.get_embedder(bounding_box=bounding_box, finest_res = args.finest_res, log2_hashmap_size=args.log2_hashmap_size, i=args.i_embed)
        embedding_params = list(embed_fn.parameters())

        # d的编码器
        embeddirs_fn, input_ch_views = hash_encoding.get_embedder(i=args.i_embed_views)

        if args.small_net == 0:
            # 使用普通网络
            # MLP网络  hθ[1,N-1]  转到im2scene/giraffe/models/decoder.py
            decoder = models.decoder_dict[decoder](
                z_dim=z_dim, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, dim_embed=input_ch, dim_embed_view=input_ch_views, **decoder_kwargs
            )

            # 定义背景生成器  hθ[N]  转到im2scene/giraffe/models/decoder.py
            if background_generator is not None:
                background_generator = \
                    models.background_generator_dict[background_generator](
                        z_dim=z_dim_bg, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, dim_embed=input_ch, dim_embed_view=input_ch_views, **background_generator_kwargs)
        else:
        # 定义模型
            decoder = models.decoder_dict['small'](
                z_dim=z_dim, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, dim_embed=input_ch, dim_embed_view=input_ch_views, **decoder_kwargs
            )
            background_generator = \
                models.background_generator_dict['small'](
                    z_dim=z_dim_bg, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, dim_embed=input_ch, dim_embed_view=input_ch_views, **background_generator_kwargs
                )

    # 定义判别器  转到im2scene/discriminator/conv.py
    if discriminator is not None:
        discriminator = discriminator_dict[discriminator](
            img_size=img_size, **discriminator_kwargs)

    # 定义边界生成器  负责控制平移旋转缩放参数  转到im2scene/giraffe/models/bounding_box_generator.py
    if bounding_box_generator is not None:
        bounding_box_generator = \
            models.bounding_box_generator_dict[bounding_box_generator](
                z_dim=z_dim, **bounding_box_generator_kwargs)

    # 神经渲染器
    if neural_renderer is not None:
        neural_renderer = models.neural_renderer_dict[neural_renderer](
            z_dim=z_dim, img_size=img_size, **neural_renderer_kwargs
        )

    # 统合生成器  同论文框架图
    # 包括对象生成器、背景生成器、bounding_box生成器、神经渲染器
    if generator is not None:
        generator = models.generator_dict[generator](
            device, z_dim=z_dim, z_dim_bg=z_dim_bg,
            decoder=decoder,
            background_generator=background_generator,
            bounding_box_generator=bounding_box_generator,
            neural_renderer=neural_renderer, **generator_kwargs)

    if cfg['test']['take_generator_average']:
        # 去平均值则添加一个副本
        generator_test = deepcopy(generator)
    else:
        generator_test = None

    model = models.GIRAFFE(
        device=device, encoder=encoder,
        discriminator=discriminator, generator=generator,
        generator_test=generator_test,
    )
    return model


def get_trainer(model, optimizer_e, optimizer, optimizer_d, cfg, device, **kwargs):
    ''' Returns the trainer object.  打包成一个训练器

    Args:
        model (nn.Module): the GIRAFFE model
        optimizer_e (optimizer): encoder optimizer object
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    out_dir = cfg['training']['out_dir']  # 输出路径
    vis_dir = os.path.join(out_dir, 'vis')  # 预测图片路径
    overwrite_visualization = cfg['training']['overwrite_visualization']
    multi_gpu = cfg['training']['multi_gpu']  # 使用多GPU
    n_eval_iterations = (
        cfg['training']['n_eval_images'] // cfg['training']['batch_size'])

    fid_file = cfg['data']['fid_file']
    assert(fid_file is not None)
    fid_dict = np.load(fid_file)

    trainer = training.Trainer(
        model, optimizer_e, optimizer, optimizer_d, device=device, vis_dir=vis_dir,
        overwrite_visualization=overwrite_visualization, multi_gpu=multi_gpu,
        fid_dict=fid_dict,
        n_eval_iterations=n_eval_iterations,
    )

    return trainer


def get_renderer(model, cfg, device, **kwargs):
    ''' Returns the renderer object.

    Args:
        model (nn.Module): GIRAFFE model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    renderer = rendering.Renderer(
        model,
        device=device,)
    return renderer
