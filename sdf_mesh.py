import torch
from munch import Munch
from torchvision import utils
import numpy as np
import os
import argparse
import logging
from tqdm import tqdm

from im2scene import config
from im2scene.config import get_params
from im2scene.sdf.models.sdf_model import Generator
from im2scene.sdf.models.sdf_utils import SDFOptions, generate_camera_params, xyz2mesh, align_volume, \
    extract_mesh_with_marching_cubes

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def generate(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent):
    g_ema.eval()
    if not opt.no_surface_renderings:
        surface_g_ema.eval()

    # set camera angles
    # 设置摄影机角度，如果为true，生成器将从一组固定的摄影机角度渲染缩进。
    if opt.fixed_camera_angles:
        # These can be changed to any other specific viewpoints.
        # You can add or remove viewpoints as you wish
        # 这些可以更改为任何其他特定的视点。
        # 您可以根据需要添加或删除视点
        locations = torch.tensor([[0, 0],
                                  [-1.5 * opt.camera.azim, 0],
                                  [-1 * opt.camera.azim, 0],
                                  [-0.5 * opt.camera.azim, 0],
                                  [0.5 * opt.camera.azim, 0],
                                  [1 * opt.camera.azim, 0],
                                  [1.5 * opt.camera.azim, 0],
                                  [0, -1.5 * opt.camera.elev],
                                  [0, -1 * opt.camera.elev],
                                  [0, -0.5 * opt.camera.elev],
                                  [0, 0.5 * opt.camera.elev],
                                  [0, 1 * opt.camera.elev],
                                  [0, 1.5 * opt.camera.elev]], device=device)
        # For zooming in/out change the values of fov
        # (This can be defined for each view separately via a custom tensor
        # like the locations tensor above. Tensor shape should be [locations.shape[0],1])
        # reasonable values are [0.75 * opt.camera.fov, 1.25 * opt.camera.fov]
        # 要缩放，请更改fov的值
        # 这可以通过自定义张量分别为每个视图定义
        # 就像上面的位置张量一样。张量形状应为[locations.shape[0]，1]）
        # 合理的值为[0.75 * opt.camera.fov, 1.25 * opt.camera.fov]
        # opt.camera.fov： 摄像机视场半角（单位：度）
        fov = opt.camera.fov * torch.ones((locations.shape[0], 1), device=device)
        num_viewdirs = locations.shape[0]
    else:  # draw random camera angles  绘制随机摄影机角度
        locations = None
        # fov = None
        fov = opt.camera.fov  # opt.camera.fov： 摄像机视场半角（单位：度）
        # num_viewdirs = opt.num_views_per_id  # 每个对象生成的视角数
        num_viewdirs = 8  # 每个对象生成的视角数

    # generate images  需要生成的对象数量
    for i in tqdm(range(64)):
        with torch.no_grad():
            chunk = 8  # chunk 类似 batch size
            # 生成潜在编码z  并根据需要生成的视角数复制扩充
            sample_z = torch.randn(1, opt.style_dim, device=device).repeat(num_viewdirs, 1)  # [num_viewdirs, 256]
            # 生成相机参数
            # sample_cam_extrinsics  [1, 3, 4]
            # sample_focals  焦距  [1, 1, 1]
            # sample_near  近界  [1, 1, 1]
            # sample_far  远界  [1, 1, 1]
            # sample_locations  [1, 2]
            sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
                generate_camera_params(opt.renderer_output_size, device, batch=num_viewdirs,
                                       locations=locations,  # input_fov=fov,
                                       uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                       elev_range=opt.camera.elev, fov_ang=fov,
                                       dist_radius=opt.camera.dist_radius)

            # 高分辨率rgb图像 [0. 3. 1024. 1024]
            rgb_images = torch.Tensor(0, 3, opt.size, opt.size)
            # 低分辨率rgb图像 [0. 3. 64. 64]
            rgb_images_thumbs = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)

            os.makedirs(os.path.join(opt.results_dst_dir, str(i)), exist_ok=True)

            # 每个对象根据视角数生成图像
            for j in range(0, num_viewdirs, chunk):
                # 前向传播 生成图像 调用 generator.forward
                # 返回元组 (rgb, thumb_rgb)
                # rgb：高分辨率图 [1, 3, 1024, 1024]
                # thumb_rgb：低分辨率图 [1, 3, 64, 64]
                out = g_ema([sample_z[j:j + chunk]],
                            sample_cam_extrinsics[j:j + chunk],
                            sample_focals[j:j + chunk],
                            sample_near[j:j + chunk],
                            sample_far[j:j + chunk],
                            truncation=opt.truncation_ratio,
                            truncation_latent=mean_latent)

                # 将生成图像添加至列表
                rgb_images = torch.cat([rgb_images, out[0].cpu()], 0)
                rgb_images_thumbs = torch.cat([rgb_images_thumbs, out[1].cpu()], 0)

                # rgb_images = out[0].cpu()
                # rgb_images_thumbs = out[1].cpu()

            for j in range(num_viewdirs):
                utils.save_image(rgb_images[j],
                                 # os.path.join(opt.results_dst_dir, 'images', '{}.png'.format(str(i).zfill(7))),
                                 os.path.join(opt.results_dst_dir, str(i), str(i) + '_' + str(j) + '.png'),
                                 nrow=num_viewdirs,
                                 normalize=True,
                                 padding=0,
                                 value_range=(-1, 1), )

                utils.save_image(rgb_images_thumbs[j],
                                 # os.path.join(opt.results_dst_dir, 'images', '{}_thumb.png'.format(str(i).zfill(7))),
                                 os.path.join(opt.results_dst_dir, str(i), 'thumb' + str(i) + '_' + str(j) + '.png'),
                                 nrow=num_viewdirs,
                                 normalize=True,
                                 padding=0,
                                 value_range=(-1, 1), )

            # this is done to fit to RTX2080 RAM size (11GB)
            del out
            torch.cuda.empty_cache()

            if not opt.no_surface_renderings:  # 生成深度视频/渲染（depth videos/renderings）
                surface_chunk = 1
                # surface_g_ema.renderer是生成器中nerf模块
                # out_im_res是输出分辨率，一号生成器的分辨率是64*64*64，二号是128*128*128
                scale = surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
                surface_sample_focals = sample_focals * scale  # 曲面采样焦距
                for j in range(0, num_viewdirs, surface_chunk):
                    # 前向传播 调用 generator.forward
                    # 计算sdf跟权重
                    # 返回(rgb, thumb_rgb, xyz, sdf, mask)
                    surface_out = surface_g_ema([sample_z[j:j + surface_chunk]],
                                                sample_cam_extrinsics[j:j + surface_chunk],
                                                surface_sample_focals[j:j + surface_chunk],
                                                sample_near[j:j + surface_chunk],
                                                sample_far[j:j + surface_chunk],
                                                truncation=opt.truncation_ratio,
                                                truncation_latent=surface_mean_latent,
                                                return_sdf=True,
                                                return_xyz=True)

                    xyz = surface_out[2].cpu()  # [1, 3, 128, 128]
                    sdf = surface_out[3].cpu()  # [1, 128, 128, 128, 1]

                    # this is done to fit to RTX2080 RAM size (11GB)
                    del surface_out
                    torch.cuda.empty_cache()

                    # mesh extractions are done one at a time  一次提取一个网格
                    for k in range(surface_chunk):
                        curr_locations = sample_locations[j:j + surface_chunk]
                        loc_str = '_azim{}_elev{}'.format(int(curr_locations[k, 0] * 180 / np.pi),
                                                          int(curr_locations[k, 1] * 180 / np.pi))

                        # extract full geometry with marching cubes  用marching cubes提取完整的几何体
                        if j == 0:
                            try:
                                frostum_aligned_sdf = align_volume(sdf)
                                marching_cubes_mesh = extract_mesh_with_marching_cubes(
                                    frostum_aligned_sdf[k:k + surface_chunk])
                            except ValueError:
                                marching_cubes_mesh = None
                                print('Marching cubes extraction failed.')
                                print('Please check whether the SDF values are all larger (or all smaller) than 0.')

                            if marching_cubes_mesh is not None:

                                marching_cubes_mesh_filename = os.path.join(opt.results_dst_dir, str(i),
                                                                            'sample_{}_marching_cubes_mesh{}.obj'.format(
                                                                                i, loc_str))
                                with open(marching_cubes_mesh_filename, 'w') as f:
                                    marching_cubes_mesh.export(f, file_type='obj')


if __name__ == "__main__":
    """
    1.读取模型各项参数
    """

    logger_py = logging.getLogger(__name__)
    np.random.seed(0)
    torch.manual_seed(0)

    args, cfg = get_params()
    args.psp = 0

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    # 读取并创建渲染文件夹
    out_dir = cfg['training']['out_dir']
    render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])

    # 配置参数
    expname = cfg['training']['out_dir'].split("/")[1]
    cfg['method'] = "sdf"
    opt = SDFOptions().parse(
        ["--expname", expname, "--size", "256", "--identities", "8"])
    opt.model.is_test = True
    opt.model.freeze_renderer = False
    opt.model.psp = args.psp
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    opt.rendering.perturb = 0
    opt.rendering.type = "ngp" if args.ngp else "sdf"
    opt.rendering.fc = args.fc
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.inference.return_xyz = opt.rendering.return_xyz
    opt.inference.results_dst_dir = render_dir
    opt.model.psp = 0
    args.sdf_opt = opt

    # 读取chpt路径
    # checkpoint_path = os.path.join(out_dir, expname + '.pt')
    checkpoint_path = os.path.join(out_dir, 'full_pipeline.pt')
    # checkpoint_path = os.path.join(out_dir, 'vol_renderer.pt')
    checkpoint = torch.load(checkpoint_path)

    # 读取模型
    gen = Generator(opt.model, opt.rendering).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = gen.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v
    gen.load_state_dict(model_dict)

    # load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
    if not opt.inference.no_surface_renderings:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model.copy()
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        opt.surf_extraction.rendering.type = "ngp" if args.ngp else "sdf"
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(
            device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None

    # get the mean latent vector for gen
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            # latent code
            mean_latent = gen.mean_latent(opt.inference.truncation_mean, device)
    else:
        surface_mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings:
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None

    generate(opt.inference, gen, surface_g_ema, device, mean_latent, surface_mean_latent)
