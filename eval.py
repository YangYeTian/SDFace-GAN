import torch
from torchvision.utils import save_image, make_grid
import os
from tqdm import tqdm
import time
import numpy as np
from math import ceil

from im2scene.config import get_params
from im2scene.eval import calculate_activation_statistics, calculate_frechet_distance
from im2scene.sdf.models.sdf_model import Generator
from im2scene.sdf.models.sdf_utils import mixing_noise, generate_camera_params
from im2scene.training_utils import get_vol_render_opt
from im2scene import config
from im2scene.checkpoints import CheckpointIO


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def normalize_image(image):
    # 将图像的范围从 [-1, 1] 映射到 [0, 1]
    normalized_image = (image + 1) / 2
    return normalized_image

if __name__ == '__main__':
    args, cfg = get_params()
    args.psp = 0

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    out_dict_file = os.path.join(out_dir, 'fid_evaluation.npz')
    out_img_file = os.path.join(out_dir, 'fid_images.npy')
    out_vis_file = os.path.join(out_dir, 'fid_images.jpg')

    fid_file = cfg['data']['fid_file']
    assert (fid_file is not None)
    # fid_dict = np.load(cfg['data']['fid_file'])
    fid_dict = None

    # n_images = cfg['test']['n_images']  # 20000
    n_images = 5000
    # batch_size = cfg['training']['batch_size'] if args.sdf == 0 else args.sdf_opt.training.batch
    batch_size = 1
    n_iter = ceil(n_images / batch_size)
    out_dict = {'n_images': n_images}

    out_dir = cfg['training']['out_dir']
    render_dir = os.path.join(out_dir, 'eval')
    if not os.path.exists(os.path.join(render_dir)):
        os.makedirs(os.path.join(render_dir))

    img_fake = []

    t0 = time.time()
    if args.sdf == 1:
        expname = cfg['training']['out_dir'].split("/")[1]
        # 体渲染模型路径 判断体渲染器是否训练完毕
        opt = get_vol_render_opt(expname, False, args)
        opt.model.psp = 0
        opt.rendering.fc = args.fc
        args.need_train_vol_render = False
        opt.rendering.type = "ngp" if args.ngp else "sdf"

        model = Generator(opt.model, opt.rendering).to(device)

        checkpoint_path = os.path.join(out_dir, 'full_pipeline.pt')
        checkpoint = torch.load(checkpoint_path)
        pretrained_weights_dict = checkpoint["g_ema"]
        model_dict = model.state_dict()
        for k, v in pretrained_weights_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v
        model.load_state_dict(model_dict)

        model.eval()

        locations = None
        # fov = None
        fov = opt.camera.fov  # opt.camera.fov： 摄像机视场半角（单位：度）
        num_viewdirs = 1  # 每个对象生成的视角数
        chunk = 8  # chunk 类似 batch size

        for i in tqdm(range(n_images)):
            with torch.no_grad():
                # 每次渲染一张
                sample_z = torch.randn(1, opt.training.style_dim, device=device).repeat(num_viewdirs, 1)  # [num_viewdirs, 256]
                sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
                    generate_camera_params(opt.model.renderer_spatial_output_dim, device, batch=num_viewdirs,
                                           locations=locations,  # input_fov=fov,
                                           uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                           elev_range=opt.camera.elev, fov_ang=fov,
                                           dist_radius=opt.camera.dist_radius)
                for j in range(0, num_viewdirs, chunk):
                    batch = model([sample_z[j:j + chunk]],
                                sample_cam_extrinsics[j:j + chunk],
                                sample_focals[j:j + chunk],
                                sample_near[j:j + chunk],
                                sample_far[j:j + chunk],
                                truncation=1,
                                truncation_latent=None)

                    save_image(batch[0],
                                 os.path.join(render_dir, '{}.png'.format(str(i).zfill(7))),
                                 nrow=num_viewdirs,
                                 normalize=True,
                                 padding=0,
                                 value_range=(-1, 1), )

                    normalized_img = normalize_image(batch[0])
                    # save_image(normalized_img,
                    #              os.path.join(render_dir, '{}.png'.format(str(i).zfill(7))),
                    #              nrow=num_viewdirs,
                    #              normalize=True,
                    #              padding=0,
                    #              value_range=(0, 1), )
                img_fake.append(normalized_img.cpu())  # [batch, 3, 256, 256]  # 未归一化

        # img_fake = torch.cat(img_fake, dim=0)[:n_images]
        # print(img_fake.shape, "img_fake.shape")  # [n_iter, 3, 256, 256]

        # torch..clamp_(0., 1.)  归一化，小于0的值变为0，大于1的值变为1
        # img_fake.clamp_(0., 1.)
        n_images = img_fake.shape[0]
    else:
        model = config.get_model(cfg, device=device, args=args)

        checkpoint_io = CheckpointIO(out_dir, model=model)
        checkpoint_io.load(cfg['test']['model_file'])

        # Generate
        model.eval()
        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                img = model(batch_size).cpu()
                # img_fake.append(img.cpu())  # [batch, 3, 256, 256]  # 未归一化
                save_image(img,
                             os.path.join(render_dir, '{}.png'.format(str(i).zfill(7))),
                             nrow=1,
                             normalize=True,
                             padding=0,
                             value_range=(0, 1), )
        img_fake = torch.cat(img_fake, dim=0)[:n_images]
        print(img_fake.shape, "img_fake.shape")  # [n_iter, 3, 256, 256]

        # torch..clamp_(0., 1.)  归一化，小于0的值变为0，大于1的值变为1
        img_fake.clamp_(0., 1.)
        n_images = img_fake.shape[0]

    t = time.time() - t0
    out_dict['time_full'] = t
    out_dict['time_image'] = t / n_images

    img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)
    np.save(out_img_file[:n_images], img_uint8)

    # use uint for eval to fairly compare
    img_fake = torch.from_numpy(img_uint8).float() / 255.
    mu, sigma = calculate_activation_statistics(img_fake)
    out_dict['m'] = mu
    out_dict['sigma'] = sigma

    # calculate FID score and save it to a dictionary
    fid_score = calculate_frechet_distance(mu, sigma, fid_dict['m'], fid_dict['s'])
    out_dict['fid'] = fid_score
    print("FID Score (%d images): %.6f" % (n_images, fid_score))
    np.savez(out_dict_file, **out_dict)

    # Save a grid of 16x16 images for visualization
    save_image(make_grid(img_fake[:256], nrow=16, pad_value=1.), out_vis_file)
