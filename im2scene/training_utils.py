import os
import time
import numpy as np
from PIL.Image import Image
import torch
from torch.nn import functional as F
from torch.nn.functional import mse_loss
from torchvision import utils
from tqdm import tqdm
from im2scene.checkpoints import CheckpointIO
from im2scene import config
from im2scene.encoder.psp_encoders import LossUtils
from im2scene.giraffe.training import reparameterize
from im2scene.sdf.models.sdf_losses import viewpoints_loss, d_logistic_loss, d_r1_loss, eikonal_loss, \
    g_nonsaturating_loss, g_path_regularize, g_content_loss
from im2scene.sdf.models.sdf_utils import get_rank, generate_camera_params, mixing_noise, accumulate, requires_grad, \
    reduce_loss_dict, sample_data, get_ckpt_nums, SDFOptions, reduce_sum, get_world_size
from im2scene.smoothLoss import smoothness

try:
    import wandb
except ImportError:
    wandb = None


def train_giraffe(cfg, logger_py, train_loader, model, optimizer_e, optimizer, optimizer_d, use_encoder, exit_after,
                  device):
    t0 = time.time()

    backup_every = cfg['training']['backup_every']
    out_dir = cfg['training']['out_dir']

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_selection_metric = cfg['training']['model_selection_metric']  # 模型评判度量：fid
    # 模型度量是取大或是取小
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                         'either maximize or minimize.')

    # Shorthands  读取各项评估的时机  如：print_every=10，即每训练10个iteration打印一行
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    # 定义训练器  将model和优化器封装在一起
    trainer = config.get_trainer(model, optimizer_e, optimizer, optimizer_d, cfg, device=device)

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

    # 读取epoch和iteration
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    # 读取最佳损失
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    # 根据设置模型度量取大或是取小修改
    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    print('Current best validation metric (%s): %.8f'
          % (model_selection_metric, metric_val_best))

    t0b = time.time()

    # 训练
    while True:
        epoch_it += 1

        for batch in train_loader:
            # 单个iteration训练
            it += 1

            loss = trainer.train_step(batch, use_encoder, it)

            # Print output  输出打印部分
            if print_every > 0 and (it % print_every) == 0:
                info_txt = '[Epoch %02d] it=%03d, time=%.3f' % (
                    epoch_it, it, time.time() - t0b)
                for (k, v) in loss.items():
                    info_txt += ', %s: %.4f' % (k, v)
                logger_py.info(info_txt)
                t0b = time.time()

            # # Visualize output
            if visualize_every > 0 and (it % visualize_every) == 0:
                logger_py.info('Visualizing')
                image_grid = trainer.visualize(it=it)

            # Save checkpoint
            if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                logger_py.info('Saving checkpoint')
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
            # Backup if necessary
            if backup_every > 0 and (it % backup_every) == 0:
                logger_py.info('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0 and (it > 0):
                print("Performing evaluation step.")
                eval_dict = trainer.evaluate()
                metric_val = eval_dict[model_selection_metric]
                logger_py.info('Validation metric (%s): %.4f'
                               % (model_selection_metric, metric_val))

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logger_py.info('New best model (loss %.4f)' % metric_val_best)
                    checkpoint_io.backup_model_best('model_best.pt')
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best)

            # Exit if necessary
            if 0 < exit_after <= (time.time() - t0):  # t0是开始的时间
                logger_py.info('Time limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
                exit(3)


# 获取体渲染的训练参数
def get_vol_render_opt(expname, need_train_vol_render, args):

    # sdf_opt = SDFOptions().parse(
    #     ["--expname", expname, "--size", "64", "--batch", "8", "--chunk", "2"])
    sdf_opt = SDFOptions().parse(
        ["--expname", expname, "--size", "256", "--batch", "8", "--chunk", "2"])
    if need_train_vol_render:
        # 体渲染训练参数
        sdf_opt.model.freeze_renderer = False
        sdf_opt.model.no_viewpoint_loss = sdf_opt.training.view_lambda == 0.0
        sdf_opt.training.camera = sdf_opt.camera
        sdf_opt.training.renderer_output_size = sdf_opt.model.renderer_spatial_output_dim
        sdf_opt.training.style_dim = sdf_opt.model.style_dim
        sdf_opt.training.with_sdf = not sdf_opt.rendering.no_sdf
        if sdf_opt.training.with_sdf and sdf_opt.training.min_surf_lambda > 0:
            sdf_opt.rendering.return_sdf = True
        sdf_opt.training.iter = 200001
        sdf_opt.rendering.no_features_output = True

        # create checkpoints directories
        os.makedirs(os.path.join(sdf_opt.training.checkpoints_dir, sdf_opt.experiment.expname, 'volume_renderer'),
                    exist_ok=True)
        os.makedirs(
            os.path.join(sdf_opt.training.checkpoints_dir, sdf_opt.experiment.expname, 'volume_renderer', 'samples'),
            exist_ok=True)

    else:
        # style-gan训练参数
        sdf_opt.training.camera = sdf_opt.camera
        sdf_opt.training.size = sdf_opt.model.size
        sdf_opt.training.renderer_output_size = sdf_opt.model.renderer_spatial_output_dim
        sdf_opt.training.style_dim = sdf_opt.model.style_dim
        sdf_opt.model.freeze_renderer = True

        sdf_opt.model.no_viewpoint_loss = sdf_opt.training.view_lambda == 0.0

        os.makedirs(os.path.join(sdf_opt.training.checkpoints_dir, sdf_opt.experiment.expname, 'full_pipeline'),
                    exist_ok=True)
        os.makedirs(
            os.path.join(sdf_opt.training.checkpoints_dir, sdf_opt.experiment.expname, 'full_pipeline', 'samples'),
            exist_ok=True)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    sdf_opt.training.distributed = n_gpu > 1
    sdf_opt.training.start_iter = 0
    sdf_opt.rendering.type = "ngp" if args.ngp else "sdf"
    sdf_opt.rendering.fc = args.fc
    sdf_opt.model.psp = args.psp

    return sdf_opt


# 训练体渲染器
def train_vol_render(opt, experiment_opt, train_loader, model, optimizer_e, optimizer, optimizer_d, use_psp,
                     device):
    """
    读取checkpoint
    """
    last_ckpt = get_ckpt_nums(os.path.join(opt.checkpoints_dir, experiment_opt.expname, "volume_renderer"))
    ckpt_path = os.path.join(opt.checkpoints_dir,
                             experiment_opt.expname,
                             "volume_renderer",
                             'models_{}.pt'.format(last_ckpt.zfill(7))) if last_ckpt is not None else "None"
    # if experiment_opt.continue_training and experiment_opt.ckpt is not None:
    if os.path.exists(ckpt_path):
        # 继续训练
        experiment_opt.continue_training = True
        experiment_opt.ckpt = last_ckpt
        if get_rank() == 0:
            print("load model:", last_ckpt)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        try:
            opt.start_iter = int(experiment_opt.ckpt) + 1

        except ValueError:
            pass
        model.generator.load_state_dict(ckpt["g"])
        model.discriminator.load_state_dict(ckpt["d"])
        model.generator_test.load_state_dict(ckpt["g_ema"])
        if "g_optim" in ckpt.keys():
            optimizer.load_state_dict(ckpt["g_optim"])
            optimizer_d.load_state_dict(ckpt["d_optim"])

    if use_psp:
        opt.style_dim = opt.style_dim * 2
        sphere_init_path = os.path.join(opt.checkpoints_dir, experiment_opt.expname, f"sdf_init_models.pt")
    else:
        sphere_init_path = os.path.join(opt.checkpoints_dir, experiment_opt.expname, f"sdf_init_models.pt")
        # sphere_init_path = './pretrained_renderer/sphere_init.pt'

    if opt.no_sphere_init:
        # 不使用 sphere SDF 初始化渲染器
        opt.sphere_init = False
    elif not experiment_opt.continue_training and opt.with_sdf and os.path.isfile(sphere_init_path):
        # 如果是第一次训练，且使用SDF，且 sphere SDF 的文件存在
        if get_rank() == 0:
            print("loading sphere initialized model")
        ckpt = torch.load(sphere_init_path, map_location=lambda storage, loc: storage)
        model.generator.load_state_dict(ckpt["g"])
        model.discriminator.load_state_dict(ckpt["d"])
        model.generator_test.load_state_dict(ckpt["g_ema"])
        opt.sphere_init = False
    else:
        opt.sphere_init = True

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    d_view_loss = torch.tensor(0.0, device=device)
    g_view_loss = torch.tensor(0.0, device=device)
    g_eikonal = torch.tensor(0.0, device=device)
    g_minimal_surface = torch.tensor(0.0, device=device)

    g_loss_val = 0
    loss_dict = {}

    viewpoint_condition = opt.view_lambda > 0

    if opt.distributed:
        g_module = model.generator.module
        d_module = model.discriminator.module
    else:
        g_module = model.generator
        d_module = model.discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = [torch.randn(opt.val_n_sample, opt.style_dim, device=device).repeat_interleave(8, dim=0)]  # latent code [1, 256]
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = generate_camera_params(opt.renderer_output_size,
                                                                                              device,
                                                                                              batch=opt.val_n_sample,
                                                                                              sweep=True,
                                                                                              uniform=opt.camera.uniform,
                                                                                              azim_range=opt.camera.azim,
                                                                                              elev_range=opt.camera.elev,
                                                                                              fov_ang=opt.camera.fov,
                                                                                              dist_radius=opt.camera.dist_radius)
    """
        nerf模块 训练1w个iteration
        如果使用 sphere_init 初始化失败了，训练1w个iteration
    """
    if opt.with_sdf and opt.sphere_init and opt.start_iter == 0:
        # 如果是第一次则初始化
        # nerf模块提前训练1w个iteration
        init_pbar = range(10000)
        if get_rank() == 0:
            init_pbar = tqdm(init_pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

        model.generator.zero_grad()
        for idx in init_pbar:

            noise = mixing_noise(3, opt.style_dim, opt.mixing, device)

            # 生成相机参数
            cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device,
                                                                                     batch=3,
                                                                                     uniform=opt.camera.uniform,
                                                                                     azim_range=opt.camera.azim,
                                                                                     elev_range=opt.camera.elev,
                                                                                     fov_ang=opt.camera.fov,
                                                                                     dist_radius=opt.camera.dist_radius)

            # 初始化使用简化的代码提升效率
            sdf, target_values = g_module.init_forward(noise, cam_extrinsics, focal, near, far)
            loss = F.l1_loss(sdf, target_values)
            loss.backward()
            optimizer.step()
            model.generator.zero_grad()
            if get_rank() == 0:
                init_pbar.set_description((f"MLP init to sphere procedure - Loss: {loss.item():.4f}"))

        accumulate(model.generator_test, g_module, 0)
        torch.save(
            {
                "g": g_module.state_dict(),
                "d": d_module.state_dict(),
                "g_ema": model.generator_test.state_dict(),
            },
            os.path.join(opt.checkpoints_dir, experiment_opt.expname, f"sdf_init_models.pt")
        )
        # 已成功保存SDF初始化MLP的检查点
        print('Successfully saved checkpoint for SDF initialized MLP.')

    """
        训练20w个iteration
    """
    # pbar = range(opt.iter)  # opt.iter 训练迭代的总数 默认为20w
    pbar = range(10000)  # opt.iter 训练迭代的总数 默认为20w
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)

    loader = sample_data(train_loader)
    for idx in pbar:  # 单个iteration
        i = idx + opt.start_iter

        if i > opt.iter:
            print("Done!")
            break

        # 先训练判别器
        requires_grad(model.generator, False)
        requires_grad(model.discriminator, True)
        model.discriminator.zero_grad()
        # _:原比例图像[batch，3，model.size, model.size] 表示最终图像大小
        # real_imgs:低比例图像[batch,3,model.renderer_spatial_output_dim, model.renderer_spatial_output_dim] 表示stylegan输入图像大小
        _, real_imgs = next(loader)
        real_imgs = real_imgs.to(device)
        # batch = next(loader)
        # real_imgs = batch.get('image').to(device)  # 读取真实图片

        noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)
        cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device,
                                                                                 batch=opt.batch,
                                                                                 uniform=opt.camera.uniform,
                                                                                 azim_range=opt.camera.azim,
                                                                                 elev_range=opt.camera.elev,
                                                                                 fov_ang=opt.camera.fov,
                                                                                 dist_radius=opt.camera.dist_radius)
        gen_imgs = []
        for j in range(0, opt.batch, opt.chunk):
            curr_noise = [n[j:j + opt.chunk] for n in noise]
            _, fake_img = model.generator(curr_noise,
                                          cam_extrinsics[j:j + opt.chunk],
                                          focal[j:j + opt.chunk],
                                          near[j:j + opt.chunk],
                                          far[j:j + opt.chunk])

            gen_imgs += [fake_img]

        gen_imgs = torch.cat(gen_imgs, 0)
        fake_pred, fake_viewpoint_pred = model.discriminator(gen_imgs.detach())
        if viewpoint_condition:
            d_view_loss = opt.view_lambda * viewpoints_loss(fake_viewpoint_pred, gt_viewpoints)

        real_imgs.requires_grad = True
        real_pred, _ = model.discriminator(real_imgs)
        d_gan_loss = d_logistic_loss(real_pred, fake_pred)
        grad_penalty = d_r1_loss(real_pred, real_imgs)
        r1_loss = opt.r1 * 0.5 * grad_penalty
        d_loss = d_gan_loss + r1_loss + d_view_loss
        d_loss.backward()
        optimizer_d.step()

        loss_dict["d"] = d_gan_loss
        loss_dict["r1"] = r1_loss
        loss_dict["d_view"] = d_view_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        # 训练生成器
        requires_grad(model.generator, True)
        requires_grad(model.discriminator, False)

        for j in range(0, opt.batch, opt.chunk):
            noise = mixing_noise(opt.chunk, opt.style_dim, opt.mixing, device)
            cam_extrinsics, focal, near, far, curr_gt_viewpoints = generate_camera_params(opt.renderer_output_size,
                                                                                          device, batch=opt.chunk,
                                                                                          uniform=opt.camera.uniform,
                                                                                          azim_range=opt.camera.azim,
                                                                                          elev_range=opt.camera.elev,
                                                                                          fov_ang=opt.camera.fov,
                                                                                          dist_radius=opt.camera.dist_radius)

            out = model.generator(noise, cam_extrinsics, focal, near, far,
                                  return_sdf=opt.min_surf_lambda > 0,
                                  return_eikonal=opt.eikonal_lambda > 0)
            fake_img = out[1]
            if opt.min_surf_lambda > 0:
                sdf = out[2]
            if opt.eikonal_lambda > 0:
                eikonal_term = out[3]

            fake_pred, fake_viewpoint_pred = model.discriminator(fake_img)
            if viewpoint_condition:
                g_view_loss = opt.view_lambda * viewpoints_loss(fake_viewpoint_pred, curr_gt_viewpoints)

            if opt.with_sdf and opt.eikonal_lambda > 0:
                g_eikonal, g_minimal_surface = eikonal_loss(eikonal_term, sdf=sdf if opt.min_surf_lambda > 0 else None,
                                                            beta=opt.min_surf_beta)
                g_eikonal = opt.eikonal_lambda * g_eikonal
                if opt.min_surf_lambda > 0:
                    g_minimal_surface = opt.min_surf_lambda * g_minimal_surface

                # 计算sdf smooth loss

                # near = torch.cat([near, near[0].unsqueeze(dim=0)])
                # far = torch.cat([far, far[0].unsqueeze(dim=0)])
                near = torch.tensor([[[-1.0]], [[-1.3]], [[-1.7]]])
                far = torch.tensor([[[7.0]], [[3.7]], [[1.4]]])
                bounding_box = torch.cat((near, far), dim=1).squeeze()
                g_smooth = smoothness(model.generator, bounding_box, noise, device)
                g_smooth = 1000 * g_smooth


            g_gan_loss = g_nonsaturating_loss(fake_pred)
            g_loss = g_gan_loss + g_view_loss + g_eikonal + g_minimal_surface + g_smooth
            g_loss.backward()

        optimizer.step()
        model.generator.zero_grad()
        loss_dict["g"] = g_gan_loss
        loss_dict["g_view"] = g_view_loss
        loss_dict["g_eikonal"] = g_eikonal
        loss_dict["g_minimal_surface"] = g_minimal_surface
        loss_dict["g_smooth"] = g_smooth

        accumulate(model.generator_test, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        d_view_val = loss_reduced["d_view"].mean().item()
        g_view_val = loss_reduced["g_view"].mean().item()
        g_eikonal_loss = loss_reduced["g_eikonal"].mean().item()
        g_minimal_surface_loss = loss_reduced["g_minimal_surface"].mean().item()
        g_smooth = loss_reduced["g_smooth"].mean().item()
        g_beta_val = g_module.renderer.sigmoid_beta.item() if opt.with_sdf else 0

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; viewpoint: {d_view_val + g_view_val:.4f}; eikonal: {g_eikonal_loss:.4f}; surf: {g_minimal_surface_loss:.4f}; smooth: {g_smooth:.4f}")
            )

            if i % 1000 == 0:  # 输出图片
                with torch.no_grad():
                    samples = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)
                    step_size = 4
                    mean_latent = g_module.mean_latent(10000, device)
                    for k in range(0, opt.val_n_sample * 8, step_size):
                        _, curr_samples = model.generator_test([sample_z[0][k:k + step_size]],
                                                               sample_cam_extrinsics[k:k + step_size],
                                                               sample_focals[k:k + step_size],
                                                               sample_near[k:k + step_size],
                                                               sample_far[k:k + step_size],
                                                               truncation=0.7,
                                                               truncation_latent=mean_latent, )
                        samples = torch.cat([samples, curr_samples.cpu()], 0)

                    if i % 10000 == 0:
                        if not os.path.exists(
                                os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer',
                                             "samples")):
                            os.makedirs(
                                os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer', "samples"))
                        utils.save_image(samples,
                                         os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer',
                                                      f"samples/{str(i).zfill(7)}.png"),
                                         nrow=int(opt.val_n_sample),
                                         normalize=True,
                                         value_range=(-1, 1), )

            if wandb and opt.wandb:
                wandb_log_dict = {"Generator": g_loss_val,
                                  "Discriminator": d_loss_val,
                                  "R1": r1_val,
                                  "Real Score": real_score_val,
                                  "Fake Score": fake_score_val,
                                  "D viewpoint": d_view_val,
                                  "G viewpoint": g_view_val,
                                  "G eikonal loss": g_eikonal_loss,
                                  "G minimal surface loss": g_minimal_surface_loss,
                                  }
                if opt.with_sdf:
                    wandb_log_dict.update({"Beta value": g_beta_val})

                if i % 1000 == 0:
                    wandb_grid = utils.make_grid(samples, nrow=int(opt.val_n_sample),
                                                 normalize=True, value_range=(-1, 1))
                    wandb_ndarr = (255 * wandb_grid.permute(1, 2, 0).numpy()).astype(np.uint8)
                    wandb_images = Image.fromarray(wandb_ndarr)
                    wandb_log_dict.update({"examples": [wandb.Image(wandb_images,
                                                                    caption="Generated samples for azimuth angles of: -35, -25, -15, -5, 5, 15, 25, 35 degrees.")]})

                wandb.log(wandb_log_dict)

            if i % 10000 == 0 or (i < 10000 and i % 1000 == 0):  #  保存模型
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": model.generator_test.state_dict(),
                    },
                    os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer',
                                 f"models_{str(i).zfill(7)}.pt")
                )
                print('Successfully saved checkpoint for iteration {}.'.format(i))

    if get_rank() == 0:
        # create final model directory
        # final_model_path = 'pretrained_renderer'
        # os.makedirs(final_model_path, exist_ok=True)
        torch.save(
            {
                "g": g_module.state_dict(),
                "d": d_module.state_dict(),
                "g_ema": model.generator_test.state_dict(),
            },
            os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'vol_renderer.pt')
        )
        print('Successfully saved final model.')


def train_full_pipeline(opt, experiment_opt, train_loader, model, optimizer_e, optimizer, optimizer_d, use_psp,
                        device):
    # 如果之前已经执行过了，则读取ckpt
    last_ckpt = get_ckpt_nums(os.path.join(opt.checkpoints_dir, experiment_opt.expname, "full_pipeline"))
    ckpt_path = os.path.join(opt.checkpoints_dir,
                             experiment_opt.expname,
                             "full_pipeline",
                             'models_{}.pt'.format(last_ckpt.zfill(7))) if last_ckpt is not None else "None"
    # if experiment_opt.continue_training and experiment_opt.ckpt is not None:
    if os.path.exists(ckpt_path):
        experiment_opt.continue_training = True
        experiment_opt.ckpt = last_ckpt
        if get_rank() == 0:
            print("load model:", experiment_opt.ckpt)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        try:
            opt.start_iter = int(experiment_opt.ckpt) + 1

        except ValueError:
            pass

        model.generator.load_state_dict(ckpt["g"])
        model.discriminator.load_state_dict(ckpt["d"])
        model.generator_test.load_state_dict(ckpt["g_ema"])
    else:
        # 第一次运行 保存配置文件
        # save configuration
        import yaml
        opt_path = os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline', f"opt.yaml")
        with open(opt_path, 'w') as f:
            yaml.safe_dump(opt, f)

    # 如果第一次执行，则读取上一阶段的运行结果
    if not experiment_opt.continue_training:
        if get_rank() == 0:
            print("loading pretrained renderer weights...")
        # pretrained_renderer_path = os.path.join('./pretrained_renderer', opt.experiment.expname + '_vol_renderer.pt')
        if opt.wod:
            pretrained_renderer_path = os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'sdf_init_models.pt')
        else:
            pretrained_renderer_path = os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'vol_renderer.pt')
        try:
            # 读取模型
            ckpt = torch.load(pretrained_renderer_path, map_location=lambda storage, loc: storage)
        except:
            print('Pretrained volume renderer experiment name does not match the full pipeline experiment name.')
            vol_renderer_expname = str(input('Please enter the pretrained volume renderer experiment name:'))
            pretrained_renderer_path = os.path.join('./pretrained_renderer',
                                                    vol_renderer_expname + '.pt')
            ckpt = torch.load(pretrained_renderer_path, map_location=lambda storage, loc: storage)

        pretrained_renderer_dict = ckpt["g_ema"]
        model_dict = model.generator.state_dict()
        for k, v in pretrained_renderer_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        model.generator.load_state_dict(model_dict)

    if use_psp:
        opt.style_dim = opt.style_dim * 2

    # initialize g_ema weights to generator weights
    accumulate(model.generator_test, model.generator, 0)

    mean_path_length = 0
    d_loss_val = 0
    g_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_gan_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if opt.distributed:
        g_module = model.generator.module
        d_module = model.discriminator.module
    else:
        g_module = model.generator
        d_module = model.discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = [torch.randn(opt.val_n_sample, opt.style_dim, device=device).repeat_interleave(8, dim=0)]
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = generate_camera_params(opt.renderer_output_size,
                                                                                              device,
                                                                                              batch=opt.val_n_sample,
                                                                                              sweep=True,
                                                                                              uniform=opt.camera.uniform,
                                                                                              azim_range=opt.camera.azim,
                                                                                              elev_range=opt.camera.elev,
                                                                                              fov_ang=opt.camera.fov,
                                                                                              dist_radius=opt.camera.dist_radius)

    pbar = range(opt.iter)  # 30w
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)
    # 初始化loader
    loader = sample_data(train_loader)

    for idx in pbar:
        i = idx + opt.start_iter
        if i > opt.iter:
            print("Done!")

            break

        requires_grad(model.generator, False)
        requires_grad(model.discriminator, True)
        model.discriminator.zero_grad()
        d_regularize = i % opt.d_reg_every == 0

        # real_imgs:原比例图像[batch，3，model.size, model.size] 表示最终图像大小
        # real_thumb_imgs:低比例图像[batch,3,model.renderer_spatial_output_dim, model.renderer_spatial_output_dim] 表示stylegan输入图像大小
        real_imgs, real_thumb_imgs = next(loader)
        real_imgs = real_imgs.to(device)
        real_thumb_imgs = real_thumb_imgs.to(device)
        # batch = next(loader)
        # real_imgs = batch.get('image').to(device)  # 读取真实图片
        noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)
        cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device,
                                                                                 batch=opt.batch,
                                                                                 uniform=opt.camera.uniform,
                                                                                 azim_range=opt.camera.azim,
                                                                                 elev_range=opt.camera.elev,
                                                                                 fov_ang=opt.camera.fov,
                                                                                 dist_radius=opt.camera.dist_radius)
        # 判别器训练
        for j in range(0, opt.batch, opt.chunk):
            curr_real_imgs = real_imgs[j:j + opt.chunk]
            curr_real_thumb_imgs = real_thumb_imgs[j:j + opt.chunk]
            curr_noise = [n[j:j + opt.chunk] for n in noise]
            gen_imgs, _ = model.generator(curr_noise,
                                          cam_extrinsics[j:j + opt.chunk],
                                          focal[j:j + opt.chunk],
                                          near[j:j + opt.chunk],
                                          far[j:j + opt.chunk])
            # gen_imgs [2,3,256,256]
            fake_pred = model.discriminator(gen_imgs.detach())

            if d_regularize:
                curr_real_imgs.requires_grad = True
                curr_real_thumb_imgs.requires_grad = True

            real_pred = model.discriminator(curr_real_imgs)
            d_gan_loss = d_logistic_loss(real_pred, fake_pred)

            if d_regularize:
                grad_penalty = d_r1_loss(real_pred, curr_real_imgs)
                r1_loss = opt.r1 * 0.5 * grad_penalty * opt.d_reg_every
            else:
                r1_loss = torch.zeros_like(r1_loss)

            d_loss = d_gan_loss + r1_loss
            d_loss.backward()

        optimizer_d.step()

        loss_dict["d"] = d_gan_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        if d_regularize or i == opt.start_iter:
            loss_dict["r1"] = r1_loss.mean()
        # 生成器训练
        requires_grad(model.generator, True)
        requires_grad(model.discriminator, False)

        for j in range(0, opt.batch, opt.chunk):
            noise = mixing_noise(opt.chunk, opt.style_dim, opt.mixing, device)
            cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device,
                                                                                     batch=opt.chunk,
                                                                                     uniform=opt.camera.uniform,
                                                                                     azim_range=opt.camera.azim,
                                                                                     elev_range=opt.camera.elev,
                                                                                     fov_ang=opt.camera.fov,
                                                                                     dist_radius=opt.camera.dist_radius)

            fake_img, fake_img_thumb = model.generator(noise, cam_extrinsics, focal, near, far)
            fake_img_up = torch.nn.Upsample(scale_factor=4)(fake_img_thumb)  # 使用双三次插值将低分辨率图转换为高分辨率图，用于计算损失
            fake_pred = model.discriminator(fake_img)
            g_gan_loss = g_nonsaturating_loss(fake_pred)
            g_cont_loss = g_content_loss(fake_img, fake_img_up)

            # g_loss = g_gan_loss
            g_loss = g_gan_loss + 0.001 * g_cont_loss
            g_loss.backward()

        optimizer.step()
        model.generator.zero_grad()

        loss_dict["g"] = g_gan_loss

        # generator path regularization
        g_regularize = (opt.g_reg_every > 0) and (i % opt.g_reg_every == 0)
        if g_regularize:
            path_batch_size = max(1, opt.batch // opt.path_batch_shrink)
            path_noise = mixing_noise(path_batch_size, opt.style_dim, opt.mixing, device)
            path_cam_extrinsics, path_focal, path_near, path_far, _ = generate_camera_params(opt.renderer_output_size,
                                                                                             device,
                                                                                             batch=path_batch_size,
                                                                                             uniform=opt.camera.uniform,
                                                                                             azim_range=opt.camera.azim,
                                                                                             elev_range=opt.camera.elev,
                                                                                             fov_ang=opt.camera.fov,
                                                                                             dist_radius=opt.camera.dist_radius)

            for j in range(0, path_batch_size, opt.chunk):
                path_fake_img, path_latents = model.generator(path_noise, path_cam_extrinsics,
                                                              path_focal, path_near, path_far,
                                                              return_latents=True)

                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    path_fake_img, path_latents, mean_path_length
                )

                weighted_path_loss = opt.path_regularize * opt.g_reg_every * path_loss  # * opt.chunk / path_batch_size
                if opt.path_batch_shrink:
                    weighted_path_loss += 0 * path_fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

            optimizer.step()
            model.generator.zero_grad()

            mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(model.generator_test, g_module, accum)
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; path: {path_loss_val:.4f}")
            )

            if i % 1000 == 0 or i == opt.start_iter:  # 每1000步保存图片
                with torch.no_grad():
                    thumbs_samples = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)
                    samples = torch.Tensor(0, 3, opt.size, opt.size)
                    step_size = 8  # 设置角度数量
                    mean_latent = g_module.mean_latent(10000, device)
                    for k in range(0, opt.val_n_sample * 8, step_size):
                        curr_samples, curr_thumbs = model.generator_test([sample_z[0][k:k + step_size]],
                                                                         sample_cam_extrinsics[k:k + step_size],
                                                                         sample_focals[k:k + step_size],
                                                                         sample_near[k:k + step_size],
                                                                         sample_far[k:k + step_size],
                                                                         truncation=0.7,
                                                                         truncation_latent=mean_latent)
                        samples = torch.cat([samples, curr_samples.cpu()], 0)
                        thumbs_samples = torch.cat([thumbs_samples, curr_thumbs.cpu()], 0)

                    if i % 10000 == 0:
                        utils.save_image(samples,
                                         os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline',
                                                      f"samples/{str(i).zfill(7)}.png"),
                                         nrow=int(opt.val_n_sample),
                                         normalize=True,
                                         value_range=(-1, 1), )

                        utils.save_image(thumbs_samples,
                                         os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline',
                                                      f"samples/{str(i).zfill(7)}_thumbs.png"),
                                         nrow=int(opt.val_n_sample),
                                         normalize=True,
                                         value_range=(-1, 1), )

            if wandb and opt.wandb:
                wandb_log_dict = {"Generator": g_loss_val,
                                  "Discriminator": d_loss_val,
                                  "R1": r1_val,
                                  "Real Score": real_score_val,
                                  "Fake Score": fake_score_val,
                                  "Path Length Regularization": path_loss_val,
                                  "Path Length": path_length_val,
                                  "Mean Path Length": mean_path_length,
                                  }
                if i % 5000 == 0:
                    wandb_grid = utils.make_grid(samples, nrow=int(opt.val_n_sample),
                                                 normalize=True, value_range=(-1, 1))
                    wandb_ndarr = (255 * wandb_grid.permute(1, 2, 0).numpy()).astype(np.uint8)
                    wandb_images = Image.fromarray(wandb_ndarr)
                    wandb_log_dict.update({"examples": [wandb.Image(wandb_images,
                                                                    caption="Generated samples for azimuth angles of: -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35 Radians.")]})

                    wandb_thumbs_grid = utils.make_grid(thumbs_samples, nrow=int(opt.val_n_sample),
                                                        normalize=True, value_range=(-1, 1))
                    wandb_thumbs_ndarr = (255 * wandb_thumbs_grid.permute(1, 2, 0).numpy()).astype(np.uint8)
                    wandb_thumbs = Image.fromarray(wandb_thumbs_ndarr)
                    wandb_log_dict.update({"thumb_examples": [wandb.Image(wandb_thumbs,
                                                                          caption="Generated samples for azimuth angles of: -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35 Radians.")]})

                wandb.log(wandb_log_dict)

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": model.generator_test.state_dict(),
                    },
                    os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline',
                                 f"models_{str(i).zfill(7)}.pt")
                )
                print('Successfully saved checkpoint for iteration {}.'.format(i))

    if get_rank() == 0:
        # create final model directory
        torch.save(
            {
                "g": g_module.state_dict(),
                "d": d_module.state_dict(),
                "g_ema": model.generator_test.state_dict(),
            },
            # os.path.join(final_model_path, experiment_opt.expname + '.pt')
            # os.path.join(opt.checkpoints_dir, experiment_opt.expname, experiment_opt.expname + '.pt')
            os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline.pt')
        )
        print('Successfully saved final model.')


def train_encoder(opt, experiment_opt, train_loader, model, optimizer_e, use_psp, device):
    last_ckpt = get_ckpt_nums(os.path.join(opt.checkpoints_dir, experiment_opt.expname, "encoder"))
    ckpt_path = os.path.join(opt.checkpoints_dir,
                             experiment_opt.expname,
                             "encoder",
                             'models_{}.pt'.format(last_ckpt.zfill(7))) if last_ckpt is not None else "None"
    # if experiment_opt.continue_training and experiment_opt.ckpt is not None:
    if os.path.exists(ckpt_path):
        experiment_opt.continue_training = True
        experiment_opt.ckpt = last_ckpt
        if get_rank() == 0:
            print("load model:", experiment_opt.ckpt)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        try:
            opt.start_iter = int(experiment_opt.ckpt) + 1

        except ValueError:
            pass

        model.encoder.load_state_dict(ckpt["e"])
        model.generator.load_state_dict(ckpt["g"])
        model.discriminator.load_state_dict(ckpt["d"])
        model.generator_test.load_state_dict(ckpt["g_ema"])
    else:
        # 第一次运行 保存配置文件
        # save configuration
        import yaml
        opt_path = os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'encoder', f"opt.yaml")
        with open(opt_path, 'w') as f:
            yaml.safe_dump(opt, f)

    # 如果第一次执行，则读取上一阶段的运行结果
    if not experiment_opt.continue_training:
        if get_rank() == 0:
            print("loading pretrained renderer weights...")
        # pretrained_renderer_path = os.path.join('./pretrained_renderer', opt.experiment.expname + '_vol_renderer.pt')
        pretrained_full_pipeline_path = os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline.pt')
        try:
            # 读取模型
            ckpt = torch.load(pretrained_full_pipeline_path, map_location=lambda storage, loc: storage)
        except:
            print('Pretrained volume renderer experiment name does not match the full pipeline experiment name.')
            full_pipeline_expname = str(input('Please enter the pretrained volume renderer experiment name:'))
            pretrained_full_pipeline_path = os.path.join('./pretrained_renderer',
                                                    full_pipeline_expname + '.pt')
            ckpt = torch.load(pretrained_full_pipeline_path, map_location=lambda storage, loc: storage)

        # 读取生成器模型
        model.generator.load_state_dict(ckpt["g"])
        model.discriminator.load_state_dict(ckpt["d"])
        model.generator_test.load_state_dict(ckpt["g_ema"])

        if use_psp:
            # 读取编码器模型
            encoder_ckpt = torch.load("pretrained_renderer/model_ir_se50.pth")
            model.encoder.load_state_dict(encoder_ckpt, strict=False)



    # initialize g_ema weights to generator weights
    accumulate(model.generator_test, model.generator, 0)

    mean_path_length = 0
    d_loss_val = 0
    g_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_gan_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if opt.distributed:
        e_module = model.encoder.module
        g_module = model.generator.module
        d_module = model.discriminator.module
    else:
        e_module = model.encoder
        g_module = model.generator
        d_module = model.discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    loss_utils = LossUtils(device)

    # 用于评价的潜在编码z
    # sample_z = [torch.randn(opt.val_n_sample, opt.style_dim, device=device).repeat_interleave(8, dim=0)]  # 【64，256】
    eval_imgs = torch.tensor(np.load("./data/ffhq/eval.npy")).to(device)
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = generate_camera_params(opt.renderer_output_size,
                                                                                              device,
                                                                                              batch=opt.val_n_sample,
                                                                                              sweep=True,
                                                                                              uniform=opt.camera.uniform,
                                                                                              azim_range=opt.camera.azim,
                                                                                              elev_range=opt.camera.elev,
                                                                                              fov_ang=opt.camera.fov,
                                                                                              dist_radius=opt.camera.dist_radius)

    # _, latent_avg = model.generator.mean_latent(10000, device)

    pbar = range(500000)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)
    # 初始化loader
    loader = sample_data(train_loader)

    for idx in pbar:
        i = idx + opt.start_iter
        if i > opt.iter:
            print("Done!")

            break

        requires_grad(model.encoder, True)
        requires_grad(model.generator, False)
        requires_grad(model.discriminator, False)

        model.encoder.zero_grad()

        # real_imgs:原比例图像[batch，3，model.size, model.size] 表示最终图像大小
        # real_thumb_imgs:低比例图像[batch,3,model.renderer_spatial_output_dim, model.renderer_spatial_output_dim] 表示stylegan输入图像大小
        real_imgs, real_thumb_imgs = next(loader)
        real_imgs = real_imgs.to(device)
        real_thumb_imgs = real_thumb_imgs.to(device)

        # batch 8 chunk 2
        # PsP
        # noise = model.encoder(real_imgs)
        # # noise = noise + latent_avg.repeat(noise.shape[0], 1, 1)

        # vae
        mu, logvar = model.encoder(real_imgs)
        noise = [reparameterize(mu, logvar)]

        # org
        # noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)  # ([batch,256],[batch,256])
        cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device,
                                                                                 batch=opt.batch,
                                                                                 uniform=opt.camera.uniform,
                                                                                 azim_range=opt.camera.azim,
                                                                                 elev_range=opt.camera.elev,
                                                                                 fov_ang=opt.camera.fov,
                                                                                 dist_radius=opt.camera.dist_radius)
        # mse_img = 0
        # mse_feat = 0

        img_list = torch.tensor([]).to(device)
        thumb_img_list = torch.tensor([]).to(device)

        # 单个chunk训练
        for j in range(0, opt.batch, opt.chunk):
            curr_real_imgs = real_imgs[j:j + opt.chunk]  # [chunk, 3, 256, 256]
            # org and vae
            curr_noise = [n[j:j + opt.chunk] for n in noise]
            # psp
            # curr_noise = [noise[j:j + opt.chunk]]
            gen_imgs, gen_imgs_thumb = model.generator(curr_noise,  # gen_imags [chunk,3,256,256]
                                          cam_extrinsics[j:j + opt.chunk],
                                          focal[j:j + opt.chunk],
                                          near[j:j + opt.chunk],
                                          far[j:j + opt.chunk],
                                          # input_is_latent=True
                                          )
            img_list = torch.cat((img_list, gen_imgs), 0)
            thumb_img_list = torch.cat((thumb_img_list, gen_imgs_thumb), 0)


            # gen_imgs [2,3,256,256]
            # fake_feat = model.discriminator.get_feat(gen_imgs.detach())
            # real_feat = model.discriminator.get_feat(curr_real_imgs.detach())
            #
            # mse_img = mse_img + mse_loss(gen_imgs, curr_real_imgs, reduction="sum")
            # mse_feat = mse_feat + torch.sum(0.5 * (real_feat - fake_feat) ** 2, 1)  # 判别器"高层特征"损失

        # 计算损失
        # kl = -0.5 * torch.sum(-logvar.exp() - torch.pow(mu, 2) + logvar + 1, 1)  # 计算kl散度
        # # e_loss = torch.sum(kl) + 0.5 * torch.sum(mse_img) + 0.5 * torch.sum(mse_feat)
        # e_loss = torch.sum(kl) + torch.sum(mse_feat)

        e_loss1, loss_dict, id_logs = loss_utils.calc_loss(real_thumb_imgs, real_thumb_imgs, thumb_img_list)
        e_loss2, loss_dict, id_logs = loss_utils.calc_loss(real_imgs, real_imgs, img_list)
        e_loss = 0.5 * e_loss1 + 0.5 * e_loss2
        e_loss.backward(retain_graph=True)
        optimizer_e.step()

        loss_dict["e"] = e_loss

        accumulate(model.generator_test, g_module, accum)
        loss_reduced = reduce_loss_dict(loss_dict)
        e_loss_val = loss_reduced["e"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                f"e: {e_loss_val:.4f}"
            )
            if i == 0:
                # 保存输入图片
                utils.save_image(eval_imgs,
                                 os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'encoder', f"samples/eval.png"),
                                 nrow=1,
                                 normalize=True,
                                 value_range=(-1, 1), )

            if i % 1000 == 0 or i == opt.start_iter: # 每1000步保存图片
            # if i % 1 == 0 or i == opt.start_iter: # 每1000步保存图片
                with torch.no_grad():
                    samples = torch.Tensor(0, 3, opt.size, opt.size)
                    thumb_samples = torch.Tensor(0, 3, 64, 64)
                    step_size = 8  # 设置角度数量
                    mu, logvar = model.encoder(real_imgs)
                    sample_z = [reparameterize(mu, logvar).repeat_interleave(8, dim=0)]
                    mean_latent = g_module.mean_latent(10000, device)  # 截断编码 list[[1,256],[1,512]]

                    # psp
                    # noise = model.encoder(real_imgs)
                    # # noise = noise + latent_avg.repeat(noise.shape[0], 1, 1)
                    # sample_z = [noise.repeat_interleave(8, dim=0)]  # 合成多視角需要重複多次

                    for k in range(0, opt.val_n_sample * 8, step_size):
                        # 合成单个对象的多视角图像
                        # psp
                        # curr_samples, curr_thumbs = model.generator_test([sample_z[0][k:k + step_size]],
                        #                                                  sample_cam_extrinsics[k:k + step_size],
                        #                                                  sample_focals[k:k + step_size],
                        #                                                  sample_near[k:k + step_size],
                        #                                                  sample_far[k:k + step_size],
                        #                                                  input_is_latent=True)
                        curr_samples, curr_thumbs = model.generator_test([sample_z[0][k:k + step_size]],
                                                                         sample_cam_extrinsics[k:k + step_size],
                                                                         sample_focals[k:k + step_size],
                                                                         sample_near[k:k + step_size],
                                                                         sample_far[k:k + step_size],
                                                                         truncation=0.5,
                                                                         truncation_latent=mean_latent)
                        samples = torch.cat([samples, curr_samples.cpu()], 0)
                        thumb_samples = torch.cat([thumb_samples, curr_thumbs.cpu()], 0)

                    if i % 10000 == 0:
                        utils.save_image(samples,
                                         os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'encoder',
                                                      f"samples/{str(i).zfill(7)}.png"),
                                         nrow=int(opt.val_n_sample),
                                         normalize=True,
                                         value_range=(-1, 1), )
                        utils.save_image(thumb_samples,
                                         os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'encoder',
                                                      f"samples/{str(i).zfill(7)}_thumb.png"),
                                         nrow=int(opt.val_n_sample),
                                         normalize=True,
                                         value_range=(-1, 1), )

            if wandb and opt.wandb:
                wandb_log_dict = {"Encoder": e_loss_val,
                                  }
                if i % 5000 == 0:
                    wandb_grid = utils.make_grid(samples, nrow=int(opt.val_n_sample),
                                                 normalize=True, value_range=(-1, 1))
                    wandb_ndarr = (255 * wandb_grid.permute(1, 2, 0).numpy()).astype(np.uint8)
                    wandb_images = Image.fromarray(wandb_ndarr)
                    wandb_log_dict.update({"examples": [wandb.Image(wandb_images,
                                                                    caption="Generated samples for azimuth angles of: -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35 Radians.")]})
                wandb.log(wandb_log_dict)

            if i % 10000 == 0:
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": model.generator_test.state_dict(),
                    },
                    os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'encoder',
                                 f"models_{str(i).zfill(7)}.pt")
                )
                print('Successfully saved checkpoint for iteration {}.'.format(i))

    if get_rank() == 0:
        # create final model directory
        torch.save(
            {
                "e": e_module.state_dict(),
                "g": g_module.state_dict(),
                "d": d_module.state_dict(),
                "g_ema": model.generator_test.state_dict(),
            },
            os.path.join(opt.checkpoints_dir, experiment_opt.expname, experiment_opt.expname + '.pt')
        )
        print('Successfully saved final model.')