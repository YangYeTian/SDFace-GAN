import torch
import random
import trimesh
import numpy as np
import lmdb
from PIL import Image
from io import BytesIO
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch import distributed as dist
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy.spatial import Delaunay
from skimage.measure import marching_cubes
from pdb import set_trace as st
import configargparse
from munch import *

import pytorch3d.io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)

def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()

######################### Dataset util functions ###########################
# Get data sampler
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


# Get data minibatch
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


############################## Model weights util functions #################
# Turn model gradients on/off
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# Exponential moving average for generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


################### Latent code (Z) sampling util functions ####################
# Sample Z space latent codes for the generator
def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


################# Camera parameters sampling ####################
def generate_camera_params(resolution, device, batch=1, locations=None, sweep=False,
                           uniform=False, azim_range=0.3, elev_range=0.15,
                           fov_ang=6, dist_radius=0.12):
    if locations != None:
        azim = locations[:, 0].view(-1, 1)
        elev = locations[:, 1].view(-1, 1)

        # generate intrinsic parameters
        # fix distance to 1
        dist = torch.ones(azim.shape[0], 1, device=device)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(azim.shape[0], 1, device=device).view(-1, 1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    elif sweep:
        # generate camera locations on the unit sphere
        azim = (-azim_range + (2 * azim_range / 7) * torch.arange(8, device=device)).view(-1, 1).repeat(batch, 1)
        elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device).repeat(1, 8).view(-1, 1))

        # generate intrinsic parameters
        dist = (torch.ones(batch, 1, device=device)).repeat(1, 8).view(-1, 1)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(batch, 1, device=device).repeat(1, 8).view(-1, 1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    else:
        # sample camera locations on the unit sphere
        if uniform:
            azim = (-azim_range + 2 * azim_range * torch.rand(batch, 1, device=device))
            elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device))
        else:
            azim = (azim_range * torch.randn(batch, 1, device=device))
            elev = (elev_range * torch.randn(batch, 1, device=device))

        # generate intrinsic parameters
        dist = torch.ones(batch, 1, device=device)  # restrict camera position to be on the unit sphere
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(batch, 1, device=device) * np.pi / 180  # full fov is 12 degrees
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)

    viewpoint = torch.cat([azim, elev], 1)

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).view(-1, 3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    up = torch.tensor([[0, 1, 0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(camera_dir, eps=1e-5)  # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = camera_loc[:, :, None]
    extrinsics = torch.cat((R.transpose(1, 2), T), -1)

    return extrinsics, focal, near, far, viewpoint


#################### Mesh generation util functions ########################
# Reshape sampling volume to camera frostum
def align_volume(volume, near=0.88, far=1.12):
    b, h, w, d, c = volume.shape
    yy, xx, zz = torch.meshgrid(torch.linspace(-1, 1, h),
                                torch.linspace(-1, 1, w),
                                torch.linspace(-1, 1, d))

    grid = torch.stack([xx, yy, zz], -1).to(volume.device)

    frostum_adjustment_coeffs = torch.linspace(far / near, 1, d).view(1, 1, 1, -1, 1).to(volume.device)
    frostum_grid = grid.unsqueeze(0)
    frostum_grid[..., :2] = frostum_grid[..., :2] * frostum_adjustment_coeffs
    out_of_boundary = torch.any((frostum_grid.lt(-1).logical_or(frostum_grid.gt(1))), -1, keepdim=True)
    frostum_grid = frostum_grid.permute(0, 3, 1, 2, 4).contiguous()
    permuted_volume = volume.permute(0, 4, 3, 1, 2).contiguous()
    final_volume = F.grid_sample(permuted_volume, frostum_grid, padding_mode="border", align_corners=True)
    final_volume = final_volume.permute(0, 3, 4, 2, 1).contiguous()
    # set a non-zero value to grid locations outside of the frostum to avoid marching cubes distortions.
    # It happens because pytorch grid_sample uses zeros padding.
    final_volume[out_of_boundary] = 1

    return final_volume


# Extract mesh with marching cubes  使用行进立方体提取网格
def extract_mesh_with_marching_cubes(sdf):
    b, h, w, d, _ = sdf.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    sdf_vol = sdf[0, ..., 0].permute(1, 0, 2).cpu().numpy()

    # scale vertices
    verts, faces, _, _ = marching_cubes(sdf_vol, 0)
    verts[:, 0] = (verts[:, 0] / float(w) - 0.5) * 0.24
    verts[:, 1] = (verts[:, 1] / float(h) - 0.5) * 0.24
    verts[:, 2] = (verts[:, 2] / float(d) - 0.5) * 0.24

    # fix normal direction
    verts[:, 2] *= -1;
    verts[:, 1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh


# Generate mesh from xyz point cloud  从xyz点云生成网格
def xyz2mesh(xyz):
    b, _, h, w = xyz.shape
    x, y = np.meshgrid(np.arange(h), np.arange(w))

    # Extract mesh faces from xyz maps
    tri = Delaunay(np.concatenate((x.reshape((h * w, 1)), y.reshape((h * w, 1))), 1))
    faces = tri.simplices

    # invert normals
    faces[:, [0, 1]] = faces[:, [1, 0]]

    # generate_meshes
    mesh = trimesh.Trimesh(xyz.squeeze(0).permute(1, 2, 0).view(h * w, 3).cpu().numpy(), faces)

    return mesh


################# Mesh rendering util functions #############################
def add_textures(meshes: Meshes, vertex_colors=None) -> Meshes:
    verts = meshes.verts_padded()
    if vertex_colors is None:
        vertex_colors = torch.ones_like(verts)  # (N, V, 3)
    textures = TexturesVertex(verts_features=vertex_colors)
    meshes_t = Meshes(
        verts=verts,
        faces=meshes.faces_padded(),
        textures=textures,
        verts_normals=meshes.verts_normals_padded(),
    )
    return meshes_t


def create_cameras(
        R=None, T=None,
        azim=0, elev=0., dist=1.,
        fov=12., znear=0.01,
        device="cuda") -> FoVPerspectiveCameras:
    """
    all the camera parameters can be a single number, a list, or a torch tensor.
    """
    if R is None or T is None:
        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, fov=fov)
    return cameras


def create_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return phong_renderer


## custom renderer
class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def create_depth_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=17,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=((-0.5, 1., 5.0),), **light_kwargs
    )
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
            device=device,
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return renderer


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def get_ckpt_nums(folder_path):
    import os
    import re

    # 列出文件夹中所有文件名
    file_names = os.listdir(folder_path)
    # 使用正则表达式匹配文件名中的数字部分
    pattern = r'models_(\d+)\.pt'
    model_numbers = []
    for file_name in file_names:
        match = re.match(pattern, file_name)
        if match:
            number = int(match.group(1))
            model_numbers.append(number)
    if model_numbers:
        # 输出提取到的数字
        max_ckpt = str(max(model_numbers))
    else:
        max_ckpt = None
    return max_ckpt


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, nerf_resolution=64):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
        # print(txn)
        buffer = BytesIO(img_bytes)

        # Image.open(path)
        img = Image.open(buffer)
        if random.random() > 0.5:
            img = TF.hflip(img)

        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution), Image.HAMMING)
        img = self.transform(img)
        thumb_img = self.transform(thumb_img)

        return img, thumb_img


class SDFOptions():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_path", type=str, default='./data/ffhq', help="path to the lmdb dataset")

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config', is_config_file=True, help='config file path')
        # experiment.add_argument("--expname", type=str, default='debug', help='experiment name')
        experiment.add_argument("--expname", type=str, default='ffhq1024x1024', help='experiment name')
        experiment.add_argument("--ckpt", type=str, default='300000', help="path to the checkpoints to resume training")
        experiment.add_argument("--continue_training", action="store_true", help="continue training the model")

        # Training loop options
        training = self.parser.add_argument_group('training')
        training.add_argument("--checkpoints_dir", type=str, default='./out', help='checkpoints directory name')
        training.add_argument("--iter", type=int, default=300000, help="total number of training iterations")  # 训练迭代的总数
        training.add_argument("--batch", type=int, default=4,
                              help="batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory.")
        training.add_argument("--chunk", type=int, default=1,
                              help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        training.add_argument("--val_n_sample", type=int, default=8,
                              help="number of test samples generated during training")
        # 将r1正则化应用于StyleGAN生成器的间隔
        training.add_argument("--d_reg_every", type=int, default=16,
                              help="interval for applying r1 regularization to the StyleGAN generator")
        # 将路径长度正则化应用于StyleGAN生成器的区间
        training.add_argument("--g_reg_every", type=int, default=4,
                              help="interval for applying path length regularization to the StyleGAN generator")
        training.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        training.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        training.add_argument("--lr", type=float, default=0.002, help="learning rate")
        training.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        training.add_argument("--view_lambda", type=float, default=15, help="weight of the viewpoint regularization")
        training.add_argument("--eikonal_lambda", type=float, default=0.1, help="weight of the eikonal regularization")
        training.add_argument("--min_surf_lambda", type=float, default=0.05,
                              help="weight of the minimal surface regularization")
        training.add_argument("--min_surf_beta", type=float, default=100.0,
                              help="weight of the minimal surface regularization")
        training.add_argument("--path_regularize", type=float, default=2,
                              help="weight of the path length regularization")
        training.add_argument("--path_batch_shrink", type=int, default=2,
                              help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        training.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        # 不要使用球体SDF初始化体积渲染器
        training.add_argument("--no_sphere_init", action="store_true",
                              help="do not initialize the volume renderer with a sphere SDF")

        # Inference Options
        inference = self.parser.add_argument_group('inference')
        inference.add_argument("--results_dir", type=str, default='./evaluations',
                               help='results/evaluations directory name')
        # 截断比控制分集与质量的折衷。更高的截断率将产生更多样的结果
        inference.add_argument("--truncation_ratio", type=float, default=0.5,
                               help="truncation ratio, controls the diversity vs. quality tradeoff. Higher truncation ratio would generate more diverse results")
        inference.add_argument("--truncation_mean", type=int, default=10000,
                               help="number of vectors to calculate mean for the truncation")
        inference.add_argument("--identities", type=int, default=16, help="number of identities to be generated")
        inference.add_argument("--num_views_per_id", type=int, default=1,
                               help="number of viewpoints generated per identity")  # 每个对象生成的视点数
        # 当为true时，将仅生成RGB输出。否则，将生成RGB和深度视频渲染。这减少了每个视频的处理时间
        inference.add_argument("--no_surface_renderings", action="store_true",
                               help="when true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. this cuts the processing time per video")
        inference.add_argument("--fixed_camera_angles", action="store_true",
                               help="when true, the generator will render indentities from a fixed set of camera angles.")  # 如果为true，生成器将从一组固定的摄影机角度渲染缩进。
        inference.add_argument("--azim_video", action="store_true",
                               help="when true, the camera trajectory will travel along the azimuth direction. Otherwise, the camera will travel along an ellipsoid trajectory.")

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--size", type=int, default=256, help="image sizes for the model")
        model.add_argument("--style_dim", type=int, default=256, help="number of style input dimensions")
        model.add_argument("--channel_multiplier", type=int, default=2,
                           help="channel multiplier factor for the StyleGAN decoder. config-f = 2, else = 1")
        model.add_argument("--n_mlp", type=int, default=8, help="number of mlp layers in stylegan's mapping network")
        model.add_argument("--lr_mapping", type=float, default=0.01,
                           help='learning rate reduction for mapping network MLP layers')
        model.add_argument("--renderer_spatial_output_dim", type=int, default=64,
                           help='spatial resolution of the StyleGAN decoder inputs')
        model.add_argument("--project_noise", action='store_true',
                           help='when true, use geometry-aware noise projection to reduce flickering effects (see supplementary section C.1 in the paper). warning: processing time significantly increases with this flag to ~20 minutes per video.')

        # Camera options
        camera = self.parser.add_argument_group('camera')
        # uniform 如果为true，则从均匀分布中对相机位置进行采样。高斯分布是默认值
        camera.add_argument("--uniform", action="store_true",
                            help="when true, the camera position is sampled from uniform distribution. Gaussian distribution is the default")
        camera.add_argument("--azim", type=float, default=0.3, help="camera azimuth angle std/range in Radians")
        camera.add_argument("--elev", type=float, default=0.15, help="camera elevation angle std/range in Radians")
        camera.add_argument("--fov", type=float, default=6,
                            help="camera field of view half angle in Degrees")  # 摄像机视场半角（单位：度）
        camera.add_argument("--dist_radius", type=float, default=0.12,
                            help="radius of points sampling distance from the origin. determines the near and far fields")

        # Volume Renderer options
        rendering = self.parser.add_argument_group('rendering')
        # MLP model parameters
        rendering.add_argument("--depth", type=int, default=8, help='layers in network')
        rendering.add_argument("--width", type=int, default=256, help='channels per layer')
        # Volume representation options
        # 默认情况下，原始MLP输出表示带下划线符号的距离字段（SDF）。当为true时，MLP输出表示传统的NeRF密度场。
        rendering.add_argument("--no_sdf", action='store_true',
                               help='By default, the raw MLP outputs represent an underline signed distance field (SDF). When true, the MLP outputs represent the traditional NeRF density field.')
        rendering.add_argument("--no_z_normalize", action='store_true',
                               help='By default, the model normalizes input coordinates such that the z coordinate is in [-1,1]. When true that feature is disabled.')
        rendering.add_argument("--static_viewdirs", action='store_true',
                               help='when true, use static viewing direction input to the MLP')
        # Ray intergration options
        rendering.add_argument("--N_samples", type=int, default=24, help='number of samples per ray')
        rendering.add_argument("--no_offset_sampling", action='store_true',
                               help='when true, use random stratified sampling when rendering the volume, otherwise offset sampling is used. (See Equation (3) in Sec. 3.2 of the paper)')
        rendering.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
        rendering.add_argument("--raw_noise_std", type=float, default=0.,
                               help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
        rendering.add_argument("--force_background", action='store_true',
                               help='force the last depth sample to act as background in case of a transparent ray')
        # Set volume renderer outputs
        rendering.add_argument("--return_xyz", action='store_true',
                               help='when true, the volume renderer also returns the xyz point could of the surface. This point cloud is used to produce depth map renderings')
        rendering.add_argument("--return_sdf", action='store_true',
                               help='when true, the volume renderer also returns the SDF network outputs for each location in the volume')

        self.initialized = True

    # 需要自定义参数时，输入对应的列表，如：["--batch", "6"]
    def parse(self, input=[]):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()
        try:
            # 输入args为空则读取命令行的参数
            args = self.parser.parse_args(input)
        except:  # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)

        return self.opt
