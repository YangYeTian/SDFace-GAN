import random
from im2scene.sdf.models.sdf_op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

from im2scene.sdf.models.sdf_utils import (
    create_cameras,
    add_textures,
    create_depth_mesh_renderer,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_euler_angles


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from numpy import pi


# Basic SIREN fully connected layer
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init

        return out

# Siren layer with frequency modulation and offset
class FiLMSiren(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(nn.Parameter(nn.init.uniform_(torch.empty(out_channel), a=-np.sqrt(1/in_channel), b=np.sqrt(1/in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)

    def forward(self, input, style):
        batch, features = style.shape
        out = F.linear(input, self.weight, bias=self.bias)
        gamma = self.gamma(style).view(batch, 1, 1, 1, features)
        beta = self.beta(style).view(batch, 1, 1, 1, features)

        out = self.activation(gamma * out + beta)

        return out


class FiLMSURF(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(nn.Parameter(nn.init.uniform_(torch.empty(out_channel), a=-np.sqrt(1/in_channel), b=np.sqrt(1/in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)

    def forward(self, input, style):
        batch, features = style.shape
        out = F.linear(input, self.weight, bias=self.bias)
        gamma = self.gamma(style).view(batch, 1, 1, 1, features)
        beta = self.beta(style).view(batch, 1, 1, 1, features)

        out = self.activation(gamma * out + beta) + input

        return out


# Siren Generator Model
class SirenGenerator(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True):
        super(SirenGenerator, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.pts_linears = nn.ModuleList(
            [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])

        self.views_linears = FiLMSiren(input_ch_views + W, W,
                                       style_dim=style_dim)
        self.rgb_linear = LinearLayer(W, 3, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def forward(self, x, styles):
        # 拆分x，d [1, 64, 64, 24, 3]
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()  # 拷贝x
        for i in range(len(self.pts_linears)):  # 8层调制MLP  SIREN块
            mlp_out = self.pts_linears[i](mlp_out, styles)  # 需要输入x和z

        # sdf层
        sdf = self.sigma_linear(mlp_out)  # [1, 64, 64, 24, 1]
        # 颜色层
        mlp_out = torch.cat([mlp_out, input_views], -1)  # 输入方向d
        out_features = self.views_linears(mlp_out, styles)  # [1, 64, 64, 24, 256]
        rgb = self.rgb_linear(out_features)  # [1, 64, 64, 24, 3]

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)
        # outputs【rgb，sdf，out_features】 在最后一个维度拼接  [1, 64, 64, 24, 3+1+256]
        return outputs


# Full volume renderer
class VolumeFeatureRenderer(nn.Module):
    def __init__(self, opt, style_dim=256, out_im_res=64, mode='train'):
        super().__init__()
        self.test = mode != 'train'
        self.perturb = opt.perturb
        self.offset_sampling = not opt.no_offset_sampling # Stratified sampling used otherwise
        self.N_samples = opt.N_samples
        self.raw_noise_std = opt.raw_noise_std
        self.return_xyz = opt.return_xyz
        self.return_sdf = opt.return_sdf
        self.static_viewdirs = opt.static_viewdirs
        self.z_normalize = not opt.no_z_normalize
        self.out_im_res = out_im_res  # 输出分辨率
        self.force_background = opt.force_background
        self.with_sdf = not opt.no_sdf
        if 'no_features_output' in opt.keys():
            self.output_features = False
        else:
            self.output_features = True

        if self.with_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        # create meshgrid to generate rays
        i, j = torch.meshgrid(torch.linspace(0.5, self.out_im_res - 0.5, self.out_im_res),
                              torch.linspace(0.5, self.out_im_res - 0.5, self.out_im_res))

        self.register_buffer('i', i.t().unsqueeze(0), persistent=False)
        self.register_buffer('j', j.t().unsqueeze(0), persistent=False)

        # create integration values
        if self.offset_sampling:
            t_vals = torch.linspace(0., 1.-1/self.N_samples, steps=self.N_samples).view(1,1,1,-1)
        else: # Original NeRF Stratified sampling
            t_vals = torch.linspace(0., 1., steps=self.N_samples).view(1,1,1,-1)

        self.register_buffer('t_vals', t_vals, persistent=False)
        self.register_buffer('inf', torch.Tensor([1e10]), persistent=False)
        self.register_buffer('zero_idx', torch.LongTensor([0]), persistent=False)

        if self.test:
            self.perturb = False
            self.raw_noise_std = 0.

        self.channel_dim = -1
        self.samples_dim = 3
        self.input_ch = 3
        self.input_ch_views = 3
        self.feature_out_size = opt.width if not opt.type == "ngp" else style_dim

        if opt.type == "ngp":
            # set NGPSiren Generator model
            self.network = NGPSIRENGenerator(D=2, W=style_dim, style_dim=style_dim, output_features=self.output_features)
        else:
            if opt.fc:
                self.network = FCGenerator(D=opt.depth, W=opt.width, style_dim=style_dim, input_ch=self.input_ch,
                                              output_ch=4, input_ch_views=self.input_ch_views,
                                              output_features=self.output_features)
            else:
                # set Siren Generator model
                self.network = SirenGenerator(D=opt.depth, W=opt.width, style_dim=style_dim, input_ch=self.input_ch,
                                              output_ch=4, input_ch_views=self.input_ch_views,
                                              output_features=self.output_features)

    def get_rays(self, focal, c2w):
        dirs = torch.stack([(self.i - self.out_im_res * .5) / focal,
                            -(self.j - self.out_im_res * .5) / focal,
                            -torch.ones_like(self.i).expand(focal.shape[0],self.out_im_res, self.out_im_res)], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * c2w[:,None,None,:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:,None,None,:3,-1].expand(rays_d.shape)
        if self.static_viewdirs:
            viewdirs = dirs
        else:
            viewdirs = rays_d

        return rays_o, rays_d, viewdirs

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                     grad_outputs=torch.ones_like(sdf),
                                     create_graph=True)[0]

        return eikonal_term

    def sdf_activation(self, input):
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta

        return sigma

    def volume_integration(self, raw, z_vals, rays_d, pts, return_eikonal=False):
        dists = z_vals[...,1:] - z_vals[...,:-1]  # 计算距离差值δ，用于计算体积渲染公式
        rays_d_norm = torch.norm(rays_d.unsqueeze(self.samples_dim), dim=self.channel_dim)
        # dists still has 4 dimensions here instead of 5, hence, in this case samples dim is actually the channel dim
        dists = torch.cat([dists, self.inf.expand(rays_d_norm.shape)], self.channel_dim)  # [N_rays, N_samples]
        dists = dists * rays_d_norm

        # If sdf modeling is off, the sdf variable stores the
        # pre-integration raw sigma MLP outputs.
        if self.output_features:
            rgb, sdf, features = torch.split(raw, [3, 1, self.feature_out_size], dim=self.channel_dim)  # 拆分网络输出
        else:
            rgb, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)

        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn_like(sdf) * self.raw_noise_std

        if self.with_sdf:
            sigma = self.sdf_activation(-sdf)  # 在通过一次变换得到sigma

            if return_eikonal:
                eikonal_term = self.get_eikonal_term(pts, sdf)
            else:
                eikonal_term = None

            sigma = 1 - torch.exp(-sigma * dists.unsqueeze(self.channel_dim))
        else:  # 如果sdf建模关闭，则sdf变量存储积分前的原始sigmaMLP输出。
            sigma = sdf
            eikonal_term = None

            sigma = 1 - torch.exp(-F.softplus(sigma + noise) * dists.unsqueeze(self.channel_dim))
        # 体积渲染公式
        visibility = torch.cumprod(torch.cat([torch.ones_like(torch.index_select(sigma, self.samples_dim, self.zero_idx)),
                                              1.-sigma + 1e-10], self.samples_dim), self.samples_dim)
        visibility = visibility[..., :-1, :]
        weights = sigma * visibility

        if self.return_sdf:
            sdf_out = sdf  # 输出维度是128*128*128，此时rgb三[1, 128, 128, 128, 3]，sdf是[1, 128, 128, 128, 1]
        else:
            sdf_out = None
        # 体积渲染公式  计算出rgb值
        if self.force_background:
            weights[..., -1, :] = 1 - weights[..., :-1, :].sum(self.samples_dim)

        rgb_map = -1 + 2 * torch.sum(weights * torch.sigmoid(rgb), self.samples_dim)  # switch to [-1,1] value range

        if self.output_features:  # 计算体积渲染特征图
            feature_map = torch.sum(weights * features, self.samples_dim)
        else:
            feature_map = None

        # Return surface point cloud in world coordinates.
        # This is used to generate the depth maps visualizations.
        # We use world coordinates to avoid transformation errors between
        # surface renderings from different viewpoints.
        # 返回世界坐标中的曲面点云。这用于生成深度图可视化。我们使用世界坐标来避免不同视点的曲面渲染之间的变换误差。
        if self.return_xyz:
            xyz = torch.sum(weights * pts, self.samples_dim)  # [1, 128, 128, 3]
            mask = weights[..., -1, :]  # background probability map
        else:
            xyz = None
            mask = None

        return rgb_map, feature_map, sdf_out, mask, xyz, eikonal_term

    def run_network(self, inputs, viewdirs, styles=None):
        input_dirs = viewdirs.unsqueeze(self.samples_dim).expand(inputs.shape)  # 将d扩展到跟x同一维度
        net_inputs = torch.cat([inputs, input_dirs], self.channel_dim)  # 合并x和d
        outputs = self.network(net_inputs, styles=styles)
        # outputs【rgb，sdf，out_features】 在最后一个维度拼接  [1, 64, 64, 24, 3+1+256]
        return outputs

    def render_rays(self, ray_batch, styles=None, return_eikonal=False):
        batch, h, w, _ = ray_batch.shape
        split_pattern = [3, 3, 2]
        # 拆分光线
        if ray_batch.shape[-1] > 8:
            split_pattern += [3]
            rays_o, rays_d, bounds, viewdirs = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
        else:
            rays_o, rays_d, bounds = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
            viewdirs = None

        near, far = torch.split(bounds, [1, 1], dim=self.channel_dim)
        # t_vals是将[0, 1]等距离分割等24个点
        # z_vals将0到1之间变换为near和far之间  [1, 64, 64, 24]
        z_vals = near * (1.-self.t_vals) + far * (self.t_vals)

        if self.perturb > 0.:  # 随机取样
            if self.offset_sampling:
                # random offset samples
                upper = torch.cat([z_vals[...,1:], far], -1)
                lower = z_vals.detach()
                t_rand = torch.rand(batch, h, w).unsqueeze(self.channel_dim).to(z_vals.device)
            else:
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(z_vals.device)

            z_vals = lower + (upper - lower) * t_rand

        # 获取pts  [1, 64, 64, 24, 3]
        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(self.samples_dim) * z_vals.unsqueeze(self.channel_dim)

        if return_eikonal:
            pts.requires_grad = True

        if self.z_normalize:  # 归一化pts
            normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))
        else:
            normalized_pts = pts
        # =====================
        # 以上部分为光线采样获取pts
        # =====================

        # 调用run_network使潜在编码z可以跟x、d一起输入网络
        raw = self.run_network(normalized_pts, viewdirs, styles=styles)  # 调用 SirenGenerator.forward
        # raw【rgb，sdf，out_features】 在最后一个维度拼接  [1, 64, 64, 24, 3+1+256]
        rgb_map, features, sdf, mask, xyz, eikonal_term = self.volume_integration(raw, z_vals, rays_d, pts, return_eikonal=return_eikonal)

        return rgb_map, features, sdf, mask, xyz, eikonal_term

    def render(self, focal, c2w, near, far, styles, c2w_staticcam=None, return_eikonal=False):
        # 生成光线， ro和rd构成光线
        # viewdirs 为归一化的 rd，作为nerf输入
        rays_o, rays_d, viewdirs = self.get_rays(focal, c2w)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # [1, 64, 64, 3] 图像是64*64，共生成这么多光线

        # Create ray batch
        near = near.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, viewdirs], -1)
        rays = rays.float()
        rgb, features, sdf, mask, xyz, eikonal_term = self.render_rays(rays, styles=styles, return_eikonal=return_eikonal)

        return rgb, features, sdf, mask, xyz, eikonal_term

    def mlp_init_pass(self, cam_poses, focal, near, far, styles=None):
        rays_o, rays_d, viewdirs = self.get_rays(focal, cam_poses)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        near = near.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        z_vals = near * (1.-self.t_vals) + far * (self.t_vals)

        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(z_vals.device)

        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(self.samples_dim) * z_vals.unsqueeze(self.channel_dim)
        if self.z_normalize:
            normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))
        else:
            normalized_pts = pts

        raw = self.run_network(normalized_pts, viewdirs, styles=styles)
        _, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)
        sdf = sdf.squeeze(self.channel_dim)
        # detach返回一个新的向量，与原向量共用地址，但是不修改梯度
        #
        target_values = pts.detach().norm(dim=-1) - ((far - near) / 4)

        return sdf, target_values

    def forward(self, cam_poses, focal, near, far, styles=None, return_eikonal=False):

        rgb, features, sdf, mask, xyz, eikonal_term = self.render(focal, c2w=cam_poses, near=near, far=far, styles=styles, return_eikonal=return_eikonal)

        rgb = rgb.permute(0,3,1,2).contiguous()
        if self.output_features:
            features = features.permute(0,3,1,2).contiguous()  # 维度变换  [1, 256, 64, 64]

        if xyz != None:  # xyz [1, 128, 128, 3]  mask [1, 128, 128, 1]
            xyz = xyz.permute(0,3,1,2).contiguous()  # 维度变换 [1, 3, 128, 128]
            mask = mask.permute(0,3,1,2).contiguous()  # 维度变换 [1, 1, 128, 128]

        return rgb, features, sdf, mask, xyz, eikonal_term

    #
    # def query_sdf(self, pts, styles):
    #     return self.network.query_sdf(pts, styles)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MappingLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, is_last=False):
        super().__init__()
        if is_last:
            weight_std = 0.25
        else:
            weight_std = 1

        self.weight = nn.Parameter(weight_std * nn.init.kaiming_normal_(torch.empty(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        if bias:
            self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))
        else:
            self.bias = None

        self.activation = activation

    def forward(self, input):
        if self.activation != None:
            out = F.linear(input, self.weight)
            out = fused_leaky_relu(out, self.bias, scale=1)
        else:
            out = F.linear(input, self.weight, bias=self.bias)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
                 activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale,
                           bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True,
                 upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self, project=False):
        super().__init__()
        self.project = project
        self.weight = nn.Parameter(torch.zeros(1))
        self.prev_noise = None
        self.mesh_fn = None
        self.vert_noise = None

    def create_pytorch_mesh(self, trimesh):
        v=trimesh.vertices; f=trimesh.faces
        verts = torch.from_numpy(np.asarray(v)).to(torch.float32).cuda()
        mesh_pytorch = Meshes(
            verts=[verts],
            faces = [torch.from_numpy(np.asarray(f)).to(torch.float32).cuda()],
            textures=None
        )
        if self.vert_noise == None or verts.shape[0] != self.vert_noise.shape[1]:
            self.vert_noise = torch.ones_like(verts)[:,0:1].cpu().normal_().expand(-1,3).unsqueeze(0)

        mesh_pytorch = add_textures(meshes=mesh_pytorch, vertex_colors=self.vert_noise.to(verts.device))

        return mesh_pytorch

    def load_mc_mesh(self, filename, resolution=128, im_res=64):
        import trimesh

        mc_tri=trimesh.load_mesh(filename)
        v=mc_tri.vertices; f=mc_tri.faces
        mesh2=trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==64 or im_res==128:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh2)
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(v,f)
        mesh2_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==256:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh2_subdiv);
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(mesh2_subdiv.vertices,mesh2_subdiv.faces)
        mesh3_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==256:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh3_subdiv);
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(mesh3_subdiv.vertices,mesh3_subdiv.faces)
        mesh4_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)

        pytorch3d_mesh = self.create_pytorch_mesh(mesh4_subdiv)

        return pytorch3d_mesh

    def project_noise(self, noise, transform, mesh_path=None):
        batch, _, height, width = noise.shape
        assert(batch == 1)  # assuming during inference batch size is 1

        angles = matrix_to_euler_angles(transform[0:1,:,:3], "ZYX")
        azim = float(angles[0][1])
        elev = float(-angles[0][2])

        cameras = create_cameras(azim=azim*180/np.pi,elev=elev*180/np.pi,fov=12.,dist=1)

        renderer = create_depth_mesh_renderer(cameras, image_size=height,
                specular_color=((0,0,0),), ambient_color=((1.,1.,1.),),diffuse_color=((0,0,0),))


        if self.mesh_fn is None or self.mesh_fn != mesh_path:
            self.mesh_fn = mesh_path

        pytorch3d_mesh = self.load_mc_mesh(mesh_path, im_res=height)
        rgb, depth = renderer(pytorch3d_mesh)

        depth_max = depth.max(-1)[0].view(-1) # (NxN)
        depth_valid = depth_max > 0.
        if self.prev_noise is None:
            self.prev_noise = noise
        noise_copy = self.prev_noise.clone()
        noise_copy.view(-1)[depth_valid] = rgb[0,:,:,0].view(-1)[depth_valid]
        noise_copy = noise_copy.reshape(1,1,height,height)  # 1x1xNxN

        return noise_copy


    def forward(self, image, noise=None, transform=None, mesh_path=None):
        batch, _, height, width = image.shape
        if noise is None:
            noise = image.new_empty(batch, 1, height, width).normal_()
        elif self.project:
            noise = self.project_noise(noise, transform, mesh_path=mesh_path)

        return image + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 upsample=False, blur_kernel=[1, 3, 3, 1], project_noise=False):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.noise = NoiseInjection(project=project_noise)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, transform=None, mesh_path=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise, transform=transform, mesh_path=mesh_path)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.upsample = upsample
        out_channels = 3
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channels, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False,
                 blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class Decoder(nn.Module):
    def __init__(self, model_opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        # decoder mapping network
        self.size = model_opt.size
        self.style_dim = model_opt.style_dim * 2
        thumb_im_size = model_opt.renderer_spatial_output_dim

        if not model_opt.psp:
            # 后五层mapping层
            layers = [PixelNorm(),
                       EqualLinear(
                           self.style_dim // 2, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
                       )]
        else:
            layers = [PixelNorm(),
                       EqualLinear(
                           self.style_dim, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
                       )]

        for i in range(4):
            layers.append(
                EqualLinear(
                    self.style_dim, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        # decoder network
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * model_opt.channel_multiplier,
            128: 128 * model_opt.channel_multiplier,
            256: 64 * model_opt.channel_multiplier,
            512: 32 * model_opt.channel_multiplier,
            1024: 16 * model_opt.channel_multiplier,
        }

        decoder_in_size = model_opt.renderer_spatial_output_dim

        # image decoder
        self.log_size = int(math.log(self.size, 2))
        self.log_in_size = int(math.log(decoder_in_size, 2))

        input_feature_channels = model_opt.feature_encoder_in_channels if not model_opt.psp else self.style_dim

        self.conv1 = StyledConv(
            input_feature_channels,
            self.channels[decoder_in_size], 3, self.style_dim, blur_kernel=blur_kernel,
            project_noise=model_opt.project_noise)

        self.to_rgb1 = ToRGB(self.channels[decoder_in_size], self.style_dim, upsample=False)

        self.num_layers = (self.log_size - self.log_in_size) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[decoder_in_size]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 2 * self.log_in_size + 1) // 2
            shape = [1, 1, 2 ** (res), 2 ** (res)]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(self.log_in_size+1, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(in_channel, out_channel, 3, self.style_dim, upsample=True,
                           blur_kernel=blur_kernel, project_noise=model_opt.project_noise)
            )

            self.convs.append(
                StyledConv(out_channel, out_channel, 3, self.style_dim,
                           blur_kernel=blur_kernel, project_noise=model_opt.project_noise)
            )

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))

            in_channel = out_channel

        self.n_latent = (self.log_size - self.log_in_size) * 2 + 2

    def mean_latent(self, renderer_latent):
        latent = self.style(renderer_latent).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def styles_and_noise_forward(self, styles, noise, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False,
                                 randomize_noise=True):
        if not input_is_latent:  # 后五层mapping层
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]
        # 同浅层的添加噪音
        if (truncation < 1):
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[1] + truncation * (style - truncation_latent[1])
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            # tensor.ndim指向量的维度  将latent重复n_latent次
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return latent, noise

    # features:特征图 [1, 256, 64, 64]
    # style：浅层latnet list[1] [1,256]
    # truncation_latent：统一潜在编码  list[2]:第一项是三层潜在编码 [1, 256]，第二项三八层潜在编码 [1, 512]
    def forward(self, features, styles, rgbd_in=None, transform=None,
                return_latents=False, inject_index=None, truncation=1,
                truncation_latent=None, input_is_latent=False, noise=None,
                randomize_noise=True, mesh_path=None):
        # 给潜在编码中加入噪音
        latent, noise = self.styles_and_noise_forward(styles, noise, inject_index, truncation,
                                                      truncation_latent, input_is_latent,
                                                      randomize_noise)

        out = self.conv1(features, latent[:, 0], noise=noise[0],
                         transform=transform, mesh_path=mesh_path)

        skip = self.to_rgb1(out, latent[:, 1], skip=rgbd_in)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1,
                           transform=transform, mesh_path=mesh_path)
            out = conv2(out, latent[:, i + 1], noise=noise2,
                                   transform=transform, mesh_path=mesh_path)
            skip = to_rgb(out, latent[:, i + 2], skip=skip)

            i += 2

        out_latent = latent if return_latents else None
        image = skip  # [1, 3, 1024, 1024]

        return image, out_latent


class Generator(nn.Module):
    # 整体生成器
    def __init__(self, model_opt, renderer_opt, blur_kernel=[1, 3, 3, 1], ema=False, full_pipeline=True):
        super().__init__()
        self.size = model_opt.size
        self.style_dim = model_opt.style_dim * 2 if model_opt.psp else model_opt.style_dim
        self.num_layers = 1
        self.train_renderer = not model_opt.freeze_renderer
        self.full_pipeline = full_pipeline
        model_opt.feature_encoder_in_channels = renderer_opt.width

        if ema or 'is_test' in model_opt.keys():
            self.is_train = False
        else:
            self.is_train = True

        # volume renderer mapping_network  体渲染的mapping网络仅使用style的前三层
        layers = []
        for i in range(3):
            layers.append(
                MappingLinear(self.style_dim, self.style_dim, activation="fused_lrelu")
            )
        # if model_opt.psp:
        #     layers.append(PixelNorm())
        #     layers.append(
        #         EqualLinear(self.style_dim, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu")
        #     )
        #
        #
        #     for i in range(4):
        #         layers.append(
        #             EqualLinear(
        #                 self.style_dim, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
        #             )
        #         )
        #

        # 体渲染的mapping网络仅使用style的前三层
        self.style = nn.Sequential(*layers)

        # volume renderer  体渲染器
        thumb_im_size = model_opt.renderer_spatial_output_dim
        self.renderer = VolumeFeatureRenderer(renderer_opt, style_dim=self.style_dim,
                                              out_im_res=thumb_im_size)

        # styleGAN 模块
        if self.full_pipeline:
            self.decoder = Decoder(model_opt)

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent, device, z=None):
        if z is None:
            # 计算平均 lantent code
            latent_in = torch.randn(n_latent, self.style_dim, device=device)  # [10000, 256]
            # style 是 mapping的前三层
            renderer_latent = self.style(latent_in)  # [10000, 256]
            # tensor.mean是对第i维度取平均值
            renderer_latent_mean = renderer_latent.mean(0, keepdim=True)  # [1, 256]
        else:
            renderer_latent_mean = self.style(z)
        if self.full_pipeline:  # 使用style的两段渲染，则需要计算后半段的lantent code
            decoder_latent_mean = self.decoder.mean_latent(renderer_latent)  # [1, 512]
        else:
            decoder_latent_mean = None

        return [renderer_latent_mean, decoder_latent_mean]

    def get_latent(self, input):
        return self.style(input)

    def styles_and_noise_forward(self, styles, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False):
        if not input_is_latent:  # 如果输入的不是经过mapping层的latent，则需要经过一层mapping
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[0] + truncation * (style - truncation_latent[0])
                )

            styles = style_t

        return styles

    def init_forward(self, styles, cam_poses, focals, near=0.88, far=1.12):
        latent = self.styles_and_noise_forward(styles)

        sdf, target_values = self.renderer.mlp_init_pass(cam_poses, focals, near, far, styles=latent[0])

        return sdf, target_values

    def forward(self, styles, cam_poses, focals, near=0.88, far=1.12, return_latents=False,
                inject_index=None, truncation=1, truncation_latent=None,
                input_is_latent=False, noise=None, randomize_noise=True,
                return_sdf=False, return_xyz=False, return_eikonal=False,
                project_noise=False, mesh_path=None):
        """
        styles：随机噪音，未mapping
        truncation: 截断比控制分集与质量的折衷。更高的截断率将产生更多样的结果，截断权重，越高表示噪音占比更高
        truncation_latent：统一潜在编码  list[2]:第一项是三层潜在编码 [1, 256]，第二项三八层潜在编码 [1, 512]
        """
        # do not calculate renderer gradients if renderer weights are frozen  如果渲染器权重冻结，则不计算渲染器梯度
        with torch.set_grad_enabled(self.is_train and self.train_renderer):
            # 给潜在编码中加入噪音  latent是经过三层的潜在编码  list[1]  [1,256]
            # latent = truncation_latent + truncation * （ style - truncation_latent ）
            latent = self.styles_and_noise_forward(styles, inject_index, truncation,
                                                   truncation_latent, input_is_latent)

            # self.renderer 是 nerf 渲染器，调用 VolumeFeatureRenderer.forward  # 第一个256用于
            if input_is_latent:
                latent0 = latent[0][:, 0]
                thumb_rgb, features, sdf, mask, xyz, eikonal_term = self.renderer(cam_poses, focals, near, far, styles=latent0, return_eikonal=return_eikonal)
            else:
                thumb_rgb, features, sdf, mask, xyz, eikonal_term = self.renderer(cam_poses, focals, near, far, styles=latent[0], return_eikonal=return_eikonal)
            # thumb_rgb：低分辨率图
            # features：特征图
            # sdf mask xyz

        # 如果使用两段渲染，调用style模块
        if self.full_pipeline:
            rgb, decoder_latent = self.decoder(features, latent,
                                               transform=cam_poses if project_noise else None,
                                               return_latents=return_latents,
                                               inject_index=inject_index, truncation=truncation,
                                               truncation_latent=truncation_latent, noise=noise,
                                               input_is_latent=input_is_latent, randomize_noise=randomize_noise,
                                               mesh_path=mesh_path)

        else:
            rgb = None

        if return_latents:
            return rgb, decoder_latent
        else:
            out = (rgb, thumb_rgb)
            if return_xyz:
                out += (xyz,)
            if return_sdf:
                out += (sdf,)
            if return_eikonal:
                out += (eikonal_term,)
            if return_xyz:
                out += (mask,)

            return out

    # def query_sdf(self, pts, styles):
    #     latent = self.styles_and_noise_forward(styles, None, 1, None, False)
    #     return self.renderer.query_sdf(pts, latent[0])


############# Volume Renderer Building Blocks & Discriminator ##################
class VolumeRenderDiscConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, activate=False):
        super(VolumeRenderDiscConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias and not activate)

        self.activate = activate
        if self.activate:
            self.activation = FusedLeakyReLU(out_channels, bias=bias, scale=1)
            bias_init_coef = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
            nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)


    def forward(self, input):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: (N,C_out,H_out,W_out）
        :return: Conv2d + activation Result
        """
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_channel = torch.arange(dim_x, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_y,1)
        yy_channel = torch.arange(dim_y, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_x,1).transpose(2,3)

        xx_channel = xx_channel / (dim_x - 1)
        yy_channel = yy_channel / (dim_y - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
        out = torch.cat([input_tensor, yy_channel, xx_channel], dim=1)

        return out


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super(CoordConv2d, self).__init__()

        self.addcoords = AddCoords()
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bias=True, activate=True):
        super(CoordConvLayer, self).__init__()
        layers = []
        stride = 1
        self.activate = activate
        self.padding = kernel_size // 2 if kernel_size > 2 else 0

        self.conv = CoordConv2d(in_channel, out_channel, kernel_size,
                                padding=self.padding, stride=stride,
                                bias=bias and not activate)

        if activate:
            self.activation = FusedLeakyReLU(out_channel, bias=bias, scale=1)

        bias_init_coef = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
        nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)

    def forward(self, input):
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class VolumeRenderResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = CoordConvLayer(in_channel, out_channel, 3)
        self.conv2 = CoordConvLayer(out_channel, out_channel, 3)
        self.pooling = nn.AvgPool2d(2)
        self.downsample = nn.AvgPool2d(2)
        if out_channel != in_channel:
            self.skip = VolumeRenderDiscConv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.pooling(out)

        downsample_in = self.downsample(input)
        if self.skip != None:
            skip_in = self.skip(downsample_in)
        else:
            skip_in = downsample_in

        out = (out + skip_in) / math.sqrt(2)

        return out


class VolumeRenderDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        init_size = opt.renderer_spatial_output_dim
        self.viewpoint_loss = not opt.no_viewpoint_loss
        final_out_channel = 3 if self.viewpoint_loss else 1
        channels = {
            2: 400,
            4: 400,
            8: 400,
            16: 400,
            32: 256,
            64: 128,
            128: 64,
        }

        convs = [VolumeRenderDiscConv2d(3, channels[init_size], 1, activate=True)]

        log_size = int(math.log(init_size, 2))

        in_channel = channels[init_size]

        for i in range(log_size-1, 0, -1):
            out_channel = channels[2 ** i]

            convs.append(VolumeRenderResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = VolumeRenderDiscConv2d(in_channel, final_out_channel, 2)

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)
        gan_preds = out[:,0:1]
        gan_preds = gan_preds.view(-1, 1)
        if self.viewpoint_loss:
            viewpoints_preds = out[:,1:]
            viewpoints_preds = viewpoints_preds.view(-1,2)
        else:
            viewpoints_preds = None

        return gan_preds, viewpoints_preds

######################### StyleGAN Discriminator ########################
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], merge=False):
        super().__init__()

        self.conv1 = ConvLayer(2 * in_channel if merge else in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(2 * in_channel if merge else in_channel, out_channel,
                              1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = (out + self.skip(input)) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        init_size = opt.size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * opt.channel_multiplier,
            128: 128 * opt.channel_multiplier,
            256: 64 * opt.channel_multiplier,
            512: 32 * opt.channel_multiplier,
            1024: 16 * opt.channel_multiplier,
        }

        convs = [ConvLayer(3, channels[init_size], 1)]

        log_size = int(math.log(init_size, 2))

        in_channel = channels[init_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        # minibatch discrimination
        in_channel += 1

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        # minibatch discrimination
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        if batch % group != 0:
            group = 3 if batch % 3 == 0 else 2

        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        final_out = torch.cat([out, stddev], 1)

        # final layers
        final_out = self.final_conv(final_out)  # [chunk,512,4,4]
        final_out = final_out.view(batch, -1)  # [chunk,8192]
        feat = final_out
        final_out = self.final_linear(final_out)
        gan_preds = final_out[:,:1]

        return gan_preds

    def get_feat(self, input):
        out = self.convs(input)

        # minibatch discrimination
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        if batch % group != 0:
            group = 3 if batch % 3 == 0 else 2

        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        final_out = torch.cat([out, stddev], 1)

        # final layers
        final_out = self.final_conv(final_out)  # [chunk,512,4,4]
        final_out = final_out.view(batch, -1)  # [chunk,8192]

        return final_out


def get_encoder(encoding, input_dim=3,
                multires=6,
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048,
                align_corners=False,
                **kwargs):
    if encoding == 'sphere_harmonics':
        from im2scene.sdf.models.shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from im2scene.sdf.models.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
                              base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                              desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
    else:
        raise NotImplementedError(
            'Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim


class NGPSIRENGenerator(nn.Module):
    def __init__(self, D=2, W=256, style_dim=256, output_features=True):
        super(NGPSIRENGenerator, self).__init__()
        # sigma network
        self.D = D
        self.W = W
        self.bound = 2
        self.style_dim = style_dim
        self.input_ch = 3
        self.input_ch_views = 3
        self.output_features = output_features
        self.encoder, self.in_dim = get_encoder("hashgrid", desired_resolution=2048 * self.bound)
        self.encoder_dir, self.in_dim_dir = get_encoder("sphere_harmonics")

        # feature network
        self.input_linear = self.rgb_linear = LinearLayer(self.in_dim, self.W, freq_init=True)

        # self.pts_linears = nn.ModuleList(
        #     [FiLMSURF(self.W, self.W, style_dim=style_dim, is_first=True)] + \
        #     [FiLMSURF(self.W, self.W, style_dim=style_dim) for i in range(self.D)])
        self.pts_linears = nn.ModuleList(
            [FiLMSiren(self.W, self.W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(self.W, self.W, style_dim=style_dim) for i in range(self.D)])

        # rgb network
        self.views_linears = FiLMSiren(self.in_dim_dir + self.W, self.W,
                                       style_dim=style_dim)
        self.rgb_linear = LinearLayer(self.W, 3, freq_init=True)

        # sigma network
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def forward(self, x, styles):
        # 拆分x，d [1, 64, 64, 24, 3]
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # 分别对 x、d 进行编码
        input_pts = self.encoder(input_pts, bound=self.bound)
        input_views = self.encoder_dir(input_views)

        mlp_out = input_pts.contiguous()  # 拷贝x

        mlp_out = self.input_linear(mlp_out)

        for i in range(len(self.pts_linears)):  # 3层调制MLP  SURF
            mlp_out = self.pts_linears[i](mlp_out, styles)  # 需要输入x和z

        # sdf层
        sdf = self.sigma_linear(mlp_out)  # [1, 64, 64, 24, 1]
        # 颜色层
        mlp_out = torch.cat([mlp_out, input_views], -1)  # 输入方向d
        out_features = self.views_linears(mlp_out, styles)  # [1, 64, 64, 24, 256]
        rgb = self.rgb_linear(out_features)  # [1, 64, 64, 24, 3]

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)
        # outputs【rgb，sdf，out_features】 在最后一个维度拼接  [1, 64, 64, 24, 3+1+256]
        return outputs

    def query_sdf(self, input_pts, styles):
        embed = self.encoder(input_pts, bound=self.bound)
        return embed


class FCGenerator(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True):
        super(FCGenerator, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.n_freq_posenc = 10  # x的位置编码对应的超参数 Lx=10
        self.n_freq_posenc_views = 4  # d的位置编码对应的超参数 Ld=4

        dim_embed = 3 * self.n_freq_posenc * 2  # x编码后的维度  60
        dim_embed_view = 3 * self.n_freq_posenc_views * 2  # d编码后的维度  24

        self.x_in = nn.Linear(dim_embed, W)  # 输入层的第一层网络
        self.style_in = nn.Linear(style_dim, W)  # 潜在编码输入的网络

        self.pts_linears = nn.ModuleList([nn.Linear(W, W) for i in range(D-1)])

        self.views_linears = nn.Linear(dim_embed_view + W, W)
        self.rgb_linear = nn.Linear(W, 3)
        self.sigma_linear = nn.Linear(W, 1)

    def transform_points(self, p, views=False):
        # view 表示是否编码方向d,默认编码位置x
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / 2

        # we consider points up to [-1, 1]
        # so no scaling required here
        L = self.n_freq_posenc_views if views else self.n_freq_posenc
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p),
             torch.cos((2 ** i) * pi * p)],
            dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, x, styles):
        # 拆分x，d [1, 64, 64, 24, 3]
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        input_pts = self.transform_points(input_pts)
        input_views = self.transform_points(input_views, True)

        mlp_out = input_pts.contiguous()  # 拷贝x

        mlp_out = self.x_in(input_pts)
        style_in = self.style_in(styles)
        for i in range(3):
            style_in = style_in.unsqueeze(1)
        mlp_out = mlp_out + style_in
        mlp_out = F.relu(mlp_out)

        for i in range(len(self.pts_linears)):  # 7层MLP
            mlp_out = self.pts_linears[i](mlp_out)  # 需要输入x和z
            mlp_out = F.relu(mlp_out)

        # sdf层
        sdf = self.sigma_linear(mlp_out)  # [1, 64, 64, 24, 1]
        # 颜色层
        mlp_out = torch.cat([mlp_out, input_views], -1)  # 输入方向d
        out_features = self.views_linears(mlp_out)  # [1, 64, 64, 24, 280]
        rgb = self.rgb_linear(out_features)  # [1, 64, 64, 24, 3]

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)
        # outputs【rgb，sdf，out_features】 在最后一个维度拼接  [1, 64, 64, 24, 3+1+256]
        return outputs