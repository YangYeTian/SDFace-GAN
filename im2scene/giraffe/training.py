from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average)
from torchvision.utils import save_image, make_grid
import os
import torch
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
from torch.autograd import Variable
from torch.nn.functional import mse_loss

logger_py = logging.getLogger(__name__)

# 从随机分布中采样
# mu：均值  logvar：标准差
def reparameterize(mu, logvar):
    logvar = logvar.mul(0.5).exp_()
    eps = Variable(logvar.data.new(logvar.size()).normal_())
    return eps.mul(logvar).add_(mu)


class Trainer(BaseTrainer):
    ''' Trainer object for GIRAFFE.
        定义一个训练器，包括生成器、判别器及二者的优化器

    Args:
        model (nn.Module): GIRAFFE model
        optimizer_e (optimizer): encoder optimizer object
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, optimizer_e, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True, **kwargs):

        self.model = model
        self.optimizer_e = optimizer_e
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu

        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations

        self.vis_dict = model.generator.get_vis_dict(16)

        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.encoder = self.model.encoder
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, use_endocer=0, it=None):
        ''' Performs a training step.
        训练一个batch size

        Args:
            data (dict): data dictionary
            it (int): training iteration
            use_endocer: 是否使用编码器
        '''
        # 生成器训练
        # 随机生成图片 返回计算损失
        if use_endocer:
            # 使用编码器
            loss_e = self.train_step_encoder(data, it, use_endocer)

        loss_g = self.train_step_generator(data, it, use_endocer)

        # 判别器训练

        loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)

        if use_endocer:
            return {
                'encoder': loss_e,
                'generator': loss_g,
                'discriminator': loss_d,
                'regularizer': reg_d,
            }
        else:
            return {
                'generator': loss_g,
                'discriminator': loss_d,
                'regularizer': reg_d,
            }

    def eval_step(self):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''

        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen().cpu()[:, :3])
        x_fake = torch.cat(x_fake, dim=0)
        x_fake.clamp_(0., 1.)
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        eval_dict = {
            'fid_score': fid_score
        }

        return eval_dict

    def train_step_encoder(self, data, it=None, use_endocer=0):
        encoder = self.encoder
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(encoder, True)
        toggle_grad(generator, False)
        toggle_grad(discriminator, False)

        encoder.train()
        generator.train()
        discriminator.train()

        self.optimizer_e.zero_grad()

        # generator 是im2scene/giraffe/models/generator.py
        if self.multi_gpu:
            # 返回一个字典，包含生成器需要输入的各项参数
            latents = generator.module.get_vis_dict()
            x_fake = generator(**latents)  # **表示可以输入字典
        else:
            if use_endocer:
                # 使用编码器
                x_real = data.get('image').to(self.device)  # 读取真实图片
                mu, logvar = encoder(x_real)
                z = reparameterize(mu, logvar)
                x_fake = generator(z=z)
            else:
                # 未定义编码器，使用原方法
                x_fake = generator()

        # 随机生成图片 x_fake shape [32, 3, 64, 64]

        # 计算损失
        # 编码器计算mse和kl损失
        d_fake, d_fake_feat = discriminator(x_fake)
        # d_fake shape [32, 1]


        # 使用编码器，计算编码器损失
        kl = -0.5 * torch.sum(-logvar.exp() - torch.pow(mu, 2) + logvar + 1, 1)  # 计算kl散度
        d_real, d_real_feat = discriminator(x_real)  # 计算真实图片结果
        # mse = mse_loss(x_fake, x_real, reduction="sum")
        mse = torch.sum(0.5 * (d_real_feat - d_fake_feat) ** 2, 1)  # 判别器"高层特征"损失
        eloss = torch.sum(kl) + torch.sum(mse)

        eloss.backward()
        self.optimizer_e.step()
        return eloss

    def train_step_generator(self, data, it=None, use_endocer=0):
        # 生成器没有用到数据集的数据
        encoder = self.encoder
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(encoder, False)
        toggle_grad(generator, True)
        toggle_grad(discriminator, False)

        # 如果网络使用了Batch Normalization 或 dropout
        # 则需要在训练前添加modle.train,在测试前添加model.eval
        if encoder is not None:
            encoder.train()
        generator.train()
        discriminator.train()

        self.optimizer.zero_grad()

        # generator 是im2scene/giraffe/models/generator.py
        if self.multi_gpu:
            # 返回一个字典，包含生成器需要输入的各项参数
            latents = generator.module.get_vis_dict()
            x_fake = generator(**latents)  # **表示可以输入字典
        else:
            if use_endocer:
                # 使用编码器
                x_real = data.get('image').to(self.device)  # 读取真实图片
                # z = encoder(x_real)
                mu, logvar = encoder(x_real)
                z = reparameterize(mu, logvar)
                x_fake = generator(z=z)
            else:
                # 未定义编码器，使用原方法
                x_fake = generator()

        # 随机生成图片 x_fake shape [32, 3, 64, 64]

        # 计算损失
        # 编码器计算mse和kl损失
        d_fake, d_fake_feat = discriminator(x_fake)
        # d_fake shape [32, 1]
        gloss = compute_bce(d_fake, 1)  # 生成器计算bce损失

        gloss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gloss.item()

        # if use_endocer:
        #     # 使用编码器，计算编码器损失
        #     toggle_grad(encoder, True)
        #     toggle_grad(generator, False)
        #     toggle_grad(discriminator, False)
        #
        #     self.optimizer_e.zero_grad()
        #
        #     kl = -0.5 * torch.sum(-logvar.exp() - torch.pow(mu, 2) + logvar + 1, 1)  # 计算kl散度
        #     d_real, d_real_feat = discriminator(x_real)  # 计算真实图片结果
        #     # mse = mse_loss(x_fake, x_real, reduction="sum")
        #     mse = torch.sum(0.5 * (d_real_feat - d_fake_feat) ** 2, 1)  # 判别器"高层特征"损失
        #     eloss = torch.sum(kl) + torch.sum(mse)
        #
        #     eloss.backward()
        #     self.optimizer_e.step()
        #     return gloss.item(), eloss
        # else:
        #     return gloss.item()

    def train_step_discriminator(self, data, it=None, z=None, use_endocer=0):

        # generator 是 im2scene/giraffe/models/generator.py
        # discriminator 是 im2scene/discriminator/conv.py
        encoder = self.encoder
        generator = self.generator
        discriminator = self.discriminator
        toggle_grad(encoder, False)
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        if encoder is not None:
            encoder.train()
        generator.train()
        discriminator.train()

        self.optimizer_d.zero_grad()

        # 读取图片数据集
        x_real = data.get('image').to(self.device)
        loss_d_full = 0.

        x_real.requires_grad_()
        # x_real [32, 3, 64, 64]  图片
        d_real, d_real_feat = discriminator(x_real)
        # d_real [32, 1]  判别器输出

        d_loss_real = compute_bce(d_real, 1)  # BCE用于“是不是”问题 即输出概率
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        # 随机生成图片
        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_vis_dict()
                x_fake = generator(**latents)
            else:
                if use_endocer:
                    mu, logvar = encoder(x_real)
                    z = reparameterize(mu, logvar)
                    x_fake = generator(z=z)
                else:
                    x_fake = generator()

        x_fake.requires_grad_()
        d_fake, d_fake_feat = discriminator(x_fake)

        d_loss_fake = compute_bce(d_fake, 0)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (
            d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())

    def visualize(self, it=0):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        with torch.no_grad():
            image_fake = self.generator(**self.vis_dict, mode='val').cpu()

        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_%010d.png' % it

        image_grid = make_grid(image_fake.clamp_(0., 1.), nrow=4)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid
