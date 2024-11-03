
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy import pi


class Decoder(nn.Module):
    ''' Decoder class.
    根据3D位置、观看方向和潜码z预测体积密度和颜色。
    MLP网络层,包括对象和背景两类MLP网络
    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''

    def __init__(self,
                 hidden_size=128,
                 n_blocks=8,
                 n_blocks_view=1,
                 skips=[4],
                 use_viewdirs=True,
                 n_freq_posenc=10,
                 n_freq_posenc_views=4,
                 z_dim=64,
                 rgb_out_dim=128,
                 final_sigmoid_activation=False,
                 downscale_p_by=2.,
                 positional_encoding="normal",
                 gauss_dim_pos=10,
                 gauss_dim_view=4,
                 gauss_std=4.,
                 embed_fn=None,  # x编码器
                 embeddirs_fn=None,  # d编码器
                 dim_embed=32,  # x编码后的维度  32
                 dim_embed_view=16,  # d编码后的维度  16
                 **kwargs):

        super().__init__()
        self.use_viewdirs = use_viewdirs  # 是否使用方向D
        self.n_freq_posenc = n_freq_posenc  # x的位置编码对应的超参数 Lx=10
        self.n_freq_posenc_views = n_freq_posenc_views  # d的位置编码对应的超参数 Ld=4
        self.skips = skips  # NeRF中的跳跃连接 list
        self.downscale_p_by = downscale_p_by  # 位置编码前的超参数,这里2表示取1/2
        self.z_dim = z_dim  # 潜在编码z维度
        self.final_sigmoid_activation = final_sigmoid_activation  # 是否最最终结果进行sigmoid激活
        self.n_blocks = n_blocks  # NeRF网络层数  8
        self.n_blocks_view = n_blocks_view  # view dep 层数  1
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding  # 位置编码的模式
        # 计算位置编码后的数据维度
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        elif embed_fn != None:
            # 使用hash编码
            self.dim_embed = dim_embed
            self.dim_embed_view = dim_embed_view
        else:  # 正常位置编码
            dim_embed = 3 * self.n_freq_posenc * 2  # x编码后的维度  60
            dim_embed_view = 3 * self.n_freq_posenc_views * 2  # d编码后的维度  24

        # Density Prediction Layers  体积密度预测层  类似NeRF
        # 第一层网络
        self.fc_in = nn.Linear(dim_embed, hidden_size)  # 输入层的第一层网络
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)  # 潜在编码输入的网络

        # 剩下7层网络
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)
        ])
        # sum函数计算列表内一共有几个元素
        n_skips = sum([i in skips for i in range(n_blocks - 1)])  # 统计有几个跳跃连接  1
        if n_skips > 0:  # 获得两个具有跳跃连接的网络
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(dim_embed, hidden_size) for i in range(n_skips)
            ])
        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [nn.Linear(dim_embed_view + hidden_size, hidden_size)
                 for i in range(n_blocks_view - 1)])

    def transform_points(self, p, views=False):
        # view 表示是否编码方向d,默认编码位置x
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed


    def hash_encoding(self, p, views=False):
        # hash编码，需要对向量进行归一化
        if views:
            """
            对d进行编码
            """
            # d [32, 16384, 3]
            # d不需要进行归一化
            a = p.shape[0]
            b = p.shape[1]                              # 保存原先形状
            p = torch.reshape(p, [-1, p.shape[-1]])     # 对向量形状进行改变  x [524288, 3]
                                                        # SH球面调和编码需要输入 [ab, 3]的数据
            p = self.embeddirs_fn(p)                    # [524288, 16] 编码成功，需要再变回[32, 16384, 16]
            p = torch.reshape(p, [a, b, self.dim_embed_view])

        else:
            """
            对x进行编码
            """

            p = torch.div(p, 15)                        # 对向量进行归一化  x [32, 16384, 3]
            a = p.shape[0]
            b = p.shape[1]                              # 保存原先形状
            p = torch.reshape(p, [-1, p.shape[-1]])     # 对向量形状进行改变  x [524288, 3]
                                                        # hash编码需要输入 [ab, 3]的数据
            p = self.embed_fn(p)                        # [524288, 32] 编码成功，需要再变回[32, 16384, 32]
            p = torch.reshape(p, [a, b, self.dim_embed])       # 最终需要返回[a, b, 32]的数据
        return p

    def get_sigma(self, p_in, z_shape):
        a = F.relu
        # 获取潜在编码zs和za
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(p_in.device)
        if self.embed_fn != None:
            p = self.hash_encoding(p_in)
        else:
            # 对x进行位置编码
            p = self.transform_points(p_in)

        # 八层MLP
        # 第一层
        net = self.fc_in(p)
        # [32, 16384, 128]
        if z_shape is not None:
            zs = self.fc_z(z_shape).unsqueeze(1)
            # [32, 1, 128]
            net = net + zs  # 将x和zs进行矩阵相加
            # [32, 16384, 128]
        net = a(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = a(layer(net))
            # 跳跃连接处采用矩阵加法将数据相加
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape).unsqueeze(1)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1

        sigma_out = self.sigma_out(net).squeeze(-1)
        return sigma_out


    # 网络前向传播
    def forward(self, p_in, ray_d, z_shape=None, z_app=None, **kwargs):
        a = F.relu
        # 获取潜在编码zs和za
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(p_in.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(p_in.device)
        if self.embed_fn != None:
            p = self.hash_encoding(p_in)
        else:
            # 对x进行位置编码
            p = self.transform_points(p_in)

        # 八层MLP
        # 第一层
        net = self.fc_in(p)
        # [32, 16384, 128]
        if z_shape is not None:
            zs =self.fc_z(z_shape).unsqueeze(1)
            # [32, 1, 128]
            net = net + zs  # 将x和zs进行矩阵相加
            # [32, 16384, 128]
        net = a(net)

        # 剩下7层MLP，包含中间跳跃连接
        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = a(layer(net))
            # 跳跃连接处采用矩阵加法将数据相加
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape).unsqueeze(1)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        # 运行完8层MLP，得到一个128维向量

        # 输出向量通过一个网络层，获取一个一维向量σ
        # tensor.squeeze(-1)表示将一个n*1的向量变成一个n维向量
        # 若向量不是这种形状，则无影响
        sigma_out = self.sigma_out(net).squeeze(-1)

        net = self.feat_view(net)
        # tensor.unsqueeze(1)与上述方法想法，为最后一个维度添加一个维度
        # 如[2]的向量变成[2,1]的向量
        net = net + self.fc_z_view(z_app).unsqueeze(1)
        # 如果使用视角方向，默认为使用
        if self.use_viewdirs and ray_d is not None:
            # norm表示求L2范数，即将方向向量进行归一化
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            if self.embeddirs_fn != None:
                ray_d = self.hash_encoding(ray_d, views=True)  # hash编码
            else:
                ray_d = self.transform_points(ray_d, views=True)  # 位置编码
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        # print(feat_out.shape, "feat_out")  [32, 16384, 128]
        # print(sigma_out.shape, "sigma_out")  [32, 16384]

        return feat_out, sigma_out


class SmallDecoder(nn.Module):
    ''' SmallDecoder class.
    重新设计的MLP网络层
    网络结构更小，使用哈希编码来代替位置编码
    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''

    def __init__(self,
                 hidden_size=64,        # 隐藏层层数
                 n_blocks=3,            # 体积密度网络层数
                 n_blocks_view=4,       # 特征网络层数
                 use_viewdirs=True,     # 是否使用方向d
                 z_dim=64,              # 潜在编码z的维度
                 geo_feat_dim=15,       # 体特征向量维度
                 rgb_out_dim=128,        # 输出特征向量的维度
                 final_sigmoid_activation=False,    # 最终使用sigmoid激活
                 embed_fn=None,  # x编码器
                 embeddirs_fn=None,  # d编码器
                 dim_embed=32,  # x编码后的维度  32
                 dim_embed_view=16,  # d编码后的维度  16
                 **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs  # 是否使用方向D
        self.z_dim = z_dim  # 潜在编码z维度
        self.final_sigmoid_activation = final_sigmoid_activation  # 是否最最终结果进行sigmoid激活
        self.n_blocks = n_blocks  # NeRF网络层数  3
        self.n_blocks_view = n_blocks_view  # view dep 层数  1
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.dim_embed = dim_embed
        self.dim_embed_view = dim_embed_view


        """
        ======
        需要修改
        ======
        """
        # 输入输出层维度
        # dim_embed = 32  # x编码后的维度  32
        # dim_embed_view = 16  # d编码后的维度  16
        dim_geo_feat = geo_feat_dim + 1

        # 体积密度网络
        # 网络输入
        self.fc_in = nn.Linear(dim_embed, hidden_size)  # x的输入网络
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)  # 潜在编码z的输入的网络

        self.blocks = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 2)  # 去掉输入输出
        ])
        # 网络输出
        self.fc_out = nn.Linear(hidden_size, dim_geo_feat)


        # 特征网络
        # 网络输入
        self.fc_z_view = nn.Linear(z_dim, hidden_size)             # zs
        self.fc_feat_view = nn.Linear(geo_feat_dim, hidden_size)   # f
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)      # d

        if use_viewdirs and n_blocks_view > 1:
            # 如果使用方向d
            self.blocks_view = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(n_blocks_view - 2)])  # 去掉输入输出
        # 网络输出
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)


    def hash_encoding(self, p, views=False):
        # hash编码，需要对向量进行归一化
        if views:
            """
            对d进行编码
            """
            # d [32, 16384, 3]
            # d不需要进行归一化
            a = p.shape[0]
            b = p.shape[1]                              # 保存原先形状
            p = torch.reshape(p, [-1, p.shape[-1]])     # 对向量形状进行改变  x [524288, 3]
                                                        # SH球面调和编码需要输入 [ab, 3]的数据
            p = self.embeddirs_fn(p)                    # [524288, 16] 编码成功，需要再变回[32, 16384, 16]
            p = torch.reshape(p, [a, b, self.dim_embed_view])

        else:
            """
            对x进行编码
            """

            p = torch.div(p, 15)                        # 对向量进行归一化  x [32, 16384, 3]
            a = p.shape[0]
            b = p.shape[1]                              # 保存原先形状
            p = torch.reshape(p, [-1, p.shape[-1]])     # 对向量形状进行改变  x [524288, 3]
                                                        # hash编码需要输入 [ab, 3]的数据
            p = self.embed_fn(p)                        # [524288, 32] 编码成功，需要再变回[32, 16384, 32]
            p = torch.reshape(p, [a, b, self.dim_embed])       # 最终需要返回[a, b, 32]的数据
        return p


    def forward(self, p_in, ray_d, z_shape=None, z_app=None, **kwargs):
        # 前向传播
        # 获取潜在编码zs和za
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(p_in.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(p_in.device)

        # 对x进行位置编码
        p = self.hash_encoding(p_in)
        # [32, 16384, 32]


        # 体积密度网络
        h = self.fc_in(p)
        if z_shape is not None:
            zs = self.fc_z(z_shape).unsqueeze(1)
            # zs [32, 1, 64]
            # h [32, 16384, 16]
            h = h + zs  # 将x和zs进行矩阵相加
        h = F.relu(h)

        # 经过体积密度MLP网络
        for layer in self.blocks:
            h = F.relu(layer(h))
        h = self.fc_out(h)  # h [32, 16384, 16]

        # 划分sigma和特征向量
        sigma_out, geo_feat = h[..., 0], h[..., 1:]
        # sigma_out [32, 16384]
        # geo_feat [32, 16384, 15]


        # 颜色网络
        h = self.fc_feat_view(geo_feat)
        # tensor.unsqueeze(1)与上述方法想法，为最后一个维度添加一个维度
        # 如[2]的向量变成[2,1]的向量
        h = h + self.fc_z_view(z_app).unsqueeze(1)

        # 如果使用视角方向，默认为使用
        if self.use_viewdirs and ray_d is not None:
            # norm表示求L2范数，即将方向向量进行归一化
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.hash_encoding(ray_d, views=True)  # 位置编码
            h = h + self.fc_view(ray_d)

        h = F.relu(h)

        for layer in self.blocks_view:
            h = F.relu(layer(h))
        feat_out = self.feat_out(h)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return feat_out, sigma_out

