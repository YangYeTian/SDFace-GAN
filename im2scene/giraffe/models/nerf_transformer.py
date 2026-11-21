from typing import Any, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


# # dense_layer = partial(nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
def dense_layer(in_, out_):
    return nn.Linear(in_features=in_, out_features=out_).cuda()


def layer_norm(input_shape):
    return nn.LayerNorm(input_shape).cuda()


droppath_prngkey = [0, 2022]  # jax.random.PRNGKey(2020)


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)], dtype=np.float32)
    # print(n_position,d_hid,sinusoid_table.shape)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    sinusoid_table = np.array(sinusoid_table, dtype=np.float32)
    return np.reshape(sinusoid_table, (1, sinusoid_table.shape[0], sinusoid_table.shape[1]))


class SinusoidPositionEmbs(nn.Module):
    def __init__(self, num_samples=192, embed_dim=256):
        super(SinusoidPositionEmbs, self).__init__()
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.setup()

    def setup(self):
        self.pos_embed = get_sinusoid_encoding_table(self.num_samples, self.embed_dim)

    def forward(self, inputs):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        return inputs + torch.Tensor(self.pos_embed).cuda()


def window_partition(x, window_size):
    print(x.shape)
    B, n, C = x.shape
    x = torch.reshape(x, (B, n // window_size, window_size, C))
    windows = torch.reshape(x, (-1, window_size, C))
    return windows


def window_reverse(windows, window_size, n):
    B = int(windows.shape[0] / (n / window_size))
    x = torch.reshape(windows, (B, n // window_size, window_size, -1))
    x = torch.reshape(x, (B, n, -1))
    return x


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    def forward(self, x):
        return x


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.
    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

    def forward(self, inputs):
        """Applies AddPositionEmbs module.
        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init.
        Args:
          inputs: Inputs to the layer.
        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
        return inputs + pe


'''使用了 PyTorch 中的 dense_layer、F.gelu 和 F.dropout 函数替代原jax实现中的 nn.Dense、nn.gelu 和 nn.Dropout'''


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, mlp_dim, dtype=np.float32, out_dim=None, dropout_rate=0.1):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

    def forward(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = dense_layer(inputs.shape[-1], self.mlp_dim)(inputs)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_rate, training=not deterministic)
        output = dense_layer(x.shape[-1], actual_out_dim)(x)
        output = F.dropout(output, p=self.dropout_rate, training=not deterministic)
        return output


class DropPath(nn.Module):
    """Create a dropout layer.

      Args:
        rate: the dropout probability.  (_not_ the keep rate!)
      """

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, inputs, deterministic=False, rng=None):
        if self.rate == 0. or deterministic:
            return inputs
        keep_prob = 1. - self.rate
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        if rng is None:
            rng = self.make_rng('params')
        random_tensor = keep_prob + np.random.uniform(rng, shape)
        random_tensor = np.floor(random_tensor)
        output = random_tensor * inputs / keep_prob
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.
    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    def __init__(self, mlp_dim, num_heads, dtype=np.float32, dropout_rate=0.1, attention_dropout_rate=0.1,
                 window_size=8, drop_path_rate=0., shift_size=0):

        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.window_size = window_size
        self.drop_path_rate = drop_path_rate
        self.shift_size = shift_size

    def forward(self, inputs, *, deterministic):
        """Applies Encoder1DBlock module.
        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.
        Returns:
          output after transformer encoder block.
        """
        b, n, c = inputs.shape

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = layer_norm(inputs.shape)(inputs)

        # cyclic shift
        if self.shift_size > 0:
            x = np.roll(x, -self.shift_size, axis=1)

        # partition windows
        x_windows = window_partition(x, self.window_size)
        # attn
        # print(x_windows.shape, x_windows.ndim, self.num_heads, self.dtype)
        x, attn_weights = nn.MultiheadAttention(
            embed_dim=x_windows.shape[-1],
            num_heads=self.num_heads,
            # dtype=self.dtype,
            dropout=self.attention_dropout_rate).cuda()(
            x_windows, x_windows, x_windows)
        x = nn.Dropout(self.dropout_rate)(x)
        # merge windows
        x = window_reverse(x, self.window_size, n)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = np.roll(x, self.shift_size, axis=1)

        if self.drop_path_rate > 0.:
            x = DropPath(rate=self.drop_path_rate)(x, deterministic=deterministic, rng=droppath_prngkey)
        x = x + inputs

        # MLP block.
        y = layer_norm(x.shape)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)
        if self.drop_path_rate > 0.:
            y = DropPath(rate=self.drop_path_rate)(y, deterministic=deterministic, rng=droppath_prngkey)

        return x + y


class LocalNerfTransformer(nn.Module):
    def __init__(self, embed_dim=256, depth=2, output_c=3, num_heads=8, mlp_ratio=1.,
                 drop_rate=0., drop_path_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, use_viewdirs=True, skips='0,1', window_size=0, shift_size=0):
        super(LocalNerfTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.output_c = output_c
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.use_viewdirs = use_viewdirs
        self.skips = skips
        self.window_size = window_size
        self.shift_size = shift_size

        # self.dense_layer1 = nn.Linear(3, self.embed_dim)
        self.norm_layer1 = self.norm_layer(self.embed_dim).cuda()
        self.dropout1 = nn.Dropout(self.drop_rate)

        self.encoder_blocks = nn.ModuleList()
        self.skips = list(map(int, self.skips.split(',')))
        for i in range(self.depth):
            self.encoder_blocks.append(
                Encoder1DBlock(
                    mlp_dim=int(self.embed_dim * self.mlp_ratio),
                    dropout_rate=self.drop_rate,
                    attention_dropout_rate=self.attn_drop_rate,
                    # dtype=f'encoderblock_{i}',
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    drop_path_rate=self.drop_path_rate,
                    shift_size=0 if i % 2 == 0 else self.shift_size
                )
            )

        # JAX中不需要明确指定输入特征的大小，而是通过Dense函数动态地推断输入的大小。而在PyTorch中则需要明确指定输入特征的大小
        self.dense_layer2 = dense_layer(self.embed_dim, self.embed_dim)
        self.dense_layer3 = dense_layer(self.embed_dim, self.embed_dim // 2)
        self.dense_layer4 = dense_layer(self.embed_dim // 2, 3)
        # self.dense_layer5 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.dense_layer6 = dense_layer(self.embed_dim, self.embed_dim // 2)
        self.dense_layer7 = dense_layer(self.embed_dim // 2, self.output_c)

    def forward_features(self, input_pts, train=True):
        input_pts = torch.Tensor(input_pts)
        x = dense_layer(input_pts.shape[-1], self.embed_dim)(input_pts)
        x = self.norm_layer1(x)
        x = SinusoidPositionEmbs(num_samples=x.shape[1], embed_dim=self.embed_dim)(x)
        x = self.dropout1(x)

        for i in range(self.depth):
            x = self.encoder_blocks[i](x, deterministic=not train)
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)
                x = dense_layer(x.shape[-1], self.embed_dim)(x)
                x = self.act_layer()(x)

        x = layer_norm(x.shape)(x)
        return x

    def forward(self, input_pts, input_views=None, train=True):

        h = self.forward_features(input_pts, train)

        if self.use_viewdirs:
            assert input_views is not None
            if len(input_views.shape) < 3:
                input_views = torch.repeat_interleave(torch.unsqueeze(input_views, 1), input_pts.shape[1], 1)
            alpha = dense_layer(h.shape[-1], 1)(h)
            feature = dense_layer(h.shape[-1], self.embed_dim)(h)
            # print(feature.shape, input_views.shape)
            h = torch.cat([feature, torch.Tensor(input_views)], -1)
            # print(h.shape)
            rgb = dense_layer(h.shape[-1], self.embed_dim // 2)(h)
            rgb = self.act_layer()(rgb)
            rgb = dense_layer(rgb.shape[-1], 3)(rgb)
            return rgb, alpha
        else:
            outputs = dense_layer(self.embed_dim // 2)
            outputs = self.act_layer(outputs)
            outputs = dense_layer(self.output_c)(outputs)

        return outputs


def get_nerf_transformer(name, **kwargs):
    if name == 'next_s':
        default_kwargs = {
            'embed_dim': 192,
            'depth': 2,
            'skips': '0,1',
            'window_size': 64
        }
    elif name == 'next_b':
        default_kwargs = {
            'embed_dim': 256,
            'depth': 2,
            'skips': '0,1',
            'window_size': 64
        }
    elif name == 'next_l':
        default_kwargs = {
            'embed_dim': 256,
            'depth': 4,
            'skips': '0,1,2,3',
            'window_size': 64
        }
    else:
        raise NotImplementedError(name)
    kwargs.update(default_kwargs)
    return LocalNerfTransformer(**kwargs)
