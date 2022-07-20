import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import biggan_layers as layers


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {
        'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
        'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
        'upsample': [True] * 7,
        'resolution': [8, 16, 32, 64, 128, 256, 512],
        'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                      for i in range(3, 10)},
    }
    arch[256] = {
        'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
        'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
        'upsample': [True] * 6,
        'resolution': [8, 16, 32, 64, 128, 256],
        'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                      for i in range(3, 9)},
    }
    arch[128] = {
        'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
        'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
        'upsample': [True] * 5,
        'resolution': [8, 16, 32, 64, 128],
        'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                      for i in range(3, 8)},
    }
    arch[64] = {
        'in_channels': [ch * item for item in [16, 16, 8, 4]],
        'out_channels': [ch * item for item in [16, 8, 4, 2]],
        'upsample': [True] * 4,
        'resolution': [8, 16, 32, 64],
        'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                      for i in range(3, 7)},
    }
    arch[32] = {
        'in_channels': [ch * item for item in [4, 4, 4]],
        'out_channels': [ch * item for item in [4, 4, 4]],
        'upsample': [True] * 3,
        'resolution': [8, 16, 32],
        'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                      for i in range(3, 6)},
    }

    return arch


class Generator(nn.Module):

    def __init__(self,
                 G_ch=96,
                 dim_z=120,
                 bottom_width=4,
                 resolution=256,
                 G_kernel_size=3,
                 G_attn='64',
                 n_classes=1000,
                 num_G_SVs=1,
                 num_G_SV_itrs=1,
                 G_shared=True,
                 shared_dim=128,
                 hier=True,
                 cross_replica=False,
                 mybn=False,
                 G_activation=nn.ReLU(inplace=True),
                 BN_eps=1e-5,
                 SN_eps=1e-12,
                 G_param='SN',
                 norm_style='bn',
                 **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = self.dim_z // self.num_slots
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(
                layers.SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=num_G_SVs,
                num_itrs=num_G_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_linear = functools.partial(
                layers.SNLinear,
                num_svs=num_G_SVs,
                num_itrs=num_G_SV_itrs,
                eps=self.SN_eps,
            )
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared else self.which_embedding)
        self.which_bn = functools.partial(
            layers.ccbn,
            which_linear=bn_linear,
            cross_replica=self.cross_replica,
            mybn=self.mybn,
            input_size=(self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes),
            norm_style=self.norm_style,
            eps=self.BN_eps,
        )

        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared else layers.identity())

        # First linear layer
        self.linear = self.which_linear(
            self.dim_z // self.num_slots,
            self.arch['in_channels'][0] * (self.bottom_width**2),
        )

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                layers.GBlock(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    which_conv=self.which_conv,
                    which_bn=self.which_bn,
                    activation=self.activation,
                    upsample=(functools.partial(F.interpolate, scale_factor=2)
                              if self.arch['upsample'][index] else None),
                )
            ]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(
            layers.bn(
                self.arch['out_channels'][-1],
                cross_replica=self.cross_replica,
                mybn=self.mybn,
            ),
            self.activation,
            self.which_conv(self.arch['out_channels'][-1], 3),
        )

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y, ret_resolution=(64, 128, 256)):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        h_ret = {}

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])
            if h.size(-1) in ret_resolution:
                h_ret[h.size(-1)] = h

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h)), h_ret


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
        'downsample': [True] * 6 + [False],
        'resolution': [128, 64, 32, 16, 8, 4, 4],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 8)},
    }
    arch[128] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
        'downsample': [True] * 5 + [False],
        'resolution': [64, 32, 16, 8, 4, 4],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 8)},
    }
    arch[64] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
        'downsample': [True] * 4 + [False],
        'resolution': [32, 16, 8, 4, 4],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 7)},
    }
    arch[32] = {
        'in_channels': [3] + [item * ch for item in [4, 4, 4]],
        'out_channels': [item * ch for item in [4, 4, 4, 4]],
        'downsample': [True, True, False, False],
        'resolution': [16, 16, 16, 16],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 6)},
    }
    return arch


class Discriminator(nn.Module):

    def __init__(self,
                 D_ch=96,
                 D_wide=True,
                 resolution=256,
                 D_kernel_size=3,
                 D_attn='64',
                 n_classes=1000,
                 num_D_SVs=1,
                 num_D_SV_itrs=1,
                 D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12,
                 output_dim=1,
                 D_param='SN',
                 update_embed=False,
                 **kwargs):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        self.update_embed = update_embed
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(
                layers.SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_linear = functools.partial(
                layers.SNLinear,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_embedding = functools.partial(
                layers.SNEmbedding,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=self.SN_eps,
            )
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            self.which_embedding = nn.Embedding

        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                layers.DBlock(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    which_conv=self.which_conv,
                    wide=self.D_wide,
                    activation=self.activation,
                    preactivation=(index > 0),
                    downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None),
                )
            ]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        for param in self.embed.parameters():
            param.requires_grad = update_embed

    def forward(self, x, max_inference_depth=3):
        """
        :param max_inference_depth: {2:[64,],3:[64,32],4:[64,32,16]}
        :return: list of dis features
        """
        # Stick x into h for cleaner for loops without flow control
        if x.size(1) == 1:
            x = x.expand(x.size(0), 3, x.size(2), x.size(3))
        h = x
        h_list = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
            if index == 0:
                continue
            h_list.append(h)

            if index >= max_inference_depth - 1:
                break

        return h_list


# Encoder architecture, same paradigm as D's above
def E_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {
        'in_channels': [1] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
        'downsample': [True] * 6 + [False],
        'resolution': [128, 64, 32, 16, 8, 4, 4],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 8)},
    }
    arch[128] = {
        'in_channels': [1] + [ch * item for item in [1, 2, 4, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
        'downsample': [True] * 5 + [False],
        'resolution': [64, 32, 16, 8, 4, 4],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 8)},
    }
    arch[64] = {
        'in_channels': [1] + [ch * item for item in [1, 2, 4, 8]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
        'downsample': [True] * 4 + [False],
        'resolution': [32, 16, 8, 4, 4],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 7)},
    }
    arch[32] = {
        'in_channels': [1] + [item * ch for item in [4, 4, 4]],
        'out_channels': [item * ch for item in [4, 4, 4, 4]],
        'downsample': [True, True, False, False],
        'resolution': [16, 16, 16, 16],
        'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                      for i in range(2, 6)},
    }
    return arch


class Encoder(nn.Module):

    def __init__(self,
                 E_ch=64,
                 E_wide=True,
                 resolution=256,
                 E_kernel_size=3,
                 E_attn='64',
                 n_classes=1000,
                 E_activation=nn.ReLU(inplace=False),
                 output_dim=120,
                 shared_dim=128,
                 **kwargs):
        super(Encoder, self).__init__()
        # Width multiplier
        self.ch = E_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = E_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = E_kernel_size
        # Attention?
        self.attention = E_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = E_activation
        # Architecture
        self.arch = E_arch(self.ch, self.attention)[resolution]

        self.shared_dim = shared_dim

        # Which convs, batchnorms, and linear layers to use
        self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
        self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding
        bn_linear = functools.partial(self.which_linear, bias=False)
        self.which_bn = functools.partial(
            layers.ccbn,
            which_linear=bn_linear,
            input_size=self.shared_dim,
            norm_style='bn',
        )
        self.shared = self.which_embedding(n_classes, self.shared_dim)

        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                layers.GBlock(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    which_conv=self.which_conv,
                    activation=self.activation,
                    upsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None),
                    which_bn=self.which_bn,
                )
            ]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in E at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

    def forward(self, x, y):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, y)
        # Apply global sum pooling as in SN-GAN
        h = torch.mean(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)

        return out
