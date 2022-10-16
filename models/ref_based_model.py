import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.dcn import DCNv2Pack
from utils import util

# import time

# -----------------------------------------------------------------------------
# utils functions
# -----------------------------------------------------------------------------


def feature_normalize(feature_in):
    feature_in_norm = (torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon)
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


def vgg_preprocess(tensor, vgg_normal_correct=False):
    if tensor.size(1) == 1:
        tensor = tensor.expand(tensor.size(0), 3, tensor.size(2), tensor.size(3))
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat(
        (tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]),
        dim=1,
    )
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst


def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


# -----------------------------------------------------------------------------
# customized loss
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# sub-modules
# -----------------------------------------------------------------------------


class SPADE(nn.Module):

    def __init__(self, norm_nc, spade_nc):
        super().__init__()
        self.param_free_norm = PositionalNorm2d
        self.conv_cond = nn.Sequential(
            nn.Conv2d(norm_nc + spade_nc, (norm_nc + spade_nc) // 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d((norm_nc + spade_nc) // 8, spade_nc, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dcn = DCNv2Pack(in_channels=spade_nc, out_channels=spade_nc, kernel_size=3, padding=1, stride=1)
        self.mlp_gamma = nn.Conv2d(spade_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(spade_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, ref):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on ref image
        cond = self.conv_cond(torch.cat((normalized, ref), dim=1))
        ref_refine = self.dcn(ref, cond)
        gamma = self.mlp_gamma(ref_refine)
        beta = self.mlp_beta(ref_refine)
        out = normalized * (1 + gamma) + beta
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            stride=stride,
        )
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            stride=stride,
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.prelu(out)
        return out


class SPADEResnetBlock(nn.Module):

    def __init__(self, fin, fout, spade_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.pad = nn.ReflectionPad2d(1)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = SPADE(fin, spade_nc)
        self.norm_1 = SPADE(fmiddle, spade_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, spade_nc)

    # note the resnet block with SPADE also takes in ref,
    # the warped ref image as input
    def forward(self, x, ref):
        x_s = self.shortcut(x, ref)
        dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, ref))))
        dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, ref))))

        out = x_s + dx

        return out

    def shortcut(self, x, ref):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, ref))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class VGG19_feature_color_torchversion(nn.Module):
    """
    NOTE: there is no need to pre-process the input
    input tensor should range in [0,1]
    """

    def __init__(self, pool='max', ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(
            self,
            x,
            out_keys=('r34', ),
            preprocess=True,
            vgg_normal_correct=True,
            max_inference_depth=5,
    ):
        """
        NOTE: input tensor should range in [0,1]
        """
        depth = 1
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=vgg_normal_correct)

        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        depth += 1
        if depth > max_inference_depth:
            return [out[key] for key in out_keys]
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        depth += 1
        if depth > max_inference_depth:
            return [out[key] for key in out_keys]
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        depth += 1
        if depth > max_inference_depth:
            return [out[key] for key in out_keys]
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        depth += 1
        if depth > max_inference_depth:
            return [out[key] for key in out_keys]
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class CorrespondenceNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # res_nc = 512
        self.private_layer = nn.ModuleList()
        for _ in range(2):
            self.private_layer.append(ResidualBlock(cfg.FEAT_NC, cfg.FEAT_NC))
        self.share_layer = nn.Sequential(
            ResidualBlock(cfg.FEAT_NC + 3, cfg.FEAT_NC + 3),
            ResidualBlock(cfg.FEAT_NC + 3, cfg.FEAT_NC + 3),
        )
        self.phi = nn.Conv2d(
            in_channels=cfg.FEAT_NC + 3,
            out_channels=cfg.NONLOCAL_NC,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.theta = nn.Conv2d(
            in_channels=cfg.FEAT_NC + 3,
            out_channels=cfg.NONLOCAL_NC,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        if cfg.WARP_FEAT:
            self.val = nn.Conv2d(
                in_channels=cfg.FEAT_NC + 3,
                out_channels=cfg.NONLOCAL_NC,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        self.cfg = cfg

    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = (torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1))
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = (torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1))
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))

        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat

    def forward(self, q, q_ref, ref, featnet, ref_biggan_feat):

        corr_out = {}

        with torch.no_grad():
            q_feat = featnet(
                q,
                max_inference_depth=self.cfg.MAX_INFERENCE_DEPTH,
            )
            ref_feat = featnet(
                ref,
                max_inference_depth=self.cfg.MAX_INFERENCE_DEPTH,
            )
            # q_ref_feat = featnet(
            #     q_ref,
            #     max_inference_depth=self.cfg.MAX_INFERENCE_DEPTH,
            # )

            if isinstance(q_feat, list):
                q_feat = torch.cat(
                    [F.upsample(q_feat[i], scale_factor=2**i) for i in range(len(q_feat))],
                    dim=1,
                )
                ref_feat = torch.cat(
                    [F.upsample(ref_feat[i], scale_factor=2**i) for i in range(len(ref_feat))],
                    dim=1,
                )
                # q_ref_feat = torch.cat(
                #     [
                #         F.upsample(q_ref_feat[i], scale_factor=2 ** i)
                #         for i in range(len(q_ref_feat))
                #     ],
                #     dim=1,
                # )

        # better to have private layers for pre-processing
        q_feat = self.private_layer[0](q_feat)
        ref_feat = self.private_layer[1](ref_feat)
        # q_ref_feat = self.private_layer[1](q_ref_feat)
        # since query and ref are usually not in the same domain, we better normalize them
        q_feat = feature_normalize(q_feat)
        ref_feat = feature_normalize(ref_feat)
        # q_ref_feat = feature_normalize(q_ref_feat)
        # domain alignment loss
        # corr_out["domain_align_loss"] = F.l1_loss(q_ref_feat, q_feat)
        corr_out['domain_align_loss'] = 0
        # add spatial coords
        q_feat = self.addcoords(q_feat)
        ref_feat = self.addcoords(ref_feat)
        # several residual layers for post-processing
        q_feat = self.share_layer(q_feat)
        ref_feat = self.share_layer(ref_feat)

        n, _, h_f, w_f = q_feat.size()

        # pairwise cosine similarity
        theta = self.theta(q_feat)
        theta = theta.view(n, self.cfg.NONLOCAL_NC, -1)
        theta = theta - theta.mean(dim=1, keepdim=True)  # center the feature
        theta_norm = (torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon)
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # n*(feature_height*feature_width)*nonlocal_nc

        phi = self.phi(ref_feat)
        phi = phi.view(n, self.cfg.NONLOCAL_NC, -1)
        phi = phi - phi.mean(dim=1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)

        corr_dot = torch.matmul(theta_permute, phi)  # n*(feature_height*feature_width)*(feature_height*feature_width)
        corr = F.softmax(corr_dot / self.cfg.TEMP, dim=-1)  # 低温可以让softmax曲线更尖

        img_to_warp = F.unfold(ref, 4, stride=4)
        img_to_warp = img_to_warp.permute(0, 2, 1)
        warp_img = torch.matmul(corr, img_to_warp)
        warp_img = warp_img.permute(0, 2, 1)
        corr_out['warp_image'] = F.fold(warp_img, 256, 4, stride=4)

        corr_out['warp_feat'] = {}
        for k in ref_biggan_feat.keys():
            s = k // 64
            ftr_to_warp = F.unfold(ref_biggan_feat[k], s, stride=s)
            ftr_to_warp = ftr_to_warp.permute(0, 2, 1)
            warp_feat_f = torch.matmul(corr, ftr_to_warp)
            warp_feat = warp_feat_f.permute(0, 2, 1)
            warp_feat = F.fold(warp_feat, s * 64, s, stride=s)
            corr_out['warp_feat'][k] = warp_feat

        return corr_out


class Generator(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        ngf = 64
        n_downsampling = cfg.N_DOWN

        # down
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(cfg.IN_NC, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        self.encoder = nn.Sequential(*encoder)

        # res modules
        self.res_blks = nn.ModuleList()
        mult = 2**n_downsampling
        for i in range(cfg.N_RES):
            self.res_blks.append(SPADEResnetBlock(ngf * mult, ngf * mult, 4 * 96))

        # up
        self.up_conv1 = nn.ConvTranspose2d(
            ngf * 4,
            ngf * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up_conv2 = nn.ConvTranspose2d(
            ngf * 2,
            ngf * 1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up_norm1 = SPADE(ngf * 4, 4 * 96)
        self.up_norm2 = SPADE(ngf * 2, 2 * 96)
        self.up_norm3 = SPADE(ngf, 96)
        self.head = nn.Sequential(nn.Conv2d(ngf, cfg.OUT_NC, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, x, ref):
        x = self.encoder(x)
        for mod in self.res_blks:
            x = mod(x, ref[64])
        x = self.up_conv1(F.leaky_relu(self.up_norm1(x, ref[64]), 0.2))
        x = self.up_conv2(F.leaky_relu(self.up_norm2(x, ref[128]), 0.2))
        x = self.head(F.leaky_relu(self.up_norm3(x, ref[256]), 0.2))
        return x


class RefBasedModel(nn.Module):

    def __init__(self, net_corr, net_gen, net_dis, vgg, biggan_dis, biggan_enc, biggan_gen, cfg):
        super().__init__()
        self.net_corr = net_corr
        self.net_gen = net_gen
        self.net_dis = net_dis
        self.vgg = vgg
        self.biggan_dis = biggan_dis
        self.biggan_enc = biggan_enc
        self.biggan_gen = biggan_gen
        self.cfg = cfg
        # self.tot_time = 0.0
        # self.cnt = 0

    def generate_fake(self, data):
        out = {}
        # torch.cuda.synchronize()
        # start_time = time.time()
        with torch.no_grad():
            # gray = (
            #     0.2989 * data["x_rgb"][:, 0:1, :, :]
            #     + 0.5870 * data["x_rgb"][:, 1:2, :, :]
            #     + 0.1140 * data["x_rgb"][:, 2:3, :, :]
            # )
            # z = self.biggan_enc(gray, self.biggan_enc.shared(data["cid"]))
            z = self.biggan_enc(data['x_l'], self.biggan_enc.shared(data['cid']))
            if 'shift' in data:
                z = z + data['shift']
            ref, ref_feat_dict = self.biggan_gen(z, self.biggan_gen.shared(data['cid']))
        coor_out = self.net_corr(data['x_l'], data['x_rgb'], ref, self.biggan_dis, ref_feat_dict)
        out['fake_ab'] = self.net_gen(data['x_l'], coor_out['warp_feat'])

        out['fake_rgb'] = util.lab_to_rgb(torch.cat((data['x_l'] * 50.0 + 50.0, out['fake_ab'] * 110.0), dim=1))
        if self.cfg.DATA.FULL_RES_OUTPUT:
            out['fake_rgb_full_res'] = util.lab_to_rgb(
                torch.cat((data['x_l_full_res'] * 50.0 + 50.0,
                           F.interpolate(out['fake_ab'], size=data['x_l_full_res'].size()[2:], mode='bicubic') * 110.0),
                          dim=1))
        # torch.cuda.synchronize()
        # model_time = time.time() - start_time
        # print(f'tot time: {self.tot_time}')
        # self.tot_time += model_time
        # print(f'tot time: {self.tot_time}')
        # self.cnt += 1
        # avg_time = self.tot_time / self.cnt
        # print(f'current time: {model_time}, avg time: {avg_time}')

        out['ref'] = ref

        return {**out, **coor_out}

    def forward(self, func, **kwargs):
        return getattr(self, func)(**kwargs)
