import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import data
import models
import utils.my_logging as logging
from utils.util import lab_to_rgb
from .base_solver import BaseSolver

logger = logging.get_logger(__name__)


class REFCOLORSolver(BaseSolver):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prepare_for_testing()

    def load_biggan(self):
        self.biggan_dis = models.biggan.Discriminator()
        self.biggan_enc = models.biggan.Encoder()
        self.biggan_gen = models.biggan.Generator()

        self.biggan_dis.load_state_dict(torch.load(self.cfg.BIGGAN_PRETRAIN.DIS))
        self.biggan_enc.load_state_dict(torch.load(self.cfg.BIGGAN_PRETRAIN.ENC))
        self.biggan_gen.load_state_dict(torch.load(self.cfg.BIGGAN_PRETRAIN.GEN))

        self.biggan_enc = self.biggan_enc.to(self.device)
        self.biggan_gen = self.biggan_gen.to(self.device)
        self.biggan_dis = self.biggan_dis.to(self.device)

        self.biggan_dis.eval()
        self.biggan_enc.eval()
        self.biggan_gen.eval()

        for param in self.biggan_dis.parameters():
            param.requires_grad = False
        for param in self.biggan_gen.parameters():
            param.requires_grad = False
        for param in self.biggan_enc.parameters():
            param.requires_grad = False

    def prepare_for_testing(self):
        self.net_corr = models.ref_based_model.CorrespondenceNet(self.cfg.REFCOLOR.CORR, )
        self.net_gen = models.ref_based_model.Generator(self.cfg.REFCOLOR.GEN, )
        self.to_device()
        self.load_biggan()
        self.model = models.ref_based_model.RefBasedModel(self.net_corr, self.net_gen, None, None, self.biggan_dis,
                                                          self.biggan_enc, self.biggan_gen, self.cfg)
        # print('net_corr: ', sum(map(lambda x: x.numel(), self.net_corr.parameters())))
        # print('net_gen: ', sum(map(lambda x: x.numel(), self.net_gen.parameters())))
        # print('biggan_dis: ', sum(map(lambda x: x.numel(), self.biggan_dis.parameters())))
        # print('biggan_enc: ', sum(map(lambda x: x.numel(), self.biggan_enc.parameters())))
        # print('biggan_gen: ', sum(map(lambda x: x.numel(), self.biggan_gen.parameters())))

    def test(self):
        self.load_from_ckp()

        self.test_dl = data.get_loader(cfg=self.cfg, ds=self.cfg.DATA.NAME)
        torch.manual_seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)
        data_iter = iter(self.test_dl)
        bs_idx = 0
        while True:
            try:
                self.read_data_from_dataiter(data_iter)
            except StopIteration:
                break
            self.validate('test_bs{}'.format(bs_idx))
            bs_idx += 1

    @torch.no_grad()
    def validate(self, info):
        self.net_gen.eval()
        self.net_corr.eval()

        out = self.model('generate_fake', data=self.sample_data)

        x_gray_lab = torch.cat(
            [
                self.sample_data['x_l'] * 50.0 + 50.0,
                0 * self.sample_data['x_l'],
                0 * self.sample_data['x_l'],
            ],
            dim=1,
        )
        out['x_gray'] = lab_to_rgb(x_gray_lab)
        out['ref'] = out['ref'] * 0.5 + 0.5
        if not self.cfg.REFCOLOR.CORR.WARP_FEAT:
            out['warp_image'] = (
                F.interpolate(
                    out['warp_image'][:, 0:3, :, :],
                    size=out['x_gray'].size()[2:],
                    mode='bilinear',
                ) * 0.5 + 0.5)
            images = torch.cat(
                [
                    out['x_gray'],
                    out['ref'],
                    out['warp_image'],
                    out['fake_rgb'],
                    self.sample_data['x_rgb'] * 0.5 + 0.5,
                ],
                dim=3,
            ).data.cpu()
        else:
            images = torch.cat(
                [
                    out['x_gray'],
                    out['ref'],
                    out['fake_rgb'],
                    self.sample_data['x_rgb'] * 0.5 + 0.5,
                ],
                dim=3,
            ).data.cpu()

        self.save_single_images(images, ('gray', 'ref', 'warp', 'fake', 'gt'))
        if self.cfg.DATA.FULL_RES_OUTPUT:
            self.save_single_images(out['fake_rgb_full_res'], ('full_resolution_results', ))

        try:
            save_image(
                images,
                os.path.join(
                    self.cfg.TEST.LOG_DIR,
                    'out_{}.png'.format(info),
                ),
                normalize=True,
                nrow=1,
            )
        except:  # noqa: E722
            pass

        self.net_gen.train()
        self.net_corr.train()

    def save_single_images(self, images, name_list):

        n, _, _, w = images.size()
        assert w % len(name_list) == 0
        for i, sub_images in enumerate(torch.split(images, w // len(name_list), dim=-1)):
            p = os.path.join(
                self.cfg.TEST.LOG_DIR,
                name_list[i],
            )
            try:
                os.makedirs(p)
            except:  # noqa: E722
                pass
            for j in range(n):
                img_name = self.sample_data['image_name'][j]
                save_image(
                    sub_images[j],
                    os.path.join(p, f'{img_name}.png'),
                    normalize=True,
                    nrow=1,
                )
