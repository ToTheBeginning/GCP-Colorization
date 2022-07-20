import torch
from abc import abstractmethod


class BaseSolver:

    def __init__(self, cfg):
        self.device = torch.device('cuda')
        self.cfg = cfg

    @abstractmethod
    def test(self):
        # TODO EVALUATION METRIC
        raise NotImplementedError

    def to_device(self):
        for k in self.__dict__.keys():
            if k.startswith('net'):
                self.__dict__[k] = self.__dict__[k].to(self.device)

    def load_from_ckp(self):
        ckp = torch.load(self.cfg.TEST.LOAD_FROM)
        for k in self.__dict__.keys():
            if k.startswith('net'):
                self.__dict__[k].load_state_dict(ckp[k])
                self.__dict__[k].eval()

    def run(self):
        self.test()

    def read_data_from_dataiter(self, data_iter):
        self.sample_data = next(data_iter)
        for key in self.sample_data.keys():
            if key != 'image_name':
                self.sample_data[key] = self.sample_data[key].to(self.device)
            else:
                self.sample_data[key] = self.sample_data[key]

        if 'x_l' not in self.sample_data:
            if '100class' not in self.cfg.BIGGAN_PRETRAIN.ENC:
                from utils.util import rgb_to_lab

                lab = rgb_to_lab((self.sample_data['x_rgb'] + 1.0) / 2.0)
                self.sample_data['x_l'] = lab[:, 0:1, :, :] / 50.0 - 1.0
                # self.sample_data['x_ab'] = lab[:, 1:, :, :] / 110.0
                if self.cfg.DATA.FULL_RES_OUTPUT:
                    lab_full_res = rgb_to_lab((self.sample_data['x_full_res'] + 1.0) / 2.0)
                    self.sample_data['x_l_full_res'] = lab_full_res[:, 0:1, :, :] / 50.0 - 1.0
            else:
                self.sample_data['x_l'] = (0.2989 * self.sample_data['x_rgb'][:, 0:1, :, :] +
                                           0.5870 * self.sample_data['x_rgb'][:, 1:2, :, :] +
                                           0.1140 * self.sample_data['x_rgb'][:, 2:3, :, :])
