from yacs.config import CfgNode

_C = CfgNode()

_C.SEED = 123
_C.MODE = 'test'

_C.DATA = CfgNode()
_C.DATA.CENTER_CROP = False
_C.DATA.FULL_RES_OUTPUT = True
_C.DATA.CROP_SIZE = 256
_C.DATA.BS = 32
_C.DATA.NUM_WORKER = 1
_C.DATA.NAME = 'imagenet_inference'
_C.DATA.INFERENCE_FOLDER = 'test_images'
_C.DATA.USER_IMAGENET_LABEL = 'assets/predicted_label_for_user_image.txt'

_C.MODEL = CfgNode()
_C.MODEL.NAME = 'REFCOLOR'

_C.DIVERSE = CfgNode()
_C.DIVERSE.CKP = 'assets/deformator_for_biggan.pt'
_C.DIVERSE.DIRECTION = -2
_C.DIVERSE.SHIFT_RANGE = 16
_C.DIVERSE.SHIFT_COUNT = 3

_C.TEST = CfgNode()
_C.TEST.LOAD_FROM = 'assets/colorization_model.pth'

_C.BIGGAN_PRETRAIN = CfgNode()
_C.BIGGAN_PRETRAIN.DIS = 'assets/biggan_D_256.pth'
_C.BIGGAN_PRETRAIN.ENC = 'assets/biggan_E.pth'
_C.BIGGAN_PRETRAIN.GEN = 'assets/biggan_G_ema_256.pth'


def prepare_ref_options(cfg):
    cfg.REFCOLOR = CfgNode()

    cfg.REFCOLOR.CORR = CfgNode()
    cfg.REFCOLOR.CORR.USE_VGG = True
    cfg.REFCOLOR.CORR.MAX_INFERENCE_DEPTH = 3
    cfg.REFCOLOR.CORR.FEAT_NC = 576
    cfg.REFCOLOR.CORR.NONLOCAL_NC = 128
    cfg.REFCOLOR.CORR.TEMP = 0.01
    cfg.REFCOLOR.CORR.WARP_FEAT = False
    cfg.REFCOLOR.CORR.CONCAT_CORR = False
    cfg.REFCOLOR.CORR.GRL_AFTER_NORM = True

    cfg.REFCOLOR.GEN = CfgNode()
    cfg.REFCOLOR.GEN.IN_NC = 1
    cfg.REFCOLOR.GEN.OUT_NC = 2
    cfg.REFCOLOR.GEN.N_RES = 6
    cfg.REFCOLOR.GEN.N_DOWN = 2
    cfg.REFCOLOR.GEN.SPADE_NC = 384

    cfg.REFCOLOR.DIS = CfgNode()
    cfg.REFCOLOR.DIS.N_DIS = 3
    cfg.REFCOLOR.DIS.TO_RGB = False

    return cfg


def assert_and_infer_cfg(cfg):
    assert cfg.MODE in ['test']

    assert not (cfg.DATA.CENTER_CROP and cfg.DATA.FULL_RES_OUTPUT)

    return cfg


def get_cfg():
    return _C.clone()
