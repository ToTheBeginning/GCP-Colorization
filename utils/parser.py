import argparse
import time
from pathlib import Path

from config.defaults import assert_and_infer_cfg, get_cfg, prepare_ref_options


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        default='',
        type=str,
    )
    parser.add_argument(
        '--expname',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--model',
        type=str,
        default='REFCOLOR',
    )
    parser.add_argument(
        '--test_folder',
        type=str,
        default='test_images',
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=1,
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def mkdir(ps):
    ps = [Path(p) for p in ps]
    for p in ps:
        if not p.exists():
            p.mkdir(parents=True)


def load_config(args):
    c = get_cfg()
    if args.model == 'REFCOLOR':
        c = prepare_ref_options(c)
    else:
        raise NotImplementedError
    if args.cfg != '':
        c.merge_from_file(args.cfg)
    if args.opts is not None:
        c.merge_from_list(args.opts)
    c = assert_and_infer_cfg(c)
    c.EXPNAME = args.expname
    c.MODEL.NAME = args.model
    timestr = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    c.DATA.INFERENCE_FOLDER = args.test_folder
    c.DATA.BS = args.bs

    if c.MODE == 'test':
        c.TEST.LOG_DIR = 'results/{}_{}'.format(args.expname, timestr)
        mkdir([c.TEST.LOG_DIR])
        return c

    raise NotImplementedError
