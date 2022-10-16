import os
import torch.utils.data as data
import torchvision.transforms as transforms
from os import path as osp
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    # code from BasicSR codebase
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNetInference(data.Dataset):

    def __init__(self, cfg):
        label_map = {}
        with open('assets/imagenet_val_label_map.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                label_map[line.strip().split(' ')[0]] = int(line.strip().split(' ')[-1])

        if osp.exists(cfg.DATA.USER_IMAGENET_LABEL):
            with open(cfg.DATA.USER_IMAGENET_LABEL, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    label_map[line.strip().split(' ')[0]] = int(line.strip().split(' ')[-1])

        inference_folder = cfg.DATA.INFERENCE_FOLDER
        # img_list = sorted(os.listdir(inference_folder))
        if osp.isfile(inference_folder):
            img_list = [inference_folder]
        else:
            img_list = sorted(scandir(inference_folder, recursive=True, full_path=True))
        self.imgs = []
        for img_path in img_list:
            if not is_image_file(img_path):
                continue
            img_name = osp.splitext(osp.basename(img_path))[0]
            assert img_name in label_map
            self.imgs.append((img_name, img_path, label_map[img_name]))

        if len(self.imgs) == 0:
            raise RuntimeError(f'no images in folder {inference_folder}')

        self.loader = pil_loader

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.resize_transform = transforms.Resize((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))

        self.crop_size = cfg.DATA.CROP_SIZE
        self.cfg = cfg

    def __getitem__(self, index):
        data = {}
        image_name, path, target = self.imgs[index]
        img = self.loader(path)

        if self.cfg.DATA.CENTER_CROP:
            w, h = img.size
            mini_size = min(w, h)
            crop_resize_transform = transforms.Compose(
                [transforms.CenterCrop(mini_size),
                 transforms.Resize(self.crop_size, interpolation=3)])
        else:
            crop_resize_transform = self.resize_transform

        data['x_rgb'] = self.transform(crop_resize_transform(img))
        data['cid'] = int(target)
        data['image_name'] = image_name
        if self.cfg.DATA.FULL_RES_OUTPUT:
            data['x_full_res'] = self.transform(img)

        return data

    def __len__(self):
        return len(self.imgs)
