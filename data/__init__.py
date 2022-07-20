import importlib
import logging
import torch

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)


def get_loader(cfg, ds='ImageNet'):
    datasetlib = importlib.import_module('data.' + ds.lower())

    dataset_cls = None
    target_dataset_name = ds.replace('_', '')
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset_cls = cls

    if dataset_cls is None:
        raise ValueError('DATASET NOT FOUND')

    dataset = dataset_cls(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA.BS,
        num_workers=cfg.DATA.NUM_WORKER,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
    )

    return dataloader
