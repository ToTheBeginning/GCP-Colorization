#!/usr/bin/env python3

import importlib
import numpy as np
import torch

from utils.parser import load_config, parse_args


def main():
    args = parse_args()
    cfg = load_config(args)
    torch.manual_seed(cfg.SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.SEED)
    solverlib = importlib.import_module('solvers.' + args.model.lower() + '_solver')
    solver_cls = None
    target_solver_name = args.model.replace('_', '') + 'Solver'
    for name, cls in solverlib.__dict__.items():
        if name.lower() == target_solver_name.lower():
            solver_cls = cls

    if solver_cls is None:
        raise ValueError('SOLVER NOT FOUND')

    sol = solver_cls(cfg)
    sol.run()


if __name__ == '__main__':
    main()
