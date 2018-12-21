from sacred import Experiment
from easydict import EasyDict as edict
import plus.utils.utils as utils
import torch

ex = Experiment()

def initialization(config, seed, mode):
    # Add it if your input size is fixed 
    # ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True 
    
    cfg = edict(config)
    utils.random_init(seed)
    logger = utils.make_logger(f'{cfg.exp.name}_{mode}', savedir=cfg.exp.savedir)
    
    return cfg, logger

@ex.command
def train(_run, _rnd, _seed):
    cfg, ex.logger = initialization(_run.config, _seed, 'train')

    model = xxx()
    model.train()

@ex.command
def test(_run, _rnd, _seed):
    cfg, ex.logger = initialization(_run.config, _seed, 'test')

    model = xxx()
    model.test()


if __name__ == '__main__':

    ex.add_config('./configs/default.yaml')
    ex.run_commandline()
