from sacred import Experiment
from easydict import EasyDict as edict
from lib.utils import utils

ex = Experiment()

def initialization(config, seed, logname_postfix):
    cfg = edict(config)
    utils.random_init(seed)
    logger = utils.create_logger(cfg, postfix=logname_postfix)
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