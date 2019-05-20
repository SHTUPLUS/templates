

def get_params(model, prefixs, suffixes, exclude=None):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    for name, module in model.named_modules():
        for prefix in prefixs:
            if name == prefix:
                for n, p in module.named_parameters():
                    n = '.'.join([name, n])
                    if type(exclude) == list and n in exclude:
                        continue
                    if type(exclude) == str and exclude in n:
                        continue

                    for suffix in suffixes:
                        if (n.split('.')[-1].startswith(suffix) or n.endswith(suffix)) and p.requires_grad:
                            yield p
                break


def create_logger(cfg, postfix=''):
    """Set up the logger for saving log file on the disk
    Args:
        cfg: configuration dict
        postfix: postfix of the log file name

    Return:
        logger: a logger for record essential information
    """
    import logging
    import os
    from logging.config import dictConfig
    import time

    logging_config = dict(
        version=1,
        formatters={
            'f_t': {'format':
                        '\n %(asctime)s | %(levelname)s | %(name)s \t %(message)s'}

        },
        handlers={
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'f_t',
                'level': logging.INFO},
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'f_t',
                'level': logging.INFO,
                'filename': None,
            }
        },
        root={
            'handlers': ['stream_handler', 'file_handler'],
            'level': logging.DEBUG,
        },
    )
    # set up logger
    log_file = '{}_{}_{}.log'.format(cfg.network.name, cfg.dataset.name, postfix + time.strftime('%Y-%m-%d-%H-%M'))
    log_file_path = os.path.join(cfg.savedir, log_file)

    logging_config['handlers']['file_handler']['filename'] = log_file_path

    open(log_file_path, 'w').close()  # Clear the content of logfile
    # get logger from dictConfig
    dictConfig(logging_config)

    logger = logging.getLogger()

    return logger


def random_init(seed=0, deterministic=False):
    """Set the seed for the random for torch and random package
    Args:
        seed: random seed
    """
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


