# Author: Xiaoyang Wu (xiaoyang.wu@smartmore.com)
import yacs.config as config


def make_config(config_file):
    with open(config_file, 'r') as f:
        cfg = config.load_cfg(f)
        cfg.set_new_allowed(True)
    return cfg.clone()