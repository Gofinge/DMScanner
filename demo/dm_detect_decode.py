# Author: Xiaoyang Wu (xiaoyang.wu@smartmore.com), Julian Zhang (julian.zhang@smartmore.com)
import os
import sys

abspath = os.path.dirname(__file__)
sys.path.insert(0, "abspath")
sys.path.append(".")

import argparse

from configs import make_config
from data import DMDataset
from engine import DMDetector, DMDecoder
from utils.miscellaneous import TqdmBar


def main():
    # add argument
    parser = argparse.ArgumentParser(description="SmartMore Data Matrix Scanner")
    parser.add_argument(
        "--config-file",
        default="default.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    config_file = args.config_file if os.path.isfile(args.config_file) else os.path.join("configs", args.config_file)

    # make config
    cfg = make_config(config_file)

    # generate output folder
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    idx = 1
    base_dir = output_dir
    while os.path.exists(output_dir):
        output_dir = base_dir+str(idx)
        idx += 1
    os.makedirs(output_dir)
    source_dir = os.path.join(cfg.SOURCE_DIR, cfg.DATASET[cfg.TARGET])
    dataset = DMDataset(source_dir)
    detector = DMDetector(cfg.DETECTOR, output_dir) if cfg.DETECTOR.ENABLE else None
    decoder = DMDecoder(cfg.DECODER, output_dir) if cfg.DECODER.ENABLE else None

    results = []
    count = 0
    bar = TqdmBar(dataset, dataset.__len__(), description="DM Scanner", use_bar=cfg.USE_BAR)
    for _, dm_image in bar.bar:
        if dm_image.img is not None:
            if detector:
                dm_image.ret_candidate = detector.detect(dm_image)
            if decoder:
                message = decoder.decode(dm_image)
                if message:
                    results.append(message)
            count += 1
            bar.set_postfix({"Correct": len(results), "Total": count})

    if decoder and cfg.DECODER.ARG.zxing_enable:
        print("Detect Method: {} \n Decode Method: {} \n Zxing Enable: {} \n Data Set: {} \n Results: {}/{} \n "
              "Accurate: {}".format(cfg.DETECTOR.METHOD, cfg.DECODER.METHOD, cfg.DECODER.ARG.zxing_enable,
                                    cfg.TARGET, len(results), count, len(results)/count))


if __name__ == '__main__':
    main()
