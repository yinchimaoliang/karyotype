import argparse

from utils import karyotype_detect


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--ckpt-path', help='checkpoint file path')
    parser.add_argument('--img-path', help='image path')
    parser.add_argument('--result-path', help='path to save the result')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config_path = args.config
    ckpt_path = args.ckpt_path
    img_path = args.img_path
    result_path = args.result_path  # noqa: F841

    karyotype_detect(config_path, ckpt_path, img_path)


if __name__ == '__main__':
    main()
