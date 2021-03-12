import argparse

from apis import inference_segmentor, init_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--ckpt-path', help='checkpoint file path')
    parser.add_argument('--img-path', help='image path')
    args = parser.parse_args()

    return args


def karyotype_segment(config_path, ckpt_path, img_path):
    segmentor = init_segmentor(config_path, ckpt_path)
    result = inference_segmentor(segmentor, img_path)
    print(result)


def main():
    args = parse_args()
    config_path = args.config
    ckpt_path = args.ckpt_path
    img_path = args.img_path
    karyotype_segment(config_path, ckpt_path, img_path)


if __name__ == '__main__':
    main()
