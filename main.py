import argparse

from apis import (inference_classifer, inference_detector, inference_segmentor,
                  init_classifier, init_detector, init_segmentor)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--ckpt-path', help='checkpoint file path')
    parser.add_argument('--img-path', help='image path')
    parser.add_argument(
        '--type',
        choices=[
            'karyotype_segment', 'karyotype_classify', 'karyotype_count',
            'karyotype_polarity'
        ],
        help='type of the job')
    args = parser.parse_args()

    return args


def karyotype_segment(config_path, ckpt_path, img_path):
    segmentor = init_segmentor(config_path, ckpt_path)
    result = inference_segmentor(segmentor, img_path)
    print(result)


def karyotype_classify(config_path, ckpt_path, img_path):
    detector = init_detector(config_path, ckpt_path)
    result = inference_detector(detector, img_path)
    print(result)


def karyotype_polarity(config_path, ckpt_path, img_path):
    classifier = init_classifier(config_path, ckpt_path)
    result = inference_classifer(classifier, img_path)
    print(result)


def main():
    args = parse_args()
    config_path = args.config
    ckpt_path = args.ckpt_path
    img_path = args.img_path
    if args.type == 'karyotype_segment':
        karyotype_segment(config_path, ckpt_path, img_path)
    if args.type == 'karyotype_classify':
        karyotype_classify(config_path, ckpt_path, img_path)
    if args.type == 'karyotype_polarity':
        karyotype_polarity(config_path, ckpt_path, img_path)


if __name__ == '__main__':
    main()
