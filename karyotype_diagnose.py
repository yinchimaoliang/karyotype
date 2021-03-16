import argparse
import mmcv
import numpy as np
import os
from os import path as osp

from utils import karyotype_detect, karyotype_segment, rotate_img

CLASS_NAMES = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
    '15', '16', '17', '18', '19', '20', '21', '22', 'x', 'y'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Karyotype diagnose')
    parser.add_argument('--detect-config', help='detect config file path')
    parser.add_argument('--segment-config', help='segment config file path')
    parser.add_argument('--polarity-config', help='polarity config file path')
    parser.add_argument(
        '--detect-ckpt-path', help='detect checkpoint file path')
    parser.add_argument(
        '--segment-ckpt-path', help='segment checkpoint file path')
    parser.add_argument(
        '--polarity-ckpt-path', help='polarity checkpoint file path')
    parser.add_argument('--img-path', help='image path')
    parser.add_argument('--result-path', help='path to save the result')
    args = parser.parse_args()

    return args


def detect_karyotype(detect_config, detect_ckpt_path, img_path, result_path):
    img = mmcv.imread(img_path, 0)
    detect_results = karyotype_detect(detect_config, detect_ckpt_path,
                                      img_path)
    assert len(detect_results) == len(CLASS_NAMES)
    for i, detect_result in enumerate(detect_results):
        for j in range(len(detect_result)):
            chromosome = img[int(detect_result[j][1]):int(detect_result[j][3]),
                             int(detect_result[j][0]):int(detect_result[j][2])]
            mmcv.imwrite(
                chromosome,
                osp.join(result_path, 'detect_results',
                         f'{CLASS_NAMES[i]}_{j}.png'))


def segment_karyotype(segment_config, segment_ckpt_path, img_path):
    img_names = os.listdir(img_path)
    for img_name in img_names:
        chromosome = mmcv.imread(osp.join(img_path, img_name), 0)
        result = karyotype_segment(segment_config, segment_ckpt_path,
                                   osp.join(img_path, img_name))
        chromosome_masked = np.multiply(chromosome, result[0])
        chromosome_rotated = rotate_img(chromosome_masked)
        mmcv.imwrite(chromosome_rotated, osp.join(img_path, img_name))


def main():
    args = parse_args()
    detect_config = args.detect_config
    segment_config = args.segment_config  # noqa: F841
    polarity_config = args.polarity_config  # noqa: F841
    detect_ckpt_path = args.detect_ckpt_path
    segment_ckpt_path = args.segment_ckpt_path  # noqa: F841
    polarity_ckpt_path = args.polarity_ckpt_path  # noqa: F841
    img_path = args.img_path
    result_path = args.result_path

    mmcv.mkdir_or_exist(osp.join(result_path, 'detect_results'))
    detect_karyotype(detect_config, detect_ckpt_path, img_path, result_path)
    segment_karyotype(segment_config, segment_ckpt_path,
                      osp.join(result_path, 'detect_results'))


if __name__ == '__main__':
    main()
