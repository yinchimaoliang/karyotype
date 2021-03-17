import argparse
import mmcv
import numpy as np
import os
from os import path as osp

from utils import (karyotype_classify, karyotype_detect, karyotype_segment,
                   rotate_img, show_diagnose_result)

CLASS_NAMES = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
    '15', '16', '17', '18', '19', '20', '21', '22', 'x', 'y'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Karyotype diagnose')
    parser.add_argument('--detect-config', help='detect config file path')
    parser.add_argument('--segment-config', help='segment config file path')
    parser.add_argument('--classify-config', help='classify config file path')
    parser.add_argument(
        '--detect-ckpt-path', help='detect checkpoint file path')
    parser.add_argument(
        '--segment-ckpt-path', help='segment checkpoint file path')
    parser.add_argument(
        '--classify-ckpt-path', help='classify checkpoint file path')
    parser.add_argument('--img-path', help='image path')
    parser.add_argument('--result-path', help='path to save the result')
    args = parser.parse_args()

    return args


def detect(detect_config, detect_ckpt_path, img_path, result_path):
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


def segment(segment_config, segment_ckpt_path, img_path):
    img_names = os.listdir(img_path)
    for img_name in img_names:
        chromosome = mmcv.imread(osp.join(img_path, img_name), 0)
        result = karyotype_segment(segment_config, segment_ckpt_path,
                                   osp.join(img_path, img_name))
        chromosome_masked = np.multiply(chromosome, result[0])
        chromosome_rotated = rotate_img(chromosome_masked)
        mmcv.imwrite(chromosome_rotated, osp.join(img_path, img_name))


def classify(classify_config, classify_ckpt_path, img_path):
    img_names = os.listdir(img_path)
    for img_name in img_names:
        chromosome = mmcv.imread(osp.join(img_path, img_name), 0)
        result = karyotype_classify(classify_config, classify_ckpt_path,
                                    osp.join(img_path, img_name))
        if result['pred_label'] == 1:
            chromosome = mmcv.imrotate(chromosome, 180)
        mmcv.imwrite(chromosome, osp.join(img_path, img_name))


def main():
    args = parse_args()
    detect_config = args.detect_config
    segment_config = args.segment_config
    classify_config = args.classify_config
    detect_ckpt_path = args.detect_ckpt_path
    segment_ckpt_path = args.segment_ckpt_path
    classify_ckpt_path = args.classify_ckpt_path
    img_path = args.img_path
    result_path = args.result_path

    mmcv.mkdir_or_exist(osp.join(result_path, 'detect_results'))
    detect(detect_config, detect_ckpt_path, img_path, result_path)
    segment(segment_config, segment_ckpt_path,
            osp.join(result_path, 'detect_results'))
    classify(classify_config, classify_ckpt_path,
             osp.join(result_path, 'detect_results'))
    show_diagnose_result(
        osp.join(result_path, 'detect_results'),
        osp.join(result_path, 'karyotype_diagnose.png'), CLASS_NAMES)


if __name__ == '__main__':
    main()
