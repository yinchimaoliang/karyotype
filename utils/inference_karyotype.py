from apis import (inference_classifer, inference_detector, inference_segmentor,
                  init_classifier, init_detector, init_segmentor)


def karyotype_segment(config_path, ckpt_path, img_path):
    segmentor = init_segmentor(config_path, ckpt_path)
    result = inference_segmentor(segmentor, img_path)
    print(result)


def karyotype_detect(config_path, ckpt_path, img_path):
    detector = init_detector(config_path, ckpt_path)
    result = inference_detector(detector, img_path)
    print(result)


def karyotype_polarity(config_path, ckpt_path, img_path):
    classifier = init_classifier(config_path, ckpt_path)
    result = inference_classifer(classifier, img_path)
    print(result)
