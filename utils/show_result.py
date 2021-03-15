import cv2
import mmcv
import numpy as np
from mmcv.image import imread, imwrite
from mmcv.visualization import color_val


def imshow_count_bboxes(img,
                        bboxes,
                        score_thr=0,
                        bbox_color='green',
                        thickness=1,
                        out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)
    img = np.ascontiguousarray(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]

    bbox_color = color_val(bbox_color)

    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
    if out_file is not None:
        imwrite(img, out_file)
    return img


def show_count_result(img,
                      result,
                      score_thr=0.3,
                      bbox_color='green',
                      thickness=1,
                      out_file=None):
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, _ = result
    else:
        bbox_result, _ = result, None
    bboxes = np.vstack(bbox_result)
    # draw bounding boxes
    imshow_count_bboxes(
        img,
        bboxes,
        score_thr=score_thr,
        bbox_color=bbox_color,
        thickness=thickness,
        out_file=out_file)