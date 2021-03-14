_base_ = [
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/datasets/karyotype_count.py'
]

model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    roi_head=dict(bbox_head=dict(num_classes=1)))