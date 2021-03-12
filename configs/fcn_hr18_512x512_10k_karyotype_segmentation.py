_base_ = [
    './_base_/models/fcn_hr18.py',
    './_base_/datasets/karyotype_segmentation.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg, ), decode_head=dict(norm_cfg=norm_cfg))
