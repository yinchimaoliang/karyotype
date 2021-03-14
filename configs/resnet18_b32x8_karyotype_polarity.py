_base_ = [
    './_base_/models/resnet18.py', './_base_/datasets/karyotype_polarity.py'
]
# model settings
model = dict(head=dict(
    num_classes=2,
    topk=(1, 2),
))
