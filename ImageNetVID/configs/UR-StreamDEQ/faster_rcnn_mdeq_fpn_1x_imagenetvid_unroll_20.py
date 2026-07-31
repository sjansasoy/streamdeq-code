_base_ = '../baseline/faster_rcnn_mdeq_fpn_1x_imagenetvid.py'


model = dict(
    backbone=dict(deq_cfg='configs/mdeq/mdeq_det_cfg_unroll_20.yaml'))
