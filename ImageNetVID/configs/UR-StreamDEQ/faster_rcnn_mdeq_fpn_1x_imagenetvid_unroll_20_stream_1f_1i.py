_base_ = '../IL-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_1f_1i.py'


model = dict(
    backbone=dict(deq_cfg='configs/mdeq/mdeq_det_cfg_unroll_20.yaml'))
