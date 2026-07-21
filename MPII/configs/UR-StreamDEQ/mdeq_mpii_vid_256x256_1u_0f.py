_base_ = '../IL-StreamDEQ/mdeq_mpii_vid_256x256_1i_0f.py'


model = dict(
    backbone=dict(deq_cfg='configs/mdeq/mdeq_cls_cfg_unroll_20.yaml'),
    f_thres=1)
