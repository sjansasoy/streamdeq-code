_base_ = './faster_rcnn_mdeq_fpn_1x_imagenetvid_unroll_20_stream_1f_1i.py'


model = dict(extra=dict(f_thres=2))
