_base_ = './faster_rcnn_mdeq_fpn_1x_imagenetvid_unroll_20_stream_1f_1i.py'


model = dict(extra=dict(f_thres=2))

data = dict(
    val=dict(ref_img_sampler=dict(num_ref_imgs=10, frame_range=[-10, 0])),
    test=dict(ref_img_sampler=dict(num_ref_imgs=10, frame_range=[-10, 0])))
