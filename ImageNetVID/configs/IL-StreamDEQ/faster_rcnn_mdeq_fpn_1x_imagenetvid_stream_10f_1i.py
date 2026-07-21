_base_ = './faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_1f_1i.py'


data = dict(
    val=dict(ref_img_sampler=dict(num_ref_imgs=10, frame_range=[-10, 0])),
    test=dict(ref_img_sampler=dict(num_ref_imgs=10, frame_range=[-10, 0])))
