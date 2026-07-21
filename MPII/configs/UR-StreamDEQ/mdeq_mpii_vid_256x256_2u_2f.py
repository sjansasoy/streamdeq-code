_base_ = './mdeq_mpii_vid_256x256_1u_0f.py'


model = dict(f_thres=2)

data = dict(
    val=dict(num_prev_frames=2),
    test=dict(num_prev_frames=2))
