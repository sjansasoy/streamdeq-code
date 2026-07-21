_base_ = './mdeq_mpii_vid_256x256_1u_0f.py'


model = dict(f_thres=8)

data = dict(
    val=dict(num_prev_frames=20),
    test=dict(num_prev_frames=20))
