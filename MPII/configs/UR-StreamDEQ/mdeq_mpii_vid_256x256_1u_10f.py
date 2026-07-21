_base_ = './mdeq_mpii_vid_256x256_1u_0f.py'


data = dict(
    val=dict(num_prev_frames=10),
    test=dict(num_prev_frames=10))
