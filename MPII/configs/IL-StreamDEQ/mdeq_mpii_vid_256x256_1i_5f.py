_base_ = './mdeq_mpii_vid_256x256_1i_0f.py'


data = dict(
    val=dict(num_prev_frames=5),
    test=dict(num_prev_frames=5))
