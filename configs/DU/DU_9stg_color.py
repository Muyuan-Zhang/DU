_base_ = [
    "../_base_/matlab_bayer.py",
]
test_data = dict(
    mask_path=None,
    mask_shape=(512, 512, 16),
    rot_flip_flag=True,
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)
seed=42

model = dict(
    type='DU',
    dim=48,
    color_dim=3,
    stage=9,
    cycle=4,
)

amp = True
checkpoints = None
checkpoints = "checkpoint/color_9stg.pth"
