_base_ = [
    "../_base_/six_gray_sim_data.py",
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

test_data = dict(
    mask_path="test_datasets/mask/gray_mask.mat",
    mask_shape=None,
)
seed=42
model = dict(
    type='DU',
    dim=48,
    color_dim=1,
    stage=5,
    cycle=4,
)

eval = dict(
    flag=True,
    interval=1
)
amp = True
checkpoints = "checkpoint/gray_5stg.pth"
