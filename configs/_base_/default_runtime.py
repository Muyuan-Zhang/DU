checkpoint_config = dict(interval=1)

log_config = dict(
    interval=100,
)
save_image_config = dict(
    interval=250,
)
optimizer = dict(type='Adam', lr=0.0004, betas=(0.9, 0.999), eps=1e-08)

epoch = 150

scheduler = dict(type='CosineAnnealingLR', T_max=epoch, eta_min=1e-6)

loss = dict(type='MSELoss')

resume = None

# seed = 42
# seed = 22
# seed = 10
# seed = 42
seed = 42
# gray 22 10
runner = dict(max_epochs=epoch)

checkpoints = None
