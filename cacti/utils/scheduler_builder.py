from cacti.utils.registry import Registry, build_from_cfg
import torch
import inspect

# 定义 Scheduler Registry
SCHEDULERS = Registry('scheduler')

def register_torch_schedulers():
    torch_schedulers = []
    scheduler_module = torch.optim.lr_scheduler

    for name in dir(scheduler_module):
        if name.startswith('__'):
            continue
        cls = getattr(scheduler_module, name)
        if not inspect.isclass(cls):
            continue

        if cls.__name__ == '_LRScheduler':
            continue

        try:
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())
            if len(params) > 1 and params[1].name == 'optimizer':
                SCHEDULERS.register_module(cls)
                torch_schedulers.append(name)
        except (TypeError, ValueError):
            continue

    # 手动注册 ReduceLROnPlateau
    if 'ReduceLROnPlateau' not in torch_schedulers:
        SCHEDULERS.register_module(torch.optim.lr_scheduler.ReduceLROnPlateau)
        torch_schedulers.append('ReduceLROnPlateau')

    return torch_schedulers

# 注册所有调度器
register_torch_schedulers()

def build_scheduler(cfg, default_args=None):
    scheduler_type = cfg.get('type')
    if scheduler_type == 'ReduceLROnPlateau':
        if default_args is not None and 'metric' in default_args:
            cfg = cfg.copy()
            cfg['metric'] = default_args['metric']
    scheduler = build_from_cfg(cfg, SCHEDULERS, default_args)
    return scheduler
