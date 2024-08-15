from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def get_2module_warmup_scheduler(
    optimizer: Optimizer,
    module1_baselr: float,
    module2_warmup_start_step: int,
    module2_warmup_steps: int,
    module2_target_lr: float,
    adjust_module1_lr_to_module2_postwarmup_lr: bool,
):
    """
    Returns a scheduler that will linearly increase the learning rate of module2 from 0 to module2_target_lr starting at module2_warmup_start_step.
    If adjust_module1_lr_to_module2_postwarmup_lr is True, the learning rate of module1 will be adjusted thoughout the module2 warmup phase to match the learning rate of module2 at the end of the warmup.
    """

    def module2_lr(step: int):
        if step < module2_warmup_start_step:
            return 0.0
        return min(
            1.0,
            (
                (step - module2_warmup_start_step) / module2_warmup_steps
                if module2_warmup_steps > 0
                else 1.0
            ),
        )

    def module1_lr(step: int):
        if (
            not adjust_module1_lr_to_module2_postwarmup_lr
            or module2_target_lr is None
            or module2_target_lr == 0.0
        ):
            return 1.0
        if step < module2_warmup_start_step:
            return 1.0
        target_factor = module2_target_lr / module1_baselr
        if step >= module2_warmup_start_step + module2_warmup_steps:
            return target_factor
        return (
            1.0
            + (target_factor - 1.0)
            * (step - module2_warmup_start_step)
            / module2_warmup_steps
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[
            module1_lr,
            module2_lr,
        ],
        verbose=True,
    )
    return scheduler
