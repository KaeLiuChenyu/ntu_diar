import torch
import math
import logging

from ntu_diar.scr.utils.checkpoints import (
  register_checkpoint_hooks,
  mark_as_saver,
  mark_as_loader,
)

logger = logging.getLogger(__name__)

#---------------------------------------------------------#
class LinearScheduler:
#---------------------------------------------------------#
    """Scheduler with linear annealing technique.
    The learning rate linearly decays over the specified number of epochs.
    Arguments
    ---------
    initial_value : float
        The value upon initialization.
    final_value : float
        The value used when the epoch count reaches ``epoch_count - 1``.
    epoch_count : int
        Number of epochs.
    Example
    -------
    >>> scheduler = LinearScheduler(1.0, 0.0, 4)
    >>> scheduler(current_epoch=1)
    (1.0, 0.666...)
    >>> scheduler(current_epoch=2)
    (0.666..., 0.333...)
    >>> scheduler(current_epoch=3)
    (0.333..., 0.0)
    >>> scheduler(current_epoch=4)
    (0.0, 0.0)
    """

    def __init__(self, initial_value, final_value, epoch_count):
        self.value_at_epoch = torch.linspace(
            initial_value, final_value, steps=epoch_count
        ).tolist()

    def __call__(self, current_epoch):
        """Returns the current and new value for the hyperparameter.
        Arguments
        ---------
        current_epoch : int
            Number of times the dataset has been iterated.
        """
        old_index = max(0, current_epoch - 1)
        index = min(current_epoch, len(self.value_at_epoch) - 1)
        return self.value_at_epoch[old_index], self.value_at_epoch[index]

#---------------------------------------------------------#
def update_learning_rate(optimizer, new_lr, param_group=None):
#---------------------------------------------------------#
    """Change the learning rate value within an optimizer.
    Arguments
    ---------
    optimizer : torch.optim object
        Updates the learning rate for this optimizer.
    new_lr : float
        The new value to use for the learning rate.
    param_group : list of int
        The param group indices to update. If not provided, all groups updated.
    Example
    -------
    >>> from torch.optim import SGD
    >>> from speechbrain.nnet.linear import Linear
    >>> model = Linear(n_neurons=10, input_size=10)
    >>> optimizer = SGD(model.parameters(), lr=0.1)
    >>> update_learning_rate(optimizer, 0.2)
    >>> optimizer.param_groups[0]["lr"]
    0.2
    """
    # Iterate all groups if none is provided
    if param_group is None:
        groups = range(len(optimizer.param_groups))
    else:
        groups = param_group

    for i in groups:
        old_lr = optimizer.param_groups[i]["lr"]

        # Change learning rate if new value is different from old.
        if new_lr != old_lr:
            optimizer.param_groups[i]["lr"] = new_lr
            optimizer.param_groups[i]["prev_lr"] = old_lr
            logger.info("Changing lr from %.2g to %.2g" % (old_lr, new_lr))


@register_checkpoint_hooks
#---------------------------------------------------------#
class CyclicLRScheduler:
#---------------------------------------------------------#
    """This implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see the reference paper.

    Arguments
    ---------
    base_lr : float
        initial learning rate which is the
        lower boundary in the cycle.
    max_lr : float
        upper boundary in the cycle. Functionally,
        it defines the cycle amplitude (max_lr - base_lr).
        The lr at any cycle is the sum of base_lr
        and some scaling of the amplitude; therefore
        max_lr may not actually be reached depending on
        scaling function.
    step_size : int
        number of training iterations per
        half cycle. The authors suggest setting step_size
        2-8 x training iterations in epoch.
    mode : str
        one of {triangular, triangular2, exp_range}.
        Default 'triangular'.
        Values correspond to policies detailed above.
        If scale_fn is not None, this argument is ignored.
    gamma : float
        constant in 'exp_range' scaling function:
        gamma**(cycle iterations)
    scale_fn : lambda function
        Custom scaling policy defined by a single
        argument lambda function, where
        0 <= scale_fn(x) <= 1 for all x >= 0.
        mode parameter is ignored
    scale_mode : str
        {'cycle', 'iterations'}.
        Defines whether scale_fn is evaluated on
        cycle number or cycle iterations (training
        iterations since start of cycle). Default is 'cycle'.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(input_size=3, n_neurons=4)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1)
    >>> output = model(inp_tensor)
    >>> scheduler = CyclicLRScheduler(base_lr=0.1, max_lr=0.3, step_size=2)
    >>> scheduler.on_batch_end(optim)
    >>> optim.param_groups[0]["lr"]
    0.2
    >>> scheduler.on_batch_end(optim)
    >>> optim.param_groups[0]["lr"]
    0.3
    >>> scheduler.on_batch_end(optim)
    >>> optim.param_groups[0]["lr"]
    0.2
    """

    def __init__(
        self,
        base_lr=0.001,
        max_lr=0.006,
        step_size=2000.0,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
    ):
        super(CyclicLRScheduler, self).__init__()

        self.losses = []
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.0

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.0

    def __call__(self, epoch):
        old_lr = self.current_lr
        new_lr = self.clr(self.clr_iterations + 1)

        return old_lr, new_lr

    def clr(self, clr_iterations):
        """Clears interations."""
        cycle = math.floor(1 + clr_iterations / (2 * self.step_size))
        x = abs(clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == "cycle":
            return self.base_lr + (self.max_lr - self.base_lr) * max(
                0, (1 - x)
            ) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * max(
                0, (1 - x)
            ) * self.scale_fn(clr_iterations)

    def on_batch_end(self, opt):
        """
        Arguments
        ---------
        opt : optimizers
            The optimizers to update using this scheduler.
        """
        self.clr_iterations += 1

        lr = self.clr(self.clr_iterations)
        current_lr = opt.param_groups[0]["lr"]

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr

    @mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {"losses": self.losses, "clr_iterations": self.clr_iterations}
        torch.save(data, path)

    @mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.clr_iterations = data["clr_iterations"]