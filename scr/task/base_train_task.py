import os
import torch
import yaml
import time
import inspect
import logging
import ntu_diar
import warnings

from tqdm.contrib import tqdm
from types import SimpleNamespace
from contextlib import contextmanager
from torch.utils.data import DataLoader

from ntu_diar.scr.dataio.sample import ReproducibleRandomSampler
from ntu_diar.scr.task.utils import Stage
from ntu_diar.scr.utils.distributed import run_on_main
from ntu_diar.scr.utils.checkpoints import (
  mark_as_saver,
  mark_as_loader,
  register_checkpoint_hooks,
)

logger = logging.getLogger(__name__)


@register_checkpoint_hooks
class Task:
  
  #---------------------------------------------------------#
  def __init__(
  #---------------------------------------------------------#
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        profiler=None,
    ):
    self.opt_class = opt_class
    self.checkpointer = checkpointer
    self.profiler = profiler

    # Add defaults params
    run_opt_defaults = {
            "debug": False,
            "debug_batches": 2,
            "debug_epochs": 2,
            "device": "cpu",
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "find_unused_parameters": False,
            "jit_module_keys": None,
            "auto_mix_prec": False,
            "max_grad_norm": 5.0,
            "nonfinite_patience": 3,
            "noprogressbar": False,
            "ckpt_interval_minutes": 0,
            "grad_accumulation_factor": 1,
            "optimizer_step_limit": None,
        }

    for arg, default in run_opt_defaults.items():
        if run_opts is not None and arg in run_opts:
            if hparams is not None and arg in hparams:
                logger.info(
                    "Info: "
                    + arg
                    + " arg overridden by command line input to: "
                    + str(run_opts[arg])
                )
            setattr(self, arg, run_opts[arg])
        else:
            # If any arg from run_opt_defaults exist in hparams and
            # not in command line args "run_opts"
            if hparams is not None and arg in hparams:
                logger.info(
                    "Info: " + arg + " arg from hparam file is used"
                )
                setattr(self, arg, hparams[arg])
            else:
                setattr(self, arg, default)
    
    
    # Load modules to device
    if self.device == "cuda":
      torch.cuda.set_device(0)
    elif "cuda" in self.device:
      torch.cuda.set_device(int(self.device[-1]))
    self.modules = torch.nn.ModuleDict(modules).to(self.device)

    # Load hparams
    self.hparams = SimpleNamespace(**hparams)

    # Prepare iterating variables
    self.step = 0
    self.optimizer_step = 0
    self.avg_train_loss = 0.0

    # Add Class checkpointer for intra-epoch checkpoints
    if self.checkpointer is not None:
      self.checkpointer.add_recoverable("brain", self)

    self.train_sampler = None
  
  #---------------------------------------------------------#
  def fit( # Add on_fit_start
  #---------------------------------------------------------#
      self,
      epoch_counter,
      train_set,
      valid_set=None,
      progressbar=None,
      train_loader_kwargs={},
      valid_loader_kwargs={},
    ):
    '''
    Train step:
    Fit:
    & on_fit_start

      - _fit_train: 
      & on_stage_start 
       - fit_batch:
        - compute_forward
        - compute_loss
      & on_stage_end

      - _fit_valid:
      & on_stage_start
       - evaluate_batch
        - compute_forward
        - compute_loss
      & on_stage_end
    '''

    # Transfer Dataset to Dataloader
    
    train_set = self.make_dataloader(
          train_set, 
          stage=Stage.TRAIN,
          **train_loader_kwargs
      )
            
    valid_set = self.make_dataloader(
          valid_set,
          stage=Stage.VALID,
          ckpt_prefix=None,
          **valid_loader_kwargs,
      )

    # def on_fit_start
    self.on_fit_start()

    # Iterate epochs
    for epoch in epoch_counter:
      self._fit_train(train_set=train_set, epoch=epoch, enable=True)
      self._fit_valid(valid_set=valid_set, epoch=epoch, enable=True)


  #---------------------------------------------------------#
  def on_fit_start(self): # Belongs to Fit
  #---------------------------------------------------------#
    # Initialize optimizers
    if self.opt_class is not None:
      self.optimizer = self.opt_class(self.modules.parameters())

      if self.checkpointer is not None:
        self.checkpointer.add_recoverable("optimizer", self.optimizer)

    # Load latest checkpoint to resume training if interrupted
    if self.checkpointer is not None:
      self.checkpointer.recover_if_possible(
          device=torch.device(self.device)
      )

  #---------------------------------------------------------#
  def _fit_train(self, train_set, epoch, enable): # Add on_stage_start & on_stage_end
  #---------------------------------------------------------#
    # def on_stage_start
    self.on_stage_start(Stage.TRAIN, epoch)

    # Inital modules for train
    self.modules.train()

    # Reset nonfinite count to 0 each epoch
    self.nonfinite_count = 0

    # Sampler
    if self.train_sampler is not None and hasattr(
        self.train_sampler, "set_epoch"
    ):
        self.train_sampler.set_epoch(epoch)

    # Train through train_set

    with tqdm(
        train_set,
        initial=self.step,
        dynamic_ncols=True,
        disable=not enable,
    ) as t:
      for batch in t:
        self.step += 1
        loss = self.fit_batch(batch)
        self.avg_train_loss = self.update_average(
            loss, self.avg_train_loss
        )
        t.set_postfix(train_loss=self.avg_train_loss)

        # # Profile only if desired (steps allow the profiler to know when all is warmed up)
        # if self.profiler is not None:
        #     if self.profiler.record_steps:
        #         self.profiler.step()

        # # Debug mode only runs a few batches
        # if self.debug and self.step == self.debug_batches:
        #     break

        # if (
        #     self.checkpointer is not None
        #     and self.ckpt_interval_minutes > 0
        #     and time.time() - last_ckpt_time
        #     >= self.ckpt_interval_minutes * 60.0
        # ):
        #     # This should not use run_on_main, because that
        #     # includes a DDP barrier. That eventually leads to a
        #     # crash when the processes'
        #     # time.time() - last_ckpt_time differ and some
        #     # processes enter this block while others don't,
        #     # missing the barrier.
        #     self._save_intra_epoch_ckpt()
        #     last_ckpt_time = time.time()

    self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
    self.avg_train_loss = 0.0
    self.step = 0

  #---------------------------------------------------------#
  def fit_batch(self, batch):
  #---------------------------------------------------------#

    should_step = self.step % self.grad_accumulation_factor == 0
    
    # Def comput_forward
    outputs = self.compute_forward(batch, Stage.TRAIN)

    # Def compute_loss
    loss = self.compute_loss(outputs, batch, Stage.TRAIN)

    with self.no_sync(not should_step):
      (loss / self.grad_accumulation_factor).backward()
    
    # Parameter update
    if should_step:
      if self.check_gradients(loss):
        self.optimizer.step()
      self.optimizer.zero_grad()
      self.optimizer_step += 1

    return loss.detach().cpu()

  
  
  
  
  #---------------------------------------------------------#
  def _fit_valid(self, valid_set, epoch, enable):
  #---------------------------------------------------------#

    if valid_set is not None:

      # Def on_stage_start
      self.on_stage_start(Stage.VALID, epoch)

      # Inital modules for train
      self.modules.eval()

      avg_valid_loss = 0.0
      with torch.no_grad():
        for batch in tqdm(
          valid_set, dynamic_ncols=True, disable=not enable
        ):
          self.step += 1
          loss = self.evaluate_batch(batch, stage=Stage.VALID)
          avg_valid_loss = self.update_average(loss, avg_valid_loss)

        # Only run validation "on_stage_end" on main process
        self.step = 0
        run_on_main(
            self.on_stage_end,
            args=[Stage.VALID, avg_valid_loss, epoch],
        )
  
  #---------------------------------------------------------#
  def evaluate_batch(self, batch, stage):
  #---------------------------------------------------------#

    # Def compute_forward
    out = self.compute_forward(batch, stage=stage)

    # Def compute_loss
    loss = self.compute_loss(out, batch, stage=stage)

    return loss.detach().cpu()


  #---------------------------------------------------------#
  def make_dataloader(self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs):
  #---------------------------------------------------------#  
    # TRAIN stage is handled specially.
    if stage == Stage.TRAIN:
      loader_kwargs = self._train_loader_specifics(dataset, loader_kwargs)
    dataloader = ntu_diar.scr.dataio.dataloader.make_dataloader(
      dataset, **loader_kwargs
    )

    if (
      self.checkpointer is not None
      and ckpt_prefix is not None
    ):
      ckpt_key = ckpt_prefix + stage.name
      self.checkpointer.add_recoverable(ckpt_key, dataloader)
    return dataloader
  
  #---------------------------------------------------------#
  def _train_loader_specifics(self, dataset, loader_kwargs):
  #---------------------------------------------------------#
    sampler = loader_kwargs.get("sampler", None)
    shuffle = loader_kwargs.get("shuffle", False)

    if shuffle:
      if sampler is not None:
        raise ValueError(
            "Cannot specify both shuffle=True"
            "and a sampler in loader_kwargs"
        )
      sampler = ReproducibleRandomSampler(dataset)
      self.train_sampler = sampler
      loader_kwargs["sampler"] = self.train_sampler
      del loader_kwargs["shuffle"]

    return loader_kwargs

  @contextmanager
  #---------------------------------------------------------#
  def no_sync(self, use=True):
  #---------------------------------------------------------#
    """Copies pytorch's implementation for doing no_sync across all modules.
    Explanation: nn.module.no_sync() is a context manager for when one does
    not want to sync gradients, which happens when using both DDP and gradient accumulation.
    Speechbrain brain's class can contain multiple modules and calling no_sync on these
    individually would be very awkward, therefore this contextmanager exists.
    Arguments
    ---------
    use : bool
        If set to `False` will still sync gradients, useful to make behaviour togglable.
    """
    if use:
      old_values_list = []
      for module in self.modules.values():
        if not hasattr(module, "require_backward_grad_sync"):
          # if not using DDP
          break
        old_values_list.append(module.require_backward_grad_sync)
        module.require_backward_grad_sync = False
      yield
      for module, old_value in zip(
        self.modules.values(), old_values_list
    ):
        if not hasattr(module, "require_backward_grad_sync"):
          break
        module.require_backward_grad_sync = old_value
    else:
        yield
        
  #---------------------------------------------------------#
  def check_gradients(self, loss):
  #---------------------------------------------------------#
    """Check if gradients are finite and not too large.
    Automatically clips large gradients.
    Arguments
    ---------
    loss : tensor
        The loss tensor after ``backward()`` has been called but
        before the optimizers ``step()``.
    Returns
    -------
    bool
        Whether or not the optimizer step should be carried out.
    """
    if not torch.isfinite(loss):
        self.nonfinite_count += 1

        # Print helpful debug info
        logger.warn(f"Loss is {loss}.")
        for p in self.modules.parameters():
            if not torch.isfinite(p).all():
                logger.warn("Parameter is not finite: " + str(p))

        # Check if patience is exhausted
        if self.nonfinite_count > self.nonfinite_patience:
            raise ValueError(
                "Loss is not finite and patience is exhausted. "
                "To debug, wrap `fit()` with "
                "autograd's `detect_anomaly()`, e.g.\n\nwith "
                "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
            )
        else:
            logger.warn("Patience not yet exhausted, ignoring this batch.")
            return False

    # Clip gradient norm
    if self.max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.modules.parameters()), self.max_grad_norm
        )

    return True

  #---------------------------------------------------------#
  def update_average(self, loss, avg_loss):
  #---------------------------------------------------------#
        """Update running average of the loss.
        Arguments
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.
        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss


  @mark_as_saver
  def _save(self, path):
      save_dict = {
          "step": self.step,
          "avg_train_loss": self.avg_train_loss,
          "optimizer_step": self.optimizer_step,
      }
      with open(path, "w") as w:
          w.write(yaml.dump(save_dict))

  @mark_as_loader
  def _recover(self, path, end_of_epoch, device):
      del end_of_epoch
      del device
      with open(path) as f:
          save_dict = yaml.safe_load(f)
      self.step = save_dict["step"]
      self.avg_train_loss = save_dict["avg_train_loss"]
      # Ensure compatibility with checkpoints from before optimizer_step:
      if "optimizer_step" not in save_dict:
          clsname = self.__class__.__name__
          MSG = f"'optimizer_step' not found in {clsname} checkpoint."
          MSG += " Using the saved 'step' value (BACKWARDS COMPATIBILITY)"
          warnings.warn(MSG)
          self.optimizer_step = self.step
      else:
          self.optimizer_step = save_dict["optimizer_step"]



