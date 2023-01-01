import logging

from torch.utils.data import DataLoader, IterableDataset

from .batch import PaddedBatch
from .dataset import DynamicItemDataset
from .sample import ReproducibleRandomSampler
from ntu_diar.scr.utils.checkpoints import (
  register_checkpoint_hooks,
  mark_as_saver,
  mark_as_loader,
)

#---------------------------------------------------------#
def make_dataloader(dataset, **loader_kwargs):
#---------------------------------------------------------#

  # PaddedBatch as default collation for DynamicItemDataset
  if "collate_fn" not in loader_kwargs and isinstance(
    dataset, DynamicItemDataset
  ):
    loader_kwargs["collate_fn"] = PaddedBatch

  # Reproducible random sampling
  if loader_kwargs.get("shuffle", False):
    if loader_kwargs.get("sampler") is not None:
      raise ValueError(
          "Cannot specify both shuffle=True and a "
          "sampler in loader_kwargs"
      )
    sampler = ReproducibleRandomSampler(dataset)
    loader_kwargs["sampler"] = sampler
    del loader_kwargs["shuffle"]
 
  # Create the loader
  if isinstance(dataset, IterableDataset):
    dataloader = DataLoader(dataset, **loader_kwargs)
  else:
    dataloader = SaveableDataLoader(dataset, **loader_kwargs)

  return dataloader



@register_checkpoint_hooks
#---------------------------------------------------------#
class SaveableDataLoader(DataLoader):
#---------------------------------------------------------#
    """
    A saveable version of the PyTorch DataLoader.
    """

    #---------------------------------------------------------#
    def __init__(self, *args, **kwargs):
    #---------------------------------------------------------#
      super().__init__(*args, **kwargs)

      if isinstance(self.dataset, IterableDataset):
        logging.warning(
            "SaveableDataLoader cannot save the position in an "
            "IterableDataset. Save the position on the dataset itself."
        )
      self.recovery_skip_to = None
      self.iterator = None

    #---------------------------------------------------------#
    def __iter__(self):
    #---------------------------------------------------------#
      iterator = super().__iter__()
      # Keep a reference to the iterator,
      # to be able to access the iterator._num_yielded value.
      # Keep a full reference (keeping the iterator alive)
      # rather than e.g. a weakref, as we may want to save a checkpoint
      # after the iterator has been exhausted, but before the full epoch has
      # ended (e.g. validation is still running)
      self.iterator = iterator
      return iterator

    @mark_as_saver
    #---------------------------------------------------------#
    def save(self, path):
    #---------------------------------------------------------#
      if isinstance(self.dataset, IterableDataset):
        logging.warning(
            "Warning again: a checkpoint was requested on "
            "SaveableDataLoader, but the dataset is an IterableDataset. "
            "Cannot save the position in an IterableDataset. Not raising "
            "an error; assuming that you know what you're doing."
        )
      if self.iterator is None:
        to_save = None
      else:
        to_save = self.iterator._num_yielded
      with open(path, "w") as fo:
        fo.write(str(to_save))

    @mark_as_loader
    #---------------------------------------------------------#
    def load(self, path, end_of_epoch, device=None):
    #---------------------------------------------------------#
      del device  # Unused here
      if self.iterator is not None:
        logging.debug(
            "SaveableDataLoader was requested to load a "
            "checkpoint, but the DataLoader has already been "
            "iterated. The DataLoader file will be ignored. "
            "This is normal in evaluation, when a checkpoint is "
            "loaded just to retrieve the best model."
        )
        return
      if end_of_epoch:
        # Don't load at end of epoch, as we actually want to start a fresh
        # epoch iteration next.
        return
      with open(path) as fi:
        saved = fi.read()
        if saved == str(None):
          # Saved at a point where e.g. an iterator did not yet exist.
          return
        else:
          self.recovery_skip_to = int(saved)