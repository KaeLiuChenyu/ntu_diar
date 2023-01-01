import re
import torch
import collections

#---------------------------------------------------------#
def batch_pad_right(tensors: list, mode="constant", value=0):
#---------------------------------------------------------#
  """
  Given a list of torch tensors it batches them together by padding to the right
  on each dimension in order to get same length for all.
  Parameters
  ----------
  tensors : list
      List of tensor we wish to pad together.
  mode : str
      Padding mode see torch.nn.functional.pad documentation.
  value : float
      Padding value see torch.nn.functional.pad documentation.
  Returns
  -------
  tensor : torch.Tensor
      Padded tensor.
  valid_vals : list
      List containing proportion for each dimension of original, non-padded values.
  """

  if not len(tensors):
    raise IndexError("Tensors list must not be empty")

  if len(tensors) == 1:
    # if there is only one tensor in the batch we simply unsqueeze it.
    return tensors[0].unsqueeze(0), torch.tensor([1.0])

  if not (
    any(
        [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
    )
  ):
    raise IndexError("All tensors must have same number of dimensions")

  # FIXME we limit the support here: we allow padding of only the first dimension
  # need to remove this when feat extraction is updated to handle multichannel.
  max_shape = []
  for dim in range(tensors[0].ndim):
    if dim != 0:
      if not all(
        [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
      ):
        raise EnvironmentError(
            "Tensors should have same dimensions except for the first one"
        )
    max_shape.append(max([x.shape[dim] for x in tensors]))

  batched = []
  valid = []
  for t in tensors:
    # for each tensor we apply pad_right_to
    padded, valid_percent = pad_right_to(
      t, max_shape, mode=mode, value=value
    )
    batched.append(padded)
    valid.append(valid_percent[0])

  batched = torch.stack(batched)

  return batched, torch.tensor(valid)

#---------------------------------------------------------#
def pad_right_to(tensor, target_shape, mode="constant", value=0,):
#---------------------------------------------------------#
  """
  This function takes a torch tensor of arbitrary shape and pads it to target
  shape by appending values on the right.
  Parameters
  ----------
  tensor : input torch tensor
      Input tensor whose dimension we need to pad.
  target_shape : (list, tuple)
      Target shape we want for the target tensor its len must be equal to tensor.ndim
  mode : str
      Pad mode, please refer to torch.nn.functional.pad documentation.
  value : float
      Pad value, please refer to torch.nn.functional.pad documentation.
  Returns
  -------
  tensor : torch.Tensor
      Padded tensor.
  valid_vals : list
      List containing proportion for each dimension of original, non-padded values.
  """
  assert len(target_shape) == tensor.ndim
  pads = []  # this contains the abs length of the padding for each dimension.
  valid_vals = []  # this contains the relative lengths for each dimension.
  i = len(target_shape) - 1  # iterating over target_shape ndims
  j = 0
  while i >= 0:
    assert (
      target_shape[i] >= tensor.shape[i]
    ), "Target shape must be >= original shape for every dim"
    pads.extend([0, target_shape[i] - tensor.shape[i]])
    valid_vals.append(tensor.shape[j] / target_shape[j])
    i -= 1
    j += 1

  tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

  return tensor, valid_vals

np_str_obj_array_pattern = re.compile(r"[SaUO]")

#---------------------------------------------------------#
def mod_default_collate(batch):
#---------------------------------------------------------#
  """Makes a tensor from list of batch values.
  Note that this doesn't need to zip(*) values together
  as PaddedBatch connects them already (by key).
  """
  elem = batch[0]
  elem_type = type(elem)
  if isinstance(elem, torch.Tensor):
    out = None
    try:
      if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
      return torch.stack(batch, 0, out=out)
    except RuntimeError:  # Unequal size:
        return batch
  elif (
    elem_type.__module__ == "numpy"
    and elem_type.__name__ != "str_"
    and elem_type.__name__ != "string_"
  ):
    try:
      if (
        elem_type.__name__ == "ndarray"
        or elem_type.__name__ == "memmap"
      ):
          # array of string classes and object
        if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
          return batch
        return mod_default_collate([torch.as_tensor(b) for b in batch])
      elif elem.shape == ():  # scalars
        return torch.as_tensor(batch)
    except RuntimeError:  # Unequal size
      return batch
  elif isinstance(elem, float):
    return torch.tensor(batch, dtype=torch.float64)
  elif isinstance(elem, int):
    return torch.tensor(batch)
  else:
    return batch

#---------------------------------------------------------#
def recursive_to(data, *args, **kwargs):
#---------------------------------------------------------#
    """Moves data to device, or other type, and handles containers.
    Very similar to torch.utils.data._utils.pin_memory.pin_memory,
    but applies .to() instead.
    """
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, collections.abc.Mapping):
        return {
            k: recursive_to(sample, *args, **kwargs)
            for k, sample in data.items()
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            *(recursive_to(sample, *args, **kwargs) for sample in data)
        )
    elif isinstance(data, collections.abc.Sequence):
        return [recursive_to(sample, *args, **kwargs) for sample in data]
    elif hasattr(data, "to"):
        return data.to(*args, **kwargs)
    # What should be done with unknown data?
    # For now, just return as they are
    else:
        return data