import re
import csv
import torch

#---------------------------------------------------------#
def load_data_csv(csv_path, replacements={}):
#---------------------------------------------------------#

  '''
  Read csv file in following format:
  | ID | duration | wav | start | stop | spk_id |
  '''

  with open(csv_path, newline="") as csvfile:
    result = {}
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    variable_finder = re.compile(r"\$([\w.]+)")
    for row in reader:
      # ID:
      try:
        data_id = row["ID"]
        del row["ID"]  # This is used as a key in result, instead.
      except KeyError:
        raise KeyError(
            "CSV has to have an 'ID' field, with unique ids"
            " for all data points"
        )
      if data_id in result:
        raise ValueError(f"Duplicate id: {data_id}")
      # Replacements:
      for key, value in row.items():
        try:
          row[key] = variable_finder.sub(
              lambda match: str(replacements[match[1]]), value
          )
        except KeyError:
          raise KeyError(
              f"The item {value} requires replacements "
              "which were not supplied."
          )
      # Duration:
      if "duration" in row:
        row["duration"] = float(row["duration"])
      result[data_id] = row
  return result



#---------------------------------------------------------#
def length_to_mask(length, max_len=None, dtype=None, device=None):
#---------------------------------------------------------#
    """Creates a binary mask for each sequence.
    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.
    Returns
    -------
    mask : tensor
        The binary mask.
    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask
