import re
import csv
import collections
import torchaudio

from ntu_diar.scr.dataio.dataset import DynamicItemDataset

CSVItem = collections.namedtuple("CSVItem", ["data", "format", "opts"])
CSVItem.__doc__ = """The Legacy Extended CSV Data item triplet"""

ITEM_POSTFIX = "_data"

#---------------------------------------------------------#
class ExtendedCSVDataset(DynamicItemDataset):
#---------------------------------------------------------#
    """Extended CSV compatibility for DynamicItemDataset.
    CSV must have an 'ID' and 'duration' fields.
    The rest of the fields come in triplets:
    ``<name>, <name>_format, <name>_opts``
    These add a <name>_sb_data item in the dict. Additionally, a basic
    DynamicItem (see DynamicItemDataset) is created, which loads the _sb_data
    item.
    Bash-like string replacements with $to_replace are supported.
    NOTE
    ----
    Mapping from legacy interface:
    - csv_file -> csvpath
    - sentence_sorting -> sorting, and "random" is not supported, use e.g.
      ``make_dataloader(..., shuffle = (sorting=="random"))``
    - avoid_if_shorter_than -> min_duration
    - avoid_if_longer_than -> max_duration
    - csv_read -> output_keys, and if you want IDs add "id" as key
    Arguments
    ---------
    csvpath : str, path
        Path to extended CSV.
    replacements : dict
        Used for Bash-like $-prefixed substitution,
        e.g. ``{"data_folder": "/home/speechbrain/data"}``, which would
        transform `$data_folder/utt1.wav` into `/home/speechbain/data/utt1.wav`
    sorting : {"original", "ascending", "descending"}
        Keep CSV order, or sort ascending or descending by duration.
    min_duration : float, int
        Minimum duration in seconds. Discards other entries.
    max_duration : float, int
        Maximum duration in seconds. Discards other entries.
    dynamic_items : list
        Configuration for extra dynamic items produced when fetching an
        example. List of DynamicItems or dicts with keys::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides
        NOTE: A dynamic item is automatically added for each CSV data-triplet
    output_keys : list, None
        The list of output keys to produce. You can refer to the names of the
        CSV data-triplets. E.G. if the CSV has: wav,wav_format,wav_opts,
        then the Dataset has a dynamic item output available with key ``"wav"``
        NOTE: If None, read all existing.
    """

    def __init__(
        self,
        csvpath,
        replacements={},
        sorting="original",
        min_duration=0,
        max_duration=36000,
        dynamic_items=[],
        output_keys=[],
    ):
        if sorting not in ["original", "ascending", "descending"]:
            clsname = self.__class__.__name__
            raise ValueError(f"{clsname} doesn't support {sorting} sorting")
        # Load the CSV, init class
        data, di_to_add, data_names = load_sb_extended_csv(
            csvpath, replacements
        )
        super().__init__(data, dynamic_items, output_keys)
        self.pipeline.add_dynamic_items(di_to_add)
        # Handle filtering, sorting:
        reverse = False
        sort_key = None
        if sorting == "ascending" or "descending":
            sort_key = "duration"
        if sorting == "descending":
            reverse = True
        filtered_sorted_ids = self._filtered_sorted_ids(
            key_min_value={"duration": min_duration},
            key_max_value={"duration": max_duration},
            sort_key=sort_key,
            reverse=reverse,
        )
        self.data_ids = filtered_sorted_ids
        # Handle None output_keys (differently than Base)
        if not output_keys:
            self.set_output_keys(data_names)

#---------------------------------------------------------#
def load_sb_extended_csv(csv_path, replacements={}):
#---------------------------------------------------------#
    """Loads SB Extended CSV and formats string values.
    Uses the SpeechBrain Extended CSV data format, where the
    CSV must have an 'ID' and 'duration' fields.
    The rest of the fields come in triplets:
    ``<name>, <name>_format, <name>_opts``.
    These add a <name>_sb_data item in the dict. Additionally, a
    basic DynamicItem (see DynamicItemDataset) is created, which
    loads the _sb_data item.
    Bash-like string replacements with $to_replace are supported.
    This format has its restriction, but they allow some tasks to
    have loading specified by the CSV.
    Arguments
    ----------
    csv_path : str
        Path to the CSV file.
    replacements : dict
        Optional dict:
        e.g. ``{"data_folder": "/home/speechbrain/data"}``
        This is used to recursively format all string values in the data.
    Returns
    -------
    dict
        CSV data with replacements applied.
    list
        List of DynamicItems to add in DynamicItemDataset.
    """
    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        if not reader.fieldnames[0] == "ID":
            raise KeyError(
                "CSV has to have an 'ID' field, with unique ids"
                " for all data points"
            )
        if not reader.fieldnames[1] == "duration":
            raise KeyError(
                "CSV has to have an 'duration' field, "
                "with the length of the data point in seconds."
            )
        if not len(reader.fieldnames[2:]) % 3 == 0:
            raise ValueError(
                "All named fields must have 3 entries: "
                "<name>, <name>_format, <name>_opts"
            )
        names = reader.fieldnames[2::3]
        for row in reader:
            # Make a triplet for each name
            data_point = {}
            # ID:
            data_id = row["ID"]
            del row["ID"]  # This is used as a key in result, instead.
            # Duration:
            data_point["duration"] = float(row["duration"])
            del row["duration"]  # This is handled specially.
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            # Only need to run these in the actual data,
            # not in _opts, _format
            for key, value in list(row.items())[::3]:
                try:
                    row[key] = variable_finder.sub(
                        lambda match: replacements[match[1]], value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            for i, name in enumerate(names):
                triplet = CSVItem(*list(row.values())[i * 3 : i * 3 + 3])
                data_point[name + ITEM_POSTFIX] = triplet
            result[data_id] = data_point
        # Make a DynamicItem for each CSV entry
        # _read_csv_item delegates reading to further
        dynamic_items_to_add = []
        for name in names:
            di = {
                "func": _read_csv_item,
                "takes": name + ITEM_POSTFIX,
                "provides": name,
            }
            dynamic_items_to_add.append(di)
        return result, dynamic_items_to_add, names

#---------------------------------------------------------#
def _read_csv_item(item):
#---------------------------------------------------------#
    """Reads the different formats supported in SB Extended CSV.
    Delegates to the relevant functions.
    """
    opts = _parse_csv_item_opts(item.opts)
    audio, _ = torchaudio.load(item.data)
    return audio.squeeze(0)

  
#---------------------------------------------------------#
def _parse_csv_item_opts(entry):
#---------------------------------------------------------#
    """Parse the _opts field in a SB Extended CSV item."""
    # Accepting even slightly weirdly formatted entries:
    entry = entry.strip()
    if len(entry) == 0:
        return {}
    opts = {}
    for opt in entry.split(" "):
        opt_name, opt_val = opt.split(":")
        opts[opt_name] = opt_val
    return opts