import contextlib

from torch.utils.data import Dataset

from .dataio import load_data_csv
from .data_pipeline import DataPipeline

class DynamicItemDataset(Dataset):

    #---------------------------------------------------------#
    def __init__(self, data, dynamic_items=[], output_keys=[],):
    #---------------------------------------------------------#

      self.data = data
      self.data_ids = list(self.data.keys())
      static_keys = list(self.data[self.data_ids[0]].keys())
      if "id" in static_keys:
        raise ValueError("The key 'id' is reserved for the data point id.")
      else:
        static_keys.append("id")
      self.pipeline = DataPipeline(static_keys, dynamic_items)
      self.set_output_keys(output_keys)


    #---------------------------------------------------------#
    def __len__(self):
    #---------------------------------------------------------#
      return len(self.data_ids)


    #---------------------------------------------------------#
    def __getitem__(self, index):
    #---------------------------------------------------------#
      data_id = self.data_ids[index]
      data_point = self.data[data_id]
      return self.pipeline.compute_outputs({"id": data_id, **data_point})


    #---------------------------------------------------------#
    def add_dynamic_item(self, func, takes=None, provides=None):
    #---------------------------------------------------------#
      self.pipeline.add_dynamic_item(func, takes, provides)


    #---------------------------------------------------------#
    def set_output_keys(self, keys):
    #---------------------------------------------------------#
      self.pipeline.set_output_keys(keys)


    @contextlib.contextmanager
    #---------------------------------------------------------#
    def output_keys_as(self, keys):
    #---------------------------------------------------------#
        saved_output = self.pipeline.output_mapping
        self.pipeline.set_output_keys(keys)
        yield self
        self.pipeline.set_output_keys(saved_output)

    @classmethod
    #---------------------------------------------------------#
    def from_csv(cls, csv_path, replacements={}, dynamic_items=[], output_keys=[]):
    #---------------------------------------------------------#
      """
      Load a data prep CSV file and create a Dataset based on it.
      """
      data = load_data_csv(csv_path, replacements)
      return cls(data, dynamic_items, output_keys)


    #---------------------------------------------------------#
    def _filtered_sorted_ids(
        self,
        key_min_value={},
        key_max_value={},
        key_test={},
        sort_key=None,
        reverse=False,
        select_n=None,
    ):
    #---------------------------------------------------------#
        """Returns a list of data ids, fulfilling the sorting and filtering."""

        def combined_filter(computed):
            """Applies filter."""
            for key, limit in key_min_value.items():
                # NOTE: docstring promises >= so using that.
                # Mathematically could also use < for nicer syntax, but
                # maybe with some super special weird edge case some one can
                # depend on the >= operator
                if computed[key] >= limit:
                    continue
                return False
            for key, limit in key_max_value.items():
                if computed[key] <= limit:
                    continue
                return False
            for key, func in key_test.items():
                if bool(func(computed[key])):
                    continue
                return False
            return True

        temp_keys = (
            set(key_min_value.keys())
            | set(key_max_value.keys())
            | set(key_test.keys())
            | set([] if sort_key is None else [sort_key])
        )
        filtered_ids = []
        with self.output_keys_as(temp_keys):
            for i, data_id in enumerate(self.data_ids):
                if select_n is not None and len(filtered_ids) == select_n:
                    break
                data_point = self.data[data_id]
                data_point["id"] = data_id
                computed = self.pipeline.compute_outputs(data_point)
                if combined_filter(computed):
                    if sort_key is not None:
                        # Add (main sorting index, current index, data_id)
                        # So that we maintain current sorting and don't compare
                        # data_id values ever.
                        filtered_ids.append((computed[sort_key], i, data_id))
                    else:
                        filtered_ids.append(data_id)
        if sort_key is not None:
            filtered_sorted_ids = [
                tup[2] for tup in sorted(filtered_ids, reverse=reverse)
            ]
        else:
            filtered_sorted_ids = filtered_ids
        return filtered_sorted_ids

#---------------------------------------------------------#
def add_dynamic_item(datasets, func, takes=None, provides=None):
#---------------------------------------------------------#
    """Helper for adding the same item to multiple datasets."""
    for dataset in datasets:
        dataset.add_dynamic_item(func, takes, provides)

#---------------------------------------------------------#
def set_output_keys(datasets, output_keys):
#---------------------------------------------------------#
    """Helper for setting the same item to multiple datasets."""
    for dataset in datasets:
        dataset.set_output_keys(output_keys)