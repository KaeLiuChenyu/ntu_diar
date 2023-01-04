#!/usr/bin/python3
import os
import speechbrain as sb
from speechbrain.utils.data_utils import download_file

if __name__ == "__main__":


    data_folder = "/path/to/Voxceleb"
    save_folder = "/path/to/save"

    verification_file = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"


    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa
  
    veri_file_path = os.path.join(
    save_folder, os.path.basename(verification_file)
    )
    download_file(verification_file, veri_file_path)
    
    sb.utils.distributed.run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": data_folder,
            "save_folder": save_folder,
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "seg_dur": 3.0,
            "skip_prep": False,
        },
    )
    
