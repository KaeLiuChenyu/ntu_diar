import os
import sys
import yaml

from pathlib import Path
from pyannote.core.utils.helper import get_class_by_name

def main():


  print("Load params")
  with open(sys.argv[1], "r") as fin:
      config = yaml.load(fin, Loader=yaml.SafeLoader)

  pipeline_name = config["pipeline"]["name"]
  Klass = get_class_by_name(
          pipeline_name
      )

  print("Initial task")
  pipeline = Klass(**config["pipeline"].get("params", {}))
  pipeline.instantiate(config["params"])

  print("Start infering")
  out_annotation = pipeline.apply(config["input_audio"])

  with open(config["pre_rttm"], 'w') as f:
    out_annotation.write_rttm(f)

  

if __name__ == '__main__':

	main()
