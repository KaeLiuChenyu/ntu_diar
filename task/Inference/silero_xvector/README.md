Infer silero-xvector
===========================


****

# Folders for Infer silero-xvector
Main script:
1. ntu_diar/task/Inference/x_vector
   1. config
   2. data
   3. infer_scr
   4. run.py

>i.config
>>config files

>ii.data
>>Audio for testing


>iii.infer_scr
>> Scripts needed for infering silero-xvector

>v.run.py
>>Main script. 


## **Usage**
Step 1(Modify config.yaml):
```
vad_folder: path/to/vad/model/file
embedding_params: path/to/embedding/config/file
embedding_folder: path/to/embedding/model/file
input_audio: path/to/test/audo
pre_rttm: path/to/output/rttm
```
Step 2:
```
python ./run.py config/config.yaml
```

