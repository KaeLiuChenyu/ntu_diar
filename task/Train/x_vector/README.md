Train x-vector
===========================


****

# Folders for train x-vector
## **config**
Main script:
1. ntu_diar/task/Train/x_vector
   1. config
   2. data
   3. train_scr
   4. utils
   5. run.py

>i.config
>>config files

>ii.data
>>.csv files containing data information

| ID           | duration          | wav | start | stop | skp_id |
|--------------|-------------------|-----|-----|-----|-----|
| utterance ID | Duration of audio | File path | Start of the utterance in the audio | Start of the utterance in the audio | Speaker of the utterance |  


>iii.train_scr
>> Scripts needed for training x-vector

>iv.utils
>>Utilities

>v.run.py
>>Main script. 


## **Usage**

```
python ./run.py config/train_x_vectors.yaml --data_folder=data
```
