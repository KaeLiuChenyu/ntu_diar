NTU Diarization
===========================


****

# Folders for ntu_diar
## **scr**
Main script:
1. scr
   1. dataio
   2. model
   3. task
   4. utils

>i.dataio
>>Includes dataset processing scripts for creating Datasets and Dataloader.

>ii.model
>>The components that make up the Diarization model. This includes feature extraction network, data augmentation, clustering, etc.

>iii.task
>>Includes scripts to implement tasks. Including training task and inference task.

>iv.utils
>>Utilities



## **task**
Training & Inference task:
1. task
   1. Inference
   2. Train


>i.Inference
>>Includes several Inference scripts for diarization models.

>ii.Train
>>Includes several Training scripts for diarization models.

## **module**
Module file:
1. module
   1. pre_train
   2. train


>i.pre_train
>>Includes pre-trained model files

>ii.Train
>>Includes checkpoint files and log files during training. 


## **additional**
The attached scripts, such as the preparation of data sets:


# Usage

## **Train X-vector**
Please check [ntu_diar/task/Train/x_vector](https://github.com/KaeLiuChenyu/ntu_diar/tree/main/task/Train/x_vector)

## **Inference**
Please check [ntu_diar/task/inference/silero_xvector](https://github.com/KaeLiuChenyu/ntu_diar/tree/main/task/Inference/silero_xvector)
