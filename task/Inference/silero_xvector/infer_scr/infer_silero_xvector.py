import os
import sys
import torch
import umap

import numpy as np

from .wav2sig import wav2sig
from .sig2emb import sig2emb
from .pre2rttm import pre2rttm

class InferTask:

    def __init__(self,hparams,):

      # Clustering
      self.cluster = hparams["clustering"]

      # VAD
      self.vad_model = hparams["vad_model"]

      # Embedding
      self.embed_model = hparams["embedding_model"]

      self.run_opts = {"device": "cuda:0"} if torch.cuda.is_available() else {
          "device": "cpu"}


    def infer(self,
          wav_file,
          outfile,
          num_speakers=2,
          ):

        # Load wav
        print("Load wav")
        recname, signal, fs = wav2sig(wav_file)

        # Infer
        print("Diarizing ...")
        cluster_labels, segments = self.sig2labels(signal, 
                                fs, 
                                outfile,
                                )
        
        #rttm
        pre2rttm(cluster_labels,
            segments, 
            fs, 
            recname, 
            outfile,
            )

    def sig2labels(self, signal, fs, outfile):

      # VAD
      speech_ts = self.vad_model.infer(signal[0])
      print('Splitting by silence found {} utterances'.format(len(speech_ts)))
      assert len(speech_ts) >= 1, "Couldn't find any speech during VAD"

      # Embeddings
      embeds, segments = sig2emb(self.embed_model,
                      signal,
                      fs, 
                      speech_ts
                      )
      
      #Clustering
      cluster_labels = self.cluster(embeds, 
                      n_clusters=2,
                      threshold=None,
                      enhance_sim=True
                      )

      return cluster_labels, segments

          