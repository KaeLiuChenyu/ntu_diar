import torch

from ntu_diar.scr.task.base_train_task import Task
from ntu_diar.scr.task.utils import Stage
from ntu_diar.scr.task.train.schedulers import update_learning_rate

class TrainTask(Task):
  "Class for speaker embedding training"

  #---------------------------------------------------------#
  def compute_forward(self, batch, stage):
  #---------------------------------------------------------#
    """
    Process training in following steps:

    Augment:
      augment_wavedrop
      augment_speed
      add_rev
      add_noise
      add_rev_noise
    Feature:
      Fbank
    modules:
      embedding_model
      classifier
      mean_var_norm

    """
    batch = batch.to(self.device)
    wavs, lens = batch.sig

    if stage == Stage.TRAIN:
      # Applying the augmentation pipeline
      wavs_aug_tot = []
      wavs_aug_tot.append(wavs)
      for count, augment in enumerate(self.hparams.augment_pipeline):

        # Apply augment
        wavs_aug = augment(wavs, lens)

        # Managing speed change
        if wavs_aug.shape[1] > wavs.shape[1]:
          wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
        else:
          zero_sig = torch.zeros_like(wavs)
          zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
          wavs_aug = zero_sig

        if self.hparams.concat_augment:
          wavs_aug_tot.append(wavs_aug)
        else:
          wavs = wavs_aug
          wavs_aug_tot[0] = wavs

      wavs = torch.cat(wavs_aug_tot, dim=0)
      self.n_augment = len(wavs_aug_tot)
      lens = torch.cat([lens] * self.n_augment)

    # Feature extraction and normalization
    feats = self.modules.compute_features(wavs)
    feats = self.modules.mean_var_norm(feats, lens)

    # Embeddings + speaker classifier
    embeddings = self.modules.embedding_model(feats)
    outputs = self.modules.classifier(embeddings)

    return outputs, lens

  
  #---------------------------------------------------------#
  def compute_loss(self, predictions, batch, stage):
  #---------------------------------------------------------#
    """
    Compute Loss using speaker-id as label
    """
    predictions, lens = predictions
    uttid = batch.id
    spkid, _ = batch.spk_id_encoded

    # Concatenate labels (due to data augmentation)
    if stage == Stage.TRAIN:
      spkid = torch.cat([spkid] * self.n_augment, dim=0)

    loss = self.hparams.compute_cost(predictions, spkid, lens)

    if stage == Stage.TRAIN and hasattr(
      self.hparams.lr_annealing, "on_batch_end"
    ):
      self.hparams.lr_annealing.on_batch_end(self.optimizer)

    if stage != Stage.TRAIN:
      self.error_metrics.append(uttid, predictions, spkid, lens)

    return loss
  
  
  #---------------------------------------------------------#
  def on_stage_start(self, stage, epoch=None):
  #---------------------------------------------------------#
    """
    Call at the beginning of each epoch
    """
    if stage != Stage.TRAIN:
      self.error_metrics = self.hparams.error_stats()


  #---------------------------------------------------------#
  def on_stage_end(self, stage, stage_loss, epoch=None):
  #---------------------------------------------------------#
    """
    Called at the end of each epoch
    """
    # Compute/store important stats
    stage_stats = {"loss": stage_loss}
    if stage == Stage.TRAIN:
      self.train_stats = stage_stats
    else:
      stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

    # Perform end-of-iteration things, like annealing, logging, etc.
    if stage == Stage.VALID:
      old_lr, new_lr = self.hparams.lr_annealing(epoch)
      update_learning_rate(self.optimizer, new_lr)

      self.hparams.train_logger.log_stats(
          stats_meta={"epoch": epoch, "lr": old_lr},
          train_stats=self.train_stats,
          valid_stats=stage_stats,
      )
      self.checkpointer.save_and_keep_only(
          meta={"ErrorRate": stage_stats["ErrorRate"]},
          min_keys=["ErrorRate"],
      )


