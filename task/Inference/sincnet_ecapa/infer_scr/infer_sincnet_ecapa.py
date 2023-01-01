import numpy as np
import math
import torch
import itertools

from typing import Callable, Optional, Union
from einops import rearrange

from pyannote.audio.utils.signal import binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.pipeline.parameter import ParamDict, Uniform
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio import Audio, Pipeline, Inference
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.utils import SpeakerDiarizationMixin, get_devices

from .sig2emb import sig2emb

class InferTask(SpeakerDiarizationMixin, Pipeline):
    def __init__(
        self,   
        segmentation_ckpt = None,
        embedding_ckpt = None,
        embedding_cfg = None,
        clustering = None,
        segmentation_onset: float = 0.58,
        segmentation_step: float = 0.1,
        segmentation_batch_size: int = 32,
        segmentataion_duration: float = 5.0,
        embedding_batch_size: int = 32,
        embedding_exclude_overlap = False,
        fs: int = 16000,
        out_channel: int = 512,
    ):
        super().__init__()

        seg_device, emb_device = get_devices(needs=2)

        self.segmentation_ckpt = segmentation_ckpt
        self.embedding_ckpt = embedding_ckpt
        self.embedding_cfg = embedding_cfg
        self.clustering = clustering
        self.segmentation_onset = segmentation_onset
        self.segmentation_step = segmentation_step
        self.segmentation_batch_size = segmentation_batch_size
        self.segmentataion_duration = segmentataion_duration
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap
        self.fs = fs
        self.out_channel = out_channel

        #Segmentation network Default: PyanNet
        model = PyanNet.from_pretrained(segmentation_ckpt)
        self.segmentation = ParamDict(
            threshold=Uniform(0.1, 0.9),
            min_duration_off=Uniform(0.0, 1.0),
        )

        model.to(seg_device)

        self._segmentation = Inference(
            model,
            #duration=model.specifications.duration,
            duration=self.segmentataion_duration,
            step=self.segmentation_step * model.specifications.duration,
            #skip_aggregation=True,
            batch_size=self.segmentation_batch_size,
        )


        #Frames info
        self._frames: SlidingWindow = self._segmentation.model.introspection.frames


        #Embedding network Default: ECAPA-TDNN
        self._embedding = sig2emb(
          embedding_cfg = self.embedding_cfg,
          embedding_ckpt = self.embedding_ckpt,
          device = emb_device,
          fs = self.fs,
        )

        #Clustering
        self.clustering = Clustering[self.clustering].value(metric="cosine")

        #Audio
        self._audio = Audio(sample_rate=self.fs, mono=True)


    def get_segmentations(self, file) -> SlidingWindowFeature:
        """Apply segmentation model
        Parameter
        ---------
        file : AudioFile
        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """
        
        segmentations: SlidingWindowFeature = self._segmentation(file)

        return segmentations

    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
    ):
        """Extract embeddings for each (chunk, speaker) pair
        Parameters
        ----------
        file : AudioFile
        binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Binarized segmentation.
        exclude_overlap : bool, optional
            Exclude overlapping speech regions when extracting embeddings.
            In case non-overlapping speech is too short, use the whole speech.
        Returns
        -------
        embeddings : (num_chunks, num_speakers, dimension) array
        """

        # when optimizing the hyper-parameters of this pipeline with frozen "segmentation_onset",
        # one can reuse the embeddings from the first trial, bringing a massive speed up to
        # the optimization process (and hence allowing to use a larger search space).

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, _ = binary_segmentations.data.shape
        
        self.counter = 0


        if exclude_overlap:
            # minimum number of samples needed to extract an embedding
            # (a lower number of samples would result in an error)
            min_num_samples = self._embedding.min_num_samples

            # corresponding minimum number of frames
            num_samples = duration * self.fs
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            # zero-out frames with overlapping speech
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )

        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data, binary_segmentations.sliding_window
            )


        def iter_waveform_and_mask():
            for (chunk, masks), (_, clean_masks) in zip(
                binary_segmentations, clean_segmentations
            ):
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray
                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=duration,
                    mode="pad",
                )
                # waveform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    # mask: (num_frames, ) np.ndarray

                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask
                    yield waveform[None], torch.from_numpy(used_mask)[None]
                    # w: (1, 1, num_samples) torch.Tensor
                    # m: (1, num_frames) torch.Tensor

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )


        embedding_batches = []

        for batch in batches:
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

            waveform_batch = torch.vstack(waveforms)
            # (batch_size, 1, num_samples) torch.Tensor

            mask_batch = torch.vstack(masks)
            # (batch_size, num_frames) torch.Tensor

            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch, out_channel=self.out_channel,
            )
            embedding_batches.append(embedding_batch)

        embedding_batches = np.vstack(embedding_batches)
        embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

        return embeddings

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build final discrete diarization out of clustered segmentation
        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Raw speaker segmentation.
        hard_clusters : (num_chunks, num_speakers) array
            Output of clustering step.
        count : (total_num_frames, 1) SlidingWindowFeature
            Instantaneous number of active speakers.
        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        num_clusters = np.max(hard_clusters) + 1
        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):

            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                # TODO: can we do better than this max here?
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )

        return self.to_diarization(clustered_segmentations, count)

    def apply(
        self,
        file: str = None,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization
        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Number of speakers, when known.
        min_speakers : int, optional
            Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Maximum number of speakers. Has no effect when `num_speakers` is provided.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)
        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        file = Audio.validate_file(file)
        
        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = self.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segmentations = self.get_segmentations(file)
        hook("segmentation", segmentations)
        #   shape: (num_chunks, num_frames, local_num_speakers)



        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            segmentations,
            onset=self.segmentation_onset,
            frames=self._frames,
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        # binarize segmentation
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations,
            onset=self.segmentation_onset,
            initial_state=False,
        )

        

        if self.clustering == "OracleClustering":
            embeddings = None
        else:
            embeddings = self.get_embeddings(
                file,
                binarized_segmentations,
                exclude_overlap=self.embedding_exclude_overlap,
            )
            hook("embeddings", embeddings)
            #   shape: (num_chunks, local_num_speakers, dimension)

        hard_clusters, _ = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            file=file,  # <== for oracle clustering
            frames=self._frames,  # <== for oracle clustering
        )
        #   hard_clusters: (num_chunks, num_speakers)
        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        hook("discrete_diarization", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=0.0,
        )
        diarization.uri = file["uri"]

        # when reference is available, use it to map hypothesized speakers
        # to reference speakers (this makes later error analysis easier
        # but does not modify the actual output of the diarization pipeline)
        if "annotation" in file and file["annotation"]:
            return self.optimal_mapping(file["annotation"], diarization)

        # when reference is not available, rename hypothesized speakers
        # to human-readable SPEAKER_00, SPEAKER_01, ...
        return diarization.rename_labels(
            {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }
        )

def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)