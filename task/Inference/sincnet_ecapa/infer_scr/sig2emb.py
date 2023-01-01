import torch
import numpy as np
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from ntu_diar.scr.task.infer.embedding import Embedding_model


try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property


class sig2emb:

    def __init__(
        self,
        embedding_cfg: str = None,
        embedding_ckpt: str = None,
        device: torch.device = None,
        fs: int = 16000,
    ):
        super().__init__()
        self.embedding_cfg = embedding_cfg
        self.embedding_ckpt = embedding_ckpt
        self.device = device
        self.fs = fs

        self.encoder = Embedding_model(
          self.embedding_cfg,
          self.embedding_ckpt,
          run_opts={"device": self.device},
        )

    @cached_property
    def min_num_samples(self) -> int:

        lower, upper = 2, round(0.5 * self.fs)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                _ = self.encoder(
                    torch.randn(1, middle).to(self.device)
                )
                upper = middle
            except RuntimeError:
                lower = middle

            middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: torch.Tensor = None, out_channel = 512,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional
        Returns
        -------
        embeddings : (batch_size, dimension)
        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)

        else:

            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)

            imasks = imasks > 0.5

            signals = pad_sequence(
                [waveform[imask] for waveform, imask in zip(waveforms, imasks)],
                batch_first=True,
            )

            wav_lens = imasks.sum(dim=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.NAN * np.zeros((batch_size, out_channel))

        too_short = wav_lens < self.min_num_samples
        wav_lens = wav_lens / max_len
        wav_lens[too_short] = 1.0

        
        embeddings = (
            self.encoder(signals, wav_lens=wav_lens)
            .squeeze(dim=1)
            .cpu()
            .numpy()
        )
        

        embeddings[too_short.cpu().numpy()] = np.NAN

        return embeddings