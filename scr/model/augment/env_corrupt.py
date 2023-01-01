import os
import torch

from .processing.speech_augmentation import (
  AddReverb,
  AddBabble,
  AddNoise,
  )

#---------------------------------------------------------#
class EnvCorrupt(torch.nn.Module):
#---------------------------------------------------------#
    """Environmental Corruptions for speech signals: noise, reverb, babble.
    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble added.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    openrir_folder : str
        If provided, download and prepare openrir to this location. The
        reverberation csv and noise csv will come from here unless overridden
        by the ``reverb_csv`` or ``noise_csv`` arguments.
    openrir_max_noise_len : float
        The maximum length in seconds for a noise segment from openrir. Only
        takes effect if ``openrir_folder`` is used for noises. Cuts longer
        noises into segments equal to or less than this length.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data.
    noise_num_workers : int
        Number of workers to use for loading noises.
    babble_speaker_count : int
        Number of speakers to use for babble. Must be less than batch size.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    rir_scale_factor : float
        It compresses or dilates the given impulse response.
        If ``0 < rir_scale_factor < 1``, the impulse response is compressed
        (less reverb), while if ``rir_scale_factor > 1`` it is dilated
        (more reverb).
    reverb_sample_rate : int
        Sample rate of input audio signals (rirs) used for reverberation.
    noise_sample_rate: int
        Sample rate of input audio signals used for adding noise.
    clean_sample_rate: int
        Sample rate of original (clean) audio signals.
    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> corrupter = EnvCorrupt(babble_speaker_count=9)
    >>> feats = corrupter(inputs, torch.ones(10))
    """

    def __init__(
        self,
        reverb_prob=1.0,
        babble_prob=1.0,
        noise_prob=1.0,
        openrir_folder=None,
        openrir_max_noise_len=None,
        reverb_csv=None,
        noise_csv=None,
        noise_num_workers=0,
        babble_speaker_count=0,
        babble_snr_low=0,
        babble_snr_high=0,
        noise_snr_low=0,
        noise_snr_high=0,
        rir_scale_factor=1.0,
        reverb_sample_rate=16000,
        noise_sample_rate=16000,
        clean_sample_rate=16000,
    ):
        super().__init__()

        # Download and prepare openrir
        if openrir_folder and (not reverb_csv or not noise_csv):

            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")

            # Specify filepath and sample rate if not specified already
            if not reverb_csv:
                reverb_csv = open_reverb_csv
                reverb_sample_rate = 16000

            if not noise_csv:
                noise_csv = open_noise_csv
                noise_sample_rate = 16000

        # Initialize corrupters
        if reverb_csv is not None and reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob,
                csv_file=reverb_csv,
                rir_scale_factor=rir_scale_factor,
                reverb_sample_rate=reverb_sample_rate,
                clean_sample_rate=clean_sample_rate,
            )

        if babble_speaker_count > 0 and babble_prob > 0.0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if noise_csv is not None and noise_prob > 0.0:
            self.add_noise = AddNoise(
                mix_prob=noise_prob,
                csv_file=noise_csv,
                num_workers=noise_num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
                noise_sample_rate=noise_sample_rate,
                clean_sample_rate=clean_sample_rate,
            )

    def forward(self, waveforms, lengths):
        """Returns the distorted waveforms.
        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort.
        """
        # Augmentation
        with torch.no_grad():
            if hasattr(self, "add_reverb"):
                try:
                    waveforms = self.add_reverb(waveforms, lengths)
                except Exception:
                    pass
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms, lengths)
            if hasattr(self, "add_noise"):
                waveforms = self.add_noise(waveforms, lengths)

        return waveforms

