import torch

from types import SimpleNamespace
from hyperpyyaml import load_hyperpyyaml

from ntu_diar.scr.model.augment.audio_normalizer import AudioNormalizer


class Pretrain_model(torch.nn.Module):

    HPARAMS_NEEDED = []
    MODULES_NEEDED = []

    def __init__(
        self, hparams_local_path, model_local_path, overrides={}, run_opts=None, freeze_params=True
    ):
        super().__init__()
        # Arguments passed via the run opts dictionary. Set a limited
        # number of these, since some don't apply to inference.
        
        with open(hparams_local_path) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Pretraining:
        pretrainer = hparams["pretrainer"]
        pretrainer.set_collect_in(model_local_path)
        # Load on the CPU. Later the params can be moved elsewhere by specifying
        # run_opts={"device": ...}
        pretrainer.load_collected(device="cpu")
        
        
        run_opt_defaults = {
            "device": "cpu",
        }
        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Put modules on the right device, accessible with dot notation
        self.mods = torch.nn.ModuleDict(hparams["modules"])
        for module in self.mods.values():
            if module is not None:
                module.to(self.device)

        # Check MODULES_NEEDED and HPARAMS_NEEDED and
        # make hyperparams available with dot notation
        if self.HPARAMS_NEEDED and hparams is None:
            raise ValueError("Need to provide hparams dict.")
        if hparams is not None:
            # Also first check that all required params are found:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams:
                    raise ValueError(f"Need hparams['{hp}']")
            self.hparams = SimpleNamespace(**hparams)

        # Prepare modules for computation, e.g. jit
        self._prepare_modules(freeze_params)

        # Audio normalization
        self.audio_normalizer = hparams.get(
            "audio_normalizer", AudioNormalizer()
        )


    def _prepare_modules(self, freeze_params):
        """Prepare modules for computation, e.g. jit.
        Arguments
        ---------
        freeze_params : bool
            Whether to freeze the parameters and call ``eval()``.
        """

        # Make jit-able
        # self._compile_jit()
        # self._wrap_distributed()

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.mods.eval()
            for p in self.mods.parameters():
                p.requires_grad = False


class Embedding_model(Pretrain_model):

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, wavs, wav_lens=None, normalize=False):
  
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings