import os
import ntu_diar
import random
import torchaudio

from ntu_diar.scr.dataio.dataset import DynamicItemDataset
from ntu_diar.scr.dataio.encoder import CategoricalEncoder

def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    #---------------------------------------------------------#
    # 1. Declarations:
    #---------------------------------------------------------#
    train_data = DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    #---------------------------------------------------------#
    # 2. Define audio pipeline:
    #---------------------------------------------------------#
    @ntu_diar.scr.dataio.data_pipeline.takes("wav", "start", "stop", "duration")
    @ntu_diar.scr.dataio.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    ntu_diar.scr.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    #---------------------------------------------------------#
    # 3. Define text pipeline:
    #---------------------------------------------------------#
    @ntu_diar.scr.dataio.data_pipeline.takes("spk_id")
    @ntu_diar.scr.dataio.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    ntu_diar.scr.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    #---------------------------------------------------------#
    # 3. Fit encoder:
    # Load or compute the label encoder
    #---------------------------------------------------------#
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    #---------------------------------------------------------#
    # 4. Set output:
    #---------------------------------------------------------#
    ntu_diar.scr.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder