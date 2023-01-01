import torch
import numpy as np
from tqdm.autonotebook import tqdm

def sig2emb(embed_model,
       signal,
       fs, 
       speech_ts, 
       window=1.5, 
       period=0.75
       ):
        """
        Takes signal and VAD output (speech_ts) and produces windowed embeddings
        returns: embeddings, segment info
        """
        all_embeds = []
        all_segments = []


        for utt in tqdm(speech_ts, desc='Utterances', position=0):
            start = utt['start']
            end = utt['end']

            utt_signal = signal[:, start:end]
            utt_embeds, utt_segments = windowed_embeds(embed_model,
                                    utt_signal,
                                    fs,
                                    window,
                                    period)
            all_embeds.append(utt_embeds)
            all_segments.append(utt_segments + start)

        all_embeds = np.concatenate(all_embeds, axis=0)
        all_segments = np.concatenate(all_segments, axis=0)
        return all_embeds, all_segments

def windowed_embeds(embed_model,
           signal, 
           fs, 
           window, 
           period=0.75
           ):
        """
        Calculates embeddings for windows across the signal
        window: length of the window, in seconds
        period: jump of the window, in seconds
        returns: embeddings, segment info
        """
        len_window = int(window * fs)
        len_period = int(period * fs)
        len_signal = signal.shape[1]

        # Get the windowed segments
        segments = []
        start = 0
        while start + len_window < len_signal:
            segments.append([start, start+len_window])
            start += len_period

        segments.append([start, len_signal-1])
        embeds = []

        with torch.no_grad():
            for i, j in segments:
                signal_seg = signal[:, i:j]
                seg_embed = embed_model(signal_seg)
                embeds.append(seg_embed.squeeze(0).squeeze(0).cpu().numpy())

        embeds = np.array(embeds)
        return embeds, np.array(segments)