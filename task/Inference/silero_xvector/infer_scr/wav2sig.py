import os
import subprocess
import torchaudio


def wav2sig(wav_file):
  recname = os.path.splitext(os.path.basename(wav_file))[0]
  if check_wav_16khz_mono(wav_file):
    signal, fs = torchaudio.load(wav_file)
  else:
    print("Converting audio file to single channel WAV using ffmpeg...")
    converted_wavfile = os.path.join(os.path.dirname(wav_file), '{}_converted.wav'.format(recname))
    convert_wavfile(wav_file, converted_wavfile)
    assert os.path.isfile(converted_wavfile), "Couldn't find converted wav file, failed for some reason"
    signal, fs = torchaudio.load(converted_wavfile)
  return recname, signal, fs


def check_wav_16khz_mono(wavfile):
    """
    Returns True if a wav file is 16khz and single channel
    """
    try:
        signal, fs = torchaudio.load(wavfile)

        mono = signal.shape[0] == 1
        freq = fs == 16000
        if mono and freq:
            return True
        else:
            return False
    except:
        return False


def convert_wavfile(wavfile, outfile):
    """
    Converts file to 16khz single channel mono wav
    """
    cmd = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(
        wavfile, outfile)
    subprocess.Popen(cmd, shell=True).wait()
    return outfile
