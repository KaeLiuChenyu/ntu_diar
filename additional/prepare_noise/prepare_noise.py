import os
import tqdm
import shutil
import pathlib
import torchaudio
import urllib.request


OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"
openrir_folder = "path/to/your/folder"
openrir_max_noise_len = 3.0


#---------------------------------------------------------#
def prepare_openrir(folder, reverb_csv, noise_csv, max_noise_len):
#---------------------------------------------------------#
    """Prepare the openrir dataset for adding reverb and noises.
    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    reverb_csv : str
        Filename for storing the prepared reverb csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    # Download and unpack if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")

    if not os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
        download_file(OPENRIR_URL, filepath, unpack=True)
    else:
        download_file(OPENRIR_URL, filepath)

    # Prepare reverb csv if necessary
    if not os.path.isfile(reverb_csv):
        rir_filelist = os.path.join(
            folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
        )
        _prepare_csv(folder, rir_filelist, reverb_csv)

    # Prepare noise csv if necessary
    if not os.path.isfile(noise_csv):
        noise_filelist = os.path.join(
            folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
        )
        _prepare_csv(folder, noise_filelist, noise_csv, max_noise_len)

#---------------------------------------------------------#
def _prepare_csv(folder, filelist, csv_file, max_length=None):
#---------------------------------------------------------#
 
    with open(csv_file, "w") as w:
        w.write("ID,duration,wav,wav_format,wav_opts\n\n")
        for line in open(filelist):

            # Read file for duration/channel info
            filename = os.path.join(folder, line.split()[-1])
            signal, rate = torchaudio.load(filename)

            # Ensure only one channel
            if signal.shape[0] > 1:
                signal = signal[0].unsqueeze(0)
                torchaudio.save(filename, signal, rate)

            ID, ext = os.path.basename(filename).split(".")
            duration = signal.shape[1] / rate

            # Handle long waveforms
            if max_length is not None and duration > max_length:
                # Delete old file
                os.remove(filename)
                for i in range(int(duration / max_length)):
                    start = int(max_length * i * rate)
                    stop = int(
                        min(max_length * (i + 1), duration) * rate
                    )
                    new_filename = (
                        filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                    )
                    torchaudio.save(
                        new_filename, signal[:, start:stop], rate
                    )
                    csv_row = (
                        f"{ID}_{i}",
                        str((stop - start) / rate),
                        new_filename,
                        ext,
                        "\n",
                    )
                    w.write(",".join(csv_row))
            else:
                w.write(
                    ",".join((ID, str(duration), filename, ext, "\n"))
                )

#---------------------------------------------------------#
def download_file(source, dest, unpack=False, dest_unpack=None, replace_existing=False):
#---------------------------------------------------------#

    class DownloadProgressBar(tqdm.tqdm):
        """ DownloadProgressBar class."""

        def update_to(self, b=1, bsize=1, tsize=None):
            """Needed to support multigpu training."""
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    # Create the destination directory if it doesn't exist
    dest_dir = pathlib.Path(dest).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "http" not in source:
        shutil.copyfile(source, dest)

    elif not os.path.isfile(dest) or (
        os.path.isfile(dest) and replace_existing
    ):
        print(f"Downloading {source} to {dest}")
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=source.split("/")[-1],
        ) as t:
            urllib.request.urlretrieve(
                source, filename=dest, reporthook=t.update_to
            )
    else:
        print(f"{dest} exists. Skipping download")

    # Unpack if necessary
    if unpack:
        if dest_unpack is None:
            dest_unpack = os.path.dirname(dest)
        print(f"Extracting {dest} to {dest_unpack}")
        shutil.unpack_archive(dest, dest_unpack)





#---------------------------------------------------------#
def main():
#---------------------------------------------------------#
    openrir_folder = "path/to/your/folder"
    open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
    open_noise_csv = os.path.join(openrir_folder, "noise.csv")
    prepare_openrir(
        openrir_folder,
        open_reverb_csv,
        open_noise_csv,
        openrir_max_noise_len,
            )

if __name__ == '__main__':

	main()
