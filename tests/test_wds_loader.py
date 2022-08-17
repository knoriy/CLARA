import torchaudio
import pathlib


def test_wds_dataloader():
    path = pathlib.Path('/home/knoriy/fsx/yuchen/processed_freesound/test/')
    glob = path.glob('*.flac')


    for i in glob:
        print()
        torchaudio.load(i)