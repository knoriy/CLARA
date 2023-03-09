import io
import json
import torch
import torchdata
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import librosa
import soundfile
import numpy as np

from text.whisper.normalizers import EnglishTextNormalizer
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer


cleaner = EnglishTextNormalizer()
tokenizer = Tokeniser() # self.tokenizer = get_tokenizer(True)

def to_sampels(data):
    a, t = data
    return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))

dp_s3_urls = torchdata.datapipes.iter.IterableWrapper(["s3://s-laion-audio/webdataset_tar/EmoV_DB/test/0.tar"])\
        .list_files_by_s3()\
        .shuffle()\
        .sharding_filter()

tars = torchdata.datapipes.iter.S3FileLoader(dp_s3_urls)\
    .load_from_tar() \
    .batch(2) \
    .map(to_sampels) \
    # .groupby(lambda x: os.path.basename(x[0]).split(".")[0],group_size=2, guaranteed_group_size=2) \


def tokeniser_encode(text:str, lanuage:str='en'):
    return tokenizer.encode(cleaner(text), language=lanuage)

def collate_fn(data):
    mels = []
    texts = []

    for (a, t) in data:
        mel = librosa.feature.melspectrogram(y=a[0], sr=a[1], fmin=0, fmax=8000, n_mels=80, n_fft=1024, win_length=1024, hop_length=512)
        mel = librosa.power_to_db(mel, ref=np.max)
        mels.append(torch.tensor(mel).T)

        texts.append(
            torch.tensor(
                tokeniser_encode(
                    ', '.join(t['text']) if isinstance(t['text'], list) else t['text'], \
                        'en' if 'original_data' not in t.keys() else t["original_data"]["language"] if "language" in t["original_data"].keys() else 'en'
                )
            ) 
        )

    mel_lengths = [mel.shape[0] for mel in mels]
    mel_lengths = torch.tensor(mel_lengths)
    text_lengths = [text.shape[0] for text in texts]
    text_lengths = torch.tensor(text_lengths)


    texts = pad_sequence(texts).T.contiguous()
    mels = pad_sequence(mels).permute(1,2,0).contiguous()

    return texts, mels, text_lengths, mel_lengths


dataloader = DataLoader(tars, 1024, collate_fn=collate_fn, num_workers=16)

for d in dataloader:
    texts, mels, text_lengths, mel_lengths = d
    print(texts.shape, mels.shape)
    break
