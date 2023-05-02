import io
import json
from typing import Optional
import soundfile
import librosa
import numpy as np

import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence

# from .base_tdm import BaseTDM
from datamodule.base_tdm import BaseTDM, group_by_filename
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer

_AGE_DICT = {"teens":0, "twenties": 1, "thirties": 2, "fourties": 3, "fifties": 4, "sixties": 5, "seventies": 6, "eighties": 7, "nineties": 8, "hundreds": 9}
_GENDER_DICT = {"male": 0,"female": 1}

class CommonVoiceTDM(BaseTDM):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokeniser = Tokeniser()

	def to_sampels(self, data):
		a, t = data
		audio, meta = soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))

		labels = {
				'text':meta["text"],
				'age':meta["original_data"]["age"],
				'gender':meta["original_data"]["gender"],
				'accent':meta["original_data"]['accent'],
				# 'language':t[0].split('/')[-3],
				}

		return audio, labels
	
	def to_keys(self, data):
		audio, labels  = data

		labels['age'] = _AGE_DICT.get(labels['age'], None)
		labels['gender'] = _GENDER_DICT.get(labels['gender'], None)

		return audio, labels 
	
	def filter_fn(self, sample):
		if _GENDER_DICT.get(sample[1]['gender'], None) and sample[1]['age']:
			return True
		return False

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files_by_fsspec(masks=["*.tar"])\
			.sharding_filter()\
			.filter(self.exclude_fn)\
			.open_files_by_fsspec(mode='rb')\
			.load_from_tar()\
			.groupby(group_by_filename, group_size=2, guaranteed_group_size=2)\
			.map(self.to_sampels)\
			.filter(self.filter_fn)\
			.map(self.to_keys)\
			# .batch(self.batch_size) \
			# .map(self.collate_fn)

		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)

		texts = [torch.tensor(self.tokeniser.encode(", ".join(l["text"]))) for l in labels]
		genders = torch.tensor([label['gender'] for label in labels])
		age = torch.tensor([label['age'] for label in labels])
		mels = []
		for a in audios:
			mel = librosa.feature.melspectrogram(y=a[0], sr=a[1], fmin=0, fmax=8000, n_mels=80, n_fft=1024, win_length=1024, hop_length=512)
			mel = librosa.power_to_db(mel, ref=np.max)
			mel = (mel+40)/40
			mels.append(torch.tensor(mel, dtype=torch.float32).T)

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [text.size(0) for text in texts]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()
		texts = pad_sequence(texts).T.contiguous()

		new_labels = {
			"texts": texts,
			"gender": genders,
			"age": age,
			}

		return new_labels, mels, text_lengths, mel_lengths
	