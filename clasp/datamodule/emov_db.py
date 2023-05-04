import io
import json
import soundfile

import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence

# from .base_tdm import BaseTDM
from datamodule.base_tdm import BaseTDM, group_by_filename
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer

from .utils import get_log_melspec


_EMOTION_DICT = {"angry": 0,"disgusted": 1,"amused": 2,"sleepy": 3,"neutral": 4}
_GENDER_DICT = {"male": 0,"male": 0, "female": 1, "females": 1}

class EmovDBTDM(BaseTDM):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokeniser = Tokeniser()

	def to_sampels(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def to_keys(self, data):
		audio, labels  = data

		new_labels = {}

		new_labels['text'] = labels['text']
		new_labels['emotion'] = _EMOTION_DICT.get(labels['original_data']['emotion'].lower())
		new_labels['gender'] = _GENDER_DICT.get(labels['original_data']['gender'])

		return audio, new_labels 
	
	def filter_fn(self, sample):
		if _GENDER_DICT.get(sample[1]['gender'], None) and sample[1]['age']:
			return True
		return False

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files_by_fsspec(masks=["*.tar"])\
			.filter(self.exclude_fn)\
			.sharding_filter()\
			.open_files_by_fsspec(mode='rb')\
			.load_from_tar()\
			.groupby(group_by_filename, group_size=2, guaranteed_group_size=2)\
			.map(self.to_sampels)\
			.map(self.to_keys)\
			# .batch(self.batch_size) \
			# .map(self.collate_fn)

		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)

		texts = [torch.tensor(self.tokeniser.encode(", ".join(l["text"]))) for l in labels]
		genders = torch.tensor([label['gender'] for label in labels])
		emotion = torch.tensor([label['emotion'] for label in labels])

		mels = [get_log_melspec(a[0], a[1]) for a in audios]
		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		
		text_lengths = [text.size(0) for text in texts]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()
		texts = pad_sequence(texts).T.contiguous()

		new_labels = {
			"texts": texts,
			"gender": genders,
			"emotion": emotion,
			}

		return new_labels, mels, text_lengths, mel_lengths