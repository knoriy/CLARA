import io
import json
import soundfile

import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence

# from .base_tdm import BaseTDM
from datamodule.base_tdm import BaseTDM
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer

from .utils import get_log_melspec, group_by_filename

class AudioSetTDM(BaseTDM):
	def __init__(self, classes, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokeniser = Tokeniser()
		self.classes = classes

	def to_sampels(self, data):
		a, t = data
		return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def to_keys(self, data):
		audio, labels  = data

		classes = [self.classes.get(l) for l in labels["original_data"]["class_names"]]

		new_labels = {
			"text": labels["text"],
			"labels": classes,
		}

		return audio, new_labels 

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files_by_fsspec(masks=["*.tar"])\
			.sharding_filter()\
			.filter(self.exclude_fn)\
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
		# WARMING: ONLY USING THE FIRST LABEL
		classes = [torch.tensor(l['labels'][0]) for l in labels]
		texts = [torch.tensor(self.tokeniser.encode(", ".join(l["text"]))) for l in labels]

		mels = [get_log_melspec(a[0], a[1]) for a in audios]

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [text.size(0) for text in texts]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()
		texts = pad_sequence(texts).T.contiguous()

		new_labels = {
			"texts": texts,
			"classes": classes,
			}

		return new_labels, mels, text_lengths, mel_lengths