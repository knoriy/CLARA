import os
import json
import soundfile
import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence

# from .base_tdm import BaseTDM
from .base_tdm import BaseTDM
from .utils import get_log_melspec

KEYS = ("id","utterance","description","emotion","date_created","status","gender","age","level","audio_recording","user_id")

def parse_csv(data):
	data = data[1].split('|')
	data = {k:v for k,v in zip(KEYS, data)}
	return data

class EMNSTDM(BaseTDM):
	def __init__(self, classes:str, *args, **kwargs):
		super().__init__(*args, **kwargs)
		with open(classes, 'r') as f:
			self.emotions = json.load(f)
	def to_sampels(self, data):
		path, label = data['audio_recording'], data['emotion']
		path = os.path.basename(path).replace('.webm', '.flac')
		path = os.path.join('/fsx/knoriy/raw_datasets/EMNS/cleaned_webm', path)

		return soundfile.read(path), label.strip().lower()

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.open_files()\
			.sharding_filter()\
			.shuffle()\
			.readlines(skip_lines=1)\
			.map(parse_csv)\
			.map(self.to_sampels)\
			.shuffle()\
			# .batch(self.batch_size) \
			# .map(self.collate_fn)

		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)

		labels = torch.tensor([self.emotions[l] for l in labels])

		mels = [get_log_melspec(a[0], a[1]) for a in audios]

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [1 for _ in labels]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()

		return labels, mels, text_lengths, mel_lengths