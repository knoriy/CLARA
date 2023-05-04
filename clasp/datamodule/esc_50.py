import io
import soundfile

import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence

from .base_tdm import BaseTDM
from .utils import get_log_melspec


class ESC50TDM(BaseTDM):
	def to_sampels(self, data):
		return soundfile.read(io.BytesIO(data[1].read())), torch.tensor(int(data[0].split('/')[-1].split('-')[-1].split('.')[0]))

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files()\
			.shuffle()\
			.sharding_filter()\
			.open_files(mode='rb')\
			.map(self.to_sampels) \
			# .batch(self.batch_size) \
			# .map(self.collate_fn)
		
		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)
		labels = torch.stack(labels)

		mels = [get_log_melspec(a[0], a[1]) for a in audios]

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [1 for _ in labels]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()

		return labels, mels, text_lengths, mel_lengths