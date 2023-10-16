import io
import torch
from torch.nn.utils.rnn import pad_sequence

import torchdata
from typing import Optional

from .td_datamodule import MultilingualTDM
from .utils import group_by_filename, Boto3FileOpenerIterDataPipe


class TensoredTDM(MultilingualTDM):
	def __init__(self, connection_timeout:Optional[int]=0, read_timeout:Optional[int]=0, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.connection_timeout = connection_timeout
		self.read_timeout = read_timeout

	def to_samples(self, data):
		a, t = data
		return torch.load(io.BytesIO(a[1].read())), torch.load(io.BytesIO(t[1].read()))
	
	def _create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.shuffle()\
			.sharding_filter()\
			
		datapipe = Boto3FileOpenerIterDataPipe(datapipe, mode='rb')\
			.load_from_tar() \
			.groupby(group_by_filename, group_size=2, guaranteed_group_size=2)\
			.shuffle(buffer_size=100)\
			.map(self.to_samples) \
			# .batch(self.batch_size) \
			# .map(self.collate_fn)
		
		return datapipe
	
	def collate_fn(self, data):
		mels, texts = zip(*data)
		mels = [mel.T for mel in mels]

		mel_lengths = [mel.T.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [text.shape[0] for text in texts]
		text_lengths = torch.tensor(text_lengths)

		texts = pad_sequence(texts).T.contiguous()
		mels = pad_sequence(mels).permute(1,2,0).contiguous()

		return texts, mels, text_lengths, mel_lengths

	def train_dataloader(self):
		self.train_dl = self._dataloader(self.train)
		return self.train_dl

	def val_dataloader(self):
		self.val_dl = self._dataloader(self.valid)
		return self.val_dl

	def test_dataloader(self):
		self.test_dl = self._dataloader(self.test)
		return self.test_dl

if __name__ == '__main__':
	import tqdm
	dataset = TensoredTDM(
		root_data_path='s3://s-laion/knoriy/tensored/', 
		dataset_list='./config/dataset_list.txt',
		exclude_list='./config/exclude_list.txt',
		batch_size = 64,
		num_workers=12,
	)
	dataset.setup()

	for i in tqdm.tqdm(dataset.train, desc="train minibatch"):
		breakpoint()
		print(i)
		break