import io
import soundfile
import librosa
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchdata
from typing import Optional

class ESC50TDM():
	def __init__(self, root_data_path:list[str], batch_size:Optional[int]=1, num_workers:Optional[int]=0):
		super().__init__()
		self.test_data_dir = root_data_path
		self.batch_size = batch_size
		self.num_workers = num_workers

	def to_sampels(self, data):
		return soundfile.read(io.BytesIO(data[1].read())), torch.tensor(int(data[0].split('/')[-1].split('-')[-1].split('.')[0]))

	def _create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files()\
			.shuffle()\
			.sharding_filter()\
			.open_files(mode='rb')\
			.map(self.to_sampels) \
			# .batch(self.batch_size) \
			# .map(self.collate_fn)
		
		return datapipe

	def setup(self, stage:Optional[str] = None):
		if len(self.test_data_dir)>0:
			self.test = self._create_pipeline(self.test_data_dir)

	def _dataloader(self, dataset):
		return DataLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

	def test_dataloader(self):
		self.test_dl = self._dataloader(self.test)
		return self.test_dl

	def collate_fn(self, batch):
		audios, labels = zip(*batch)
		labels = torch.stack(labels)

		mels = []
		for a in audios:
			mel = librosa.feature.melspectrogram(y=a[0], sr=a[1], fmin=0, fmax=8000, n_mels=80, n_fft=1024, win_length=1024, hop_length=512)
			mel = librosa.power_to_db(mel, ref=np.max)
			mel = (mel+40)/40
			mels.append(torch.tensor(mel, dtype=torch.float32).T)

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [1 for _ in labels]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()

		return labels, mels, text_lengths, mel_lengths

if __name__ == '__main__':
	import tqdm
	dataset = ESC50TDM(
		root_data_path=['/fsx/knoriy/code/CLASP/tmp/raw_datasets/ESC-50-master/audio/'],
		batch_size=2,
	)
	dataset.setup()

	for i in tqdm.tqdm(dataset.test_dataloader(), desc="train minibatch"):
		print(i)
		break