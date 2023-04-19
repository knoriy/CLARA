import io
import soundfile
import librosa
import numpy as np

import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence

from base_tdm import BaseTDM


def gender_to_int(gender):
	if gender == 'males':
		return 0
	elif gender == 'females':
		return 1
	else:
		return 2

def map_gender_to_int(genders):
	return list(map(gender_to_int, genders))

class VoxCelebTDM(BaseTDM):
	def to_sampels(self, data):
		return soundfile.read(io.BytesIO(data[1].read())), data[0].split("/")[-2]

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.list_files(masks=["*.flac"], recursive=True)\
			.shuffle()\
			.sharding_filter()\
			.open_files(mode='rb')\
			.map(self.to_sampels) \
			# .batch(self.batch_size) \
			# .map(self.collate_fn)
		
		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)
		labels = map_gender_to_int(labels)

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
		labels = torch.tensor(labels)

		return labels, mels, text_lengths, mel_lengths

if __name__ == '__main__':
	dataset = VoxCelebTDM(
				test_urls=['/fsx/knoriy/raw_datasets/VoxCeleb_gender/'],
				batch_size = 100,
				num_workers=12,
			)
	dataset.setup()
	for batch in dataset.test_dataloader():
		print(batch[0])
		break