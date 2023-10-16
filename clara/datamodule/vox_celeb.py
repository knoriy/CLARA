import io
import soundfile
import torch
import torchdata
from torch.nn.utils.rnn import pad_sequence

from .base_tdm import BaseTDM
from .utils import get_log_melspec


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
	def to_samples(self, data):
		return soundfile.read(io.BytesIO(data[1].read())), data[0].split("/")[-2]

	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
		    .list_files_by_fsspec(masks=["*.tar"])\
			.shuffle()\
			.sharding_filter()\
			.open_files_by_fsspec(mode='rb')\
			.load_from_tar()\
			.map(self.to_samples) \
			# .batch(self.batch_size) \
			# .map(self.collate_fn)
		
		return datapipe

	def collate_fn(self, batch):
		audios, labels = zip(*batch)
		labels = map_gender_to_int(labels)

		mels = [get_log_melspec(a[0], a[1]) for a in audios]


		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [1 for _ in labels]
		text_lengths = torch.tensor(text_lengths)

		mels = pad_sequence(mels).permute(1,2,0).contiguous()
		labels = torch.tensor(labels)

		return labels, mels, text_lengths, mel_lengths