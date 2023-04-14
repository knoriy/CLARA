import io
import json
import torch
import torchdata

from torch.nn.utils.rnn import pad_sequence

from .td_datamodule import MultilingualTDM

class TensoredTDM(MultilingualTDM):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def to_sampels(self, data):
		a, t = data
		return torch.load(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
	
	def _create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.shuffle()\
			.sharding_filter()\
			.open_files_by_fsspec(mode='rb')\
			.load_from_tar() \
			.batch(2) \
			.shuffle(buffer_size=self.batch_size)\
			.map(self.to_sampels) \
			.batch(self.batch_size) \
			.map(self.collate_fn)
		
		return datapipe
	
	def collate_fn(self, data):
		def to_token(t):
			return torch.tensor(
					self.tokeniser_encode(
						', '.join(t['text']) if isinstance(t['text'], list) else t['text'], \
							'en' if 'original_data' not in t.keys() else t["original_data"]["language"] if "language" in t["original_data"].keys() else 'en'
					)
				)
	
		mels = [a.T for (a, _) in data]
		texts = [to_token(t) for (_, t) in data]

		mel_lengths = [mel.shape[0] for mel in mels]
		mel_lengths = torch.tensor(mel_lengths)
		text_lengths = [text.shape[0] for text in texts]
		text_lengths = torch.tensor(text_lengths)

		texts = pad_sequence(texts).T.contiguous()
		mels = pad_sequence(mels).permute(1,2,0).contiguous()

		return texts, mels, text_lengths, mel_lengths


if __name__ == '__main__':
	import tqdm
	dataset = TensoredTDM(
		root_data_path='s3://s-laion/knoriy/tensored/', 
		dataset_list='/fsx/knoriy/code/CLASP/config/test_list.txt',
		exclude_list='/fsx/knoriy/code/CLASP/config/exclude_list.txt',
		batch_size = 64,
		num_workers=0,
	)
	dataset.setup()

	for i in tqdm.tqdm(dataset.train, desc="train minibatch"):
		print(i)
		break