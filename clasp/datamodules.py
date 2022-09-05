import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torchaudio

import pytorch_lightning as pl
import webdataset as wds

from typing import Optional

from text import text_to_sequence
import audio as Audio


class WebdatasetDataModule(pl.LightningDataModule):
	def __init__(self, train_data_dir:str, test_data_dir:str, valid_data_dir:str, epochs:int=1, batch_size:int = 32, num_workers:int=0, audio_backend:str=None):
		super().__init__()
		# if not audio_backend:
		# torchaudio.set_audio_backend('soundfile') # Forching backend to soundfile, due to known bug in torch audio (https://github.com/pytorch/audio/issues/2356)

		self.train_data_dir = train_data_dir
		self.test_data_dir = test_data_dir
		self.valid_data_dir = valid_data_dir

		self.epochs = epochs
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.stft_fn =Audio.stft.TacotronSTFT(
			filter_length=1024,
			hop_length=256,
			win_length=1024,
			n_mel_channels=80,
			sampling_rate=48000,
			mel_fmin=0,
			mel_fmax=8000,
		)

	def setup(self, stage:Optional[str] = None):
		if len(self.train_data_dir)>0:
			self.train =  wds.WebDataset(self.train_data_dir, resampled=True).decode(wds.torch_audio).to_tuple("flac", "json").batched(self.batch_size).map(self.collate_fn)
		if len(self.test_data_dir)>0:
			self.test =  wds.WebDataset(self.test_data_dir, resampled=True).decode(wds.torch_audio).to_tuple("flac", "json").batched(self.batch_size).map(self.collate_fn)
		if len(self.valid_data_dir)>0:
			self.valid =  wds.WebDataset(self.valid_data_dir, resampled=True).decode(wds.torch_audio).to_tuple("flac", "json").batched(self.batch_size).map(self.collate_fn)

	def train_dataloader(self):
		if self.train:
			return wds.WebLoader(self.train, num_workers=self.num_workers)

	def val_dataloader(self):
		if self.valid:
			return wds.WebLoader(self.valid, num_workers=self.num_workers)

	def test_dataloader(self):
		if self.test:
			return wds.WebLoader(self.test, num_workers=self.num_workers)

	# 	return text, mel
	def collate_fn(self, data):
		raw_audios, raw_texts = data
		# # split values into own varable
		mels = [Audio.tools.get_mel_from_wav(audio[0][0].numpy(), self.stft_fn)[0] for audio in raw_audios]
		mels = [torch.tensor(mel).T for mel in mels]
		texts = [torch.tensor(text_to_sequence(text['text'][0], ["english_cleaners"])) for text in raw_texts]

		texts = pad_sequence(texts).T
		mels = pad_sequence(mels).permute(1,2,0)

		return texts, mels
	
if __name__ == '__main__':
	import tqdm
	from utils.get_wds_urls import get_tar_path_s3
	dataset_names = [
		# '130000_MIDI_SONGS', #PASS
		# 'CREMA-D', #PASS
		# 'Clotho', #PASS
		# 'CoVoST_2',#PASS
		# 'EmoV_DB', #PASS
		# 'FSD50K', #PASS
		# 'Urbansound8K', #PASS
		# 'audiocaps', #PASS
		# 'audioset', #PASS
		# 'audiostock', #PASS
		# 'cambridge_dictionary', #PASS
		'esc50', #PASS
		# 'free_to_use_sounds', #PASS
		# 'freesound', #PASS
		# 'midi50k', #PASS
		# 'paramount_motion', #PASS
		# 'sonniss_game_effects', #PASS
		# 'wesoundeffects', #PASS
		# 'FMA_updated', #FAIL
		# 'LJSpeech', #FAIL
		# 'VocalSketch', #FAIL
		# 'YT_dataset', #FAIL
		# 'clotho_mixed', #FAIL
		# 'ravdess', #FAIL
		# # 'tmp_eval',
		# 'BBCSoundEffects', #FAIL
	]
	urls = get_tar_path_s3(
		's-laion-audio/webdataset_tar/', 
		['train', 'test', 'valid'],
		dataset_names,
		cache_path='/tmp/url_cache.json',
		recache=True,
		)
	for url in urls.values():
		print(len(url))
	dataset = WebdatasetDataModule(	train_data_dir = urls['train'], 
									test_data_dir =urls['test'], 
									valid_data_dir = urls['valid'], 
									batch_size = 512,
									num_workers=6)

	dataset.setup()

	for i in tqdm.tqdm(zip(dataset.train_dataloader())):
		pass
	# for i in tqdm.tqdm(dataset.val_dataloader()):
	# 	pass
	# for i in tqdm.tqdm(dataset.test_dataloader()):
	# 	pass