import sys
sys.path.append('/fsx/knoriy/code/CLASP/clasp')

import webdataset as wds
import torch
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
from text.whisper.normalizers import EnglishTextNormalizer
from text.tokeniser import Tokeniser # from text.whisper.tokenizer import get_tokenizer

import audio as Audio

stft_fn = Audio.stft.MelSpecPipeline()
cleaner = EnglishTextNormalizer()
tokenizer = Tokeniser()
def collate_fn(data):
		raw_audios, raw_texts, __url = data

		# mels = [stft_fn(audio[0][0]).T for audio in raw_audios]
		# texts = [torch.tensor(
		# 	tokenizer.encode(
		# 		cleaner(', '.join(text['text']) if isinstance(text['text'], list) else text['text']), 
		# 		language = 'en' if 'original_data' not in text.keys() else text["original_data"]["language"] if "language" in text["original_data"].keys() else 'en'
		# 		)) for text in raw_texts]

		for audio, text , url in zip(raw_audios, raw_texts, __url):
			# Audio
			if audio[0].shape[1] == 0:
				print(f"File with 0 length; {audio[0].shape}, URL: {url}")
			if audio[0].shape[1] > 48000*20:
				print(f"Big file found; {audio[0].shape[1]} = {audio[0].shape[1]/48000}, URL: {url}")
			if audio[0].shape[1] < 48000*1:
				print(f"File found with less than 1 sec; {audio[0].shape[1]} = {audio[0].shape[1]/48000}, URL: {url}")
			
			# Text
			text_len = len(', '.join(text['text']) if isinstance(text['text'], list) else text['text'])
			if text_len > 1024:
				print(f"Large string sound: length is {text_len}, URL: {url}")
			if text_len == 0:
				print(f"Empty string found: URL: {url}")

		# mel_lengths = [mel.shape[0] for mel in mels]
		# mel_lengths = torch.tensor(mel_lengths)
		# text_lengths = [text.shape[0] for text in texts]
		# text_lengths = torch.tensor(text_lengths)

		# texts = pad_sequence(texts).T
		# mels = pad_sequence(mels).permute(1,2,0)

		# return texts, mels, text_lengths, mel_lengths
		return torch.rand((1,1))

def test_datasets():
	# data_dir = "pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/common_voice/train/{0..1695}.tar -"
	data_dir = "pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/common_voice/train/227.tar -"
	batch_size = 64

	pipeline = []
	pipeline.extend([
		wds.SimpleShardList(data_dir),
		# wds.detshuffle(),
		# wds.split_by_node,
		wds.split_by_worker
		])

	pipeline.extend([wds.tarfile_to_samples()])

	pipeline.extend([
		wds.decode(wds.torch_audio),
		wds.to_tuple("flac", "json", "__url__"),
		wds.batched(batch_size),
		wds.map(collate_fn)
		])


	dataset = wds.DataPipeline(*pipeline)
	dataloader = DataLoader(dataset, batch_size=None, num_workers=6, persistent_workers=True)

	for i, _ in enumerate(dataloader):
		pass

if __name__ == '__main__':
	print("starting test")
	test_datasets()