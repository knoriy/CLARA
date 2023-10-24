import sys
sys.path.append('/fsx/knoriy/CLARA/clara')
import argparse

import webdataset as wds
from torch.utils.data import DataLoader

from utils.get_wds_urls import get_tar_path_s3

__LONG_AUDIO:int = 30
__SHORT_AUDIO:int = 0.2
__LONG_TEXT:int = 1024

def collate_fn(data):
	keys, urls, raw_audios, raw_texts = data

	messages = []
	for key, url, audio, text in zip(keys, urls, raw_audios, raw_texts):
		try:
		# Audio
			_sample_rate = audio[1]
			if audio[0].shape[1] == 0:
				messages.append(f"0 length: {audio[0].shape}, URL: {url}, file: {key}, file: {key.split('/')[-1]}")
			if audio[0].shape[1] > _sample_rate*__LONG_AUDIO:
				messages.append(f"Long audio: {(audio[0].shape[1]/_sample_rate):.2f}s, URL: {url}, file: {key.split('/')[-1]}")
			if audio[0].shape[1] < _sample_rate*__SHORT_AUDIO:
				messages.append(f"Short audio: {(audio[0].shape[1]/_sample_rate):.2f}s, URL: {url}, file: {key.split('/')[-1]}")

			text_len = len(', '.join(text['text']) if isinstance(text['text'], list) else text['text'])
			if text_len > __LONG_TEXT:
				messages.append(f"Long string: {text_len}, URL: {url}, file: {key.split('/')[-1]}")
			if text_len == 0:
				messages.append(f"Empty string: {text_len}, URL: {url}, file: {key.split('/')[-1]}")
		except:
			messages.append(f"Unknown Error: URL: {url}, file: {key.split('/')[-1]}, audio: {audio[0].shape} text: {text}")

	return messages

def test_datasets(
		base_s3_path:str,
		dataset_names:list[str]=[], 
		splits:list[str]=['train', 'test', 'valid'], 
		):
	exclude = []
	urls = get_tar_path_s3(
			base_s3_path		= base_s3_path, 
			train_valid_test	= splits,
			dataset_names		= dataset_names, 
			exclude				= exclude,
			)
	batch_size = 64

	total_fails = 0

	for key in urls:
		print(f'#'*50)
		print(f'# Checking: {key} urls')
		print(f'#'*50)
		pipeline = []
		pipeline.extend([
			wds.SimpleShardList(urls[key]),
			wds.split_by_worker
			])

		pipeline.extend([wds.tarfile_to_samples()])

		pipeline.extend([
			wds.decode(wds.torch_audio),
			wds.to_tuple('__key__', '__url__', 'flac', 'json'),
			wds.batched(batch_size),
			wds.map(collate_fn)
			])

		dataset = wds.DataPipeline(*pipeline)
		dataloader = DataLoader(dataset, batch_size=None, num_workers=96, persistent_workers=True)

		messages = filter(lambda x: len(x) > 0, dataloader)

		for message in messages:
			for m in message:
				print(f"\t{m}")

		total_fails += len(list(messages))

	assert total_fails == 0, f"{total_fails} error found, please see above log."

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_names', nargs='+', type=str, required=True)
	parser.add_argument('--base_s3_path', type=str, default='laion-west-audio/webdataset_tar/')

	args = parser.parse_args()

	test_datasets(base_s3_path=args.base_s3_path, dataset_names=args.dataset_names)

