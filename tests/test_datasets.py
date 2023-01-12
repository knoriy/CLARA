import webdataset as wds
from torch.utils.data import DataLoader
import webdataset as wds

__LONG_AUDIO:int = 30
__SHORT_AUDIO:int = 0.4
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

def test_datasets():
	data_dir = "pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/epidemic_sound_effects/train/{0..1}.tar -"
	batch_size = 64

	pipeline = []
	pipeline.extend([
		wds.SimpleShardList(data_dir),
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
	dataloader = DataLoader(dataset, batch_size=None, num_workers=6, persistent_workers=True)

	messages = filter(lambda x: len(x) > 0, dataloader)

	for message in messages:
		for m in message:
			print(f"\t{m}")

	assert len(list(messages)) != 0, "At least one error found, please see above list and corresponding error."

if __name__ == '__main__':
	test_datasets()

