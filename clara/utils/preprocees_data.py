import os
import io
import json
import librosa
import soundfile
import numpy as np
import tqdm
import tarfile
import torch
import torchdata

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamodule import MultilingualTDM

class PreCacheTDM(MultilingualTDM):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		######## Uncomment for audioset ########
		# self.train_data_dir = self.urls['balanced_train']
		# self.test_data_dir = self.urls['unbalanced_train']
		# self.valid_data_dir = self.urls['eval']
		######## Uncomment for audioset ########

	def to_sampels(self, data):
		a, t = data
		audio = soundfile.read(io.BytesIO(a[1].read()))
		text = json.loads(t[1].read().decode('utf-8'))
		file_path = a[0].split('.tar')
		file_path = file_path[0] + '.tar',  file_path[1]

		return audio, text, file_path

	def to_token(self, t):
		return torch.tensor(
				self.tokeniser_encode(
					', '.join(t['text']) if isinstance(t['text'], list) else t['text'], \
						'en' if 'original_data' not in t.keys() else t["original_data"]["language"] if "language" in t["original_data"].keys() else 'en'
				)
			)

	def create_tar_file(self, data_list):
		filedata = {}
		for item in data_list:
			filepath = item["path_tar"]
			if filepath not in filedata:
				filedata[filepath] = []
			filedata[filepath].append(item)

		out_tars_path = []
		for tar_path in filedata:
			local_tar_path = os.path.join("./tmp", *tar_path.split('/')[-3:])
			os.makedirs(os.path.dirname(local_tar_path), exist_ok=True)
			with tarfile.open(local_tar_path, mode='a') as tar:
				for data in filedata[tar_path]:
					audio_file_name = data['path_audio'].split('/')[-1].replace('.flac', '.pta')
					text_file_name = audio_file_name.replace('.pta', '.ptt')
					audio_data = data['mel']
					text_data = data['text']

					# Add audio file to tar
					buffer = io.BytesIO()
					torch.save(audio_data, buffer)
					buffer.seek(0)
					audio_info = tarfile.TarInfo(audio_file_name)
					audio_info.size = buffer.getbuffer().nbytes
					tar.addfile(audio_info, fileobj=buffer)

					# Add text file to tar
					buffer = io.BytesIO()
					torch.save(text_data, buffer)
					buffer.seek(0)
					text_info = tarfile.TarInfo(text_file_name)
					text_info.size = buffer.getbuffer().nbytes
					tar.addfile(text_info, fileobj=buffer)

			out_tars_path.append(local_tar_path)

		return out_tars_path
	
	def collate_fn(self, data):
		output = []

		for (a, t, path) in data:
			mel = librosa.feature.melspectrogram(y=a[0], sr=a[1], fmin=0, fmax=8000, n_mels=80, n_fft=1024, win_length=1024, hop_length=512)
			mel = librosa.power_to_db(mel, ref=np.max)
			mel = (mel+40)/40
			output.append({"mel":torch.tensor(mel, dtype=torch.float32), 
		  				   "text":self.to_token(t), 
						   "path_tar":path[0],
						   "path_audio":path[1]})

		return output
	def _create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.sharding_filter()\
			.open_files_by_fsspec(mode='rb')\
			.load_from_tar() \
			.batch(2) \
			.map(self.to_sampels) \
			.batch(self.batch_size) \
			.map(self.collate_fn) \
			.map(self.create_tar_file) \
		
		return datapipe

	def train_dataloader(self):
		self.train_dl = self._dataloader2(self.train)
		return self.train_dl

	def val_dataloader(self):
		self.val_dl = self._dataloader2(self.valid)
		return self.val_dl

	def test_dataloader(self):
		self.test_dl = self._dataloader2(self.test)
		return self.test_dl


def main():
	batch_size = 64
	samples_per_tar = 512
	num_workers = 32

	dataset = PreCacheTDM(
			root_data_path='s3://s-laion-audio/webdataset_tar/', 
			dataset_list='./config/test_list.txt',
			# exclude_list='./config/exclude_list.txt',
			batch_size = batch_size,
			num_workers=num_workers,
			cache_path='./tmp/tensored_list.json', 
			train_valid_test=['balanced_train', 'eval', 'unbalanced_train'],
		)

	dataset.setup()

	total = len(dataset.train_data_dir)*(samples_per_tar//batch_size)
	pbar = tqdm.tqdm(dataset.train_dataloader(), desc="train", total=total)
	for i in pbar:
		pbar.set_description(f"Processing {i}")
		pbar.update(1)
	print("train_dataloader end")

	# total = len(dataset.valid_data_dir)*(samples_per_tar//batch_size)
	# pbar = tqdm.tqdm(dataset.val_dataloader(), desc="valid", total=total)
	# for i in pbar:
	# 	pbar.set_description(f"Processing {i}")
	# 	pbar.update(1)
	# print("val_dataloader end")

	# total = len(dataset.test_data_dir)*(samples_per_tar//batch_size)
	# pbar = tqdm.tqdm(dataset.test_dataloader(), desc="test", total=total)
	# for i in pbar:
	# 	pbar.set_description(f"Processing {i}")
	# 	pbar.update(1)
	# print("test_dataloader end")

	# dataset.teardown('fit')

if __name__ == '__main__':
	main()
	upload_to_s3_cmd = "srun --partition=cpu64 --exclusive --comment clap --job-name=aws s3 cp --recursive ./tmp/tensored/ s3://s-laion/knoriy/tensored/ --dryrun"
	# srun --partition=cpu32 --exclusive --comment clap --job-name=aws /fsx/home-knoriy/miniconda3/envs/clasp_2/bin/python clasp/utils/preprocees_data.py
	print("end")