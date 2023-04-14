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
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamodule import MultilingualTDM

class PreCacheTDM(MultilingualTDM):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def to_sampels(self, data):
		a, t = data
		audio = soundfile.read(io.BytesIO(a[1].read()))
		text = json.loads(t[1].read().decode('utf-8'))
		file_path = a[0].split('.tar')
		file_path = file_path[0] + '.tar',  file_path[1]

		return audio, text, file_path

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
					audio_file_name = data['path_audio'].split('/')[-1].replace('.flac', '.pt')
					text_file_name = audio_file_name.replace('.pt', '.json')
					audio_data = data['mel']
					text_data = data['text']

					# Add audio file to tar
					buffer = io.BytesIO()
					torch.save(audio_data, buffer)
					buffer.seek(0)
					audio_info = tarfile.TarInfo(audio_file_name)
					audio_info.size = buffer.getbuffer().nbytes
					tar.addfile(audio_info, fileobj=buffer)

					# # Add text file to tar
					buffer = io.BytesIO()
					json_bytes = json.dumps(text_data, ensure_ascii=False).encode('utf-8')
					buffer.write(json_bytes)
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
		  				   "text":t, 
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
	dataset = PreCacheTDM(
			root_data_path='s3://s-laion-audio/webdataset_tar/', 
			dataset_list='/fsx/knoriy/code/CLASP/config/test_list.txt',
			exclude_list='/fsx/knoriy/code/CLASP/config/exclude_list.txt',
			batch_size = 64,
			num_workers=90,
		)

	dataset.setup()

	# pbar = tqdm.tqdm(dataset.train_dataloader(), desc="train")
	# for i in pbar:
	# 	pbar.set_description(f"Processing {i}")
	# 	pbar.update(1)

	pbar = tqdm.tqdm(dataset.val_dataloader(), desc="valid")
	for i in pbar:
		pbar.set_description(f"Processing {i}")
		pbar.update(1)

	# pbar = tqdm.tqdm(dataset.test_dataloader(), desc="test")
	# for i in pbar:
	# 	pbar.set_description(f"Processing {i}")
	# 	pbar.update(1)

	dataset.teardown('fit')

if __name__ == '__main__':
	main()