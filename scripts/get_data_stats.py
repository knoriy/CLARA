import sys
sys.path.append("/fsx/knoriy/CLASP/clasp/")

import os
import io
import logging
import torchdata
import json
import soundfile
import tarfile

from typing import Optional
from torchdata.datapipes.iter import FSSpecFileOpener

from datamodule.td_datamodule import BaseTDM
from utils import get_s3_paths, get_local_paths, get_lists
from datamodule.utils import Boto3FileOpenerIterDataPipe as Boto3FileOpener
from datamodule.utils import group_by_filename

pl_logger = logging.getLogger('pytorch_lightning')

class GetStatsTDM(BaseTDM):
	def __init__(self, 
			root_data_path:str,#'s-laion-audio/webdataset_tar/' or '/fsx/knoriy/processed_datasets/', 
			dataset_list:str,
			exclude_list:Optional[str]=None,
			cache_path:Optional[str]='',
			use_cache:Optional[bool]=False,
			recache:Optional[bool]=False,
			train_valid_test:Optional[list]=['train', 'valid', 'test'],
			*args,**kwargs,
		):
		super().__init__(*args,**kwargs)
		self.use_cache = use_cache

		exclude = []
		if exclude_list:
			exclude = get_lists(exclude_list)

		dataset_names = get_lists(dataset_list)
		nl = "\n\t"
		pl_logger.info(f"Dataset names:{nl}{nl.join(map(str ,dataset_names))}")

		dataset_names_intersection = set(dataset_names).intersection(exclude)
		if dataset_names_intersection:
			raise Warning(f'Found similary dataset names in datasets and excluded dataset: {dataset_names_intersection}')
		
		if root_data_path.startswith('s3://'):
			self.is_local = False
			root_data_path = root_data_path.replace('s3://', '')
			self.urls = get_s3_paths(
				base_path			= root_data_path,
				train_valid_test	= train_valid_test,
				dataset_names		= dataset_names, 
				exclude				= exclude,
				cache_path			= cache_path,
				use_cache			= use_cache,
				recache				= recache
				)
		else:
			self.is_local = True
			self.urls = get_local_paths(
				base_path			= root_data_path,
				train_valid_test	= train_valid_test,
				dataset_names		= dataset_names, 
				exclude				= exclude,
				cache_path			= cache_path,
				use_cache			= use_cache,
				recache				= recache
				)

		pl_logger.info(f"Urls found: \
			\n\tTrain: {len(self.urls.get(train_valid_test[0], []))} \
			\n\tValid: {len(self.urls.get(train_valid_test[1], []))} \
			\n\tTest: {len(self.urls.get (train_valid_test[2], []))}"
		)
		
		self.train_data_dir = self.urls.get(train_valid_test[0], None)
		self.valid_data_dir = self.urls.get(train_valid_test[1], None)
		self.test_data_dir = self.urls.get(train_valid_test[2], None)

		self.male_keys = ['male', 'man', 'men', 'males']
		self.female_keys = ['female', 'woman', 'women', 'females']

	def to_samples(self, data):
		a, t = data
		try:
			return soundfile.read(io.BytesIO(a[1].read())), json.loads(t[1].read().decode('utf-8'))
		except:
			return None

	def get_stat(self, data):
		audio, text = data
		keys = ['gender', 'actor']
		
		gender = 'unknown'
		try:
			for key in keys:
				if key in text['original_data']:
					gender = text['original_data'][key].lower()
					break
		except KeyError:
			for key in self.male_keys + self.female_keys:
				if f'a {key} saying' in text['text'][0].lower():
					gender = key
					break


		sample_stats = {'time': len(audio[0])/audio[1], 'gender': gender}

		return sample_stats
	
	def collate_fn(self, data):
		male_keys = ['male', 'man', 'men', 'males']
		female_keys = ['female', 'woman', 'women', 'females']

		total_stats = {'time':0, 'male':0, 'female':0, 'unknown':0}

		for sample in data:
			total_stats['time'] += sample['time']
			if sample['gender'] in male_keys:
				total_stats['male'] += 1
			elif sample['gender'] in female_keys:
				total_stats['female'] += 1 
			else:
				total_stats['unknown'] += 1
		return total_stats

	def filter_fn(self, sample):
		if sample is None:
			return False
		else: 
			return True

	def filter_must_have_flac_and_json(self, data):
		try:
			_, _ = data
			return True
		except ValueError:
			return False

	
	def _log_tar(self, data):
		# log to file
		with open('/fsx/knoriy/CLASP/logs/get_data_stats.txt', 'a') as f:
			f.write(f"{data[0].replace('s3://laion-west-audio/webdataset_tar/', '')}\n")
		return data

	def _log_sample(self, data):
		a, t = data
		with open('/fsx/knoriy/CLASP/logs/get_data_stats.txt', 'a') as f:
			f.write(f"{a[0]}\n")
		return data
	def create_pipeline(self, data_dir):
		datapipe = torchdata.datapipes.iter.IterableWrapper(data_dir)\
			.sharding_filter() # Sharding filter here causes the dataloader to hang when using multiple GPUs
		
		if self.is_local:
			datapipe = FSSpecFileOpener(datapipe, mode='rb')
		else:
			datapipe = Boto3FileOpener(datapipe, mode='rb')
		datapipe = datapipe.load_from_tar() \
			.groupby(group_by_filename, group_size=2, guaranteed_group_size=2, drop_remaining=True)\
			.filter(self.filter_must_have_flac_and_json) \
			.map(self.to_samples) \
			.filter(self.filter_fn)\
			.map(self.get_stat) \
		
		if self.dataloader2:
			datapipe = datapipe.batch(self.batch_size) \
				.map(self.collate_fn)\
				.fullsync()
		return datapipe


class LangStatsTDM(GetStatsTDM):
	def get_stat(self, data):
		audio, text = data
		keys = ['gender', 'actor']
		
		gender = 'unknown'
		try:
			for key in keys:
				if key in text['original_data']:
					gender = text['original_data'][key].lower()
					break
		except KeyError:
			for key in self.male_keys + self.female_keys:
				if f'a {key} saying' in text['text'][0].lower():
					gender = key
					break


		sample_stats = {'time': len(audio[0])/audio[1], 'gender': gender, 'lang': text['original_data']['language']}

		return sample_stats
	
	def collate_fn(self, data):
		male_keys = ['male', 'man', 'men', 'males']
		female_keys = ['female', 'woman', 'women', 'females']

		total_stats = {'time':0, 'male':0, 'female':0, 'unknown':0}
		lang_stats = {}

		for sample in data:
			if sample['lang'] not in lang_stats:
				lang_stats[sample['lang']] = total_stats
			
			lang_stats[sample['lang']]['time'] += sample['time']

			if sample['gender'] in male_keys:
				lang_stats[sample['lang']]['male'] += 1
			elif sample['gender'] in female_keys:
				lang_stats[sample['lang']]['female'] += 1 
			else:
				lang_stats[sample['lang']]['unknown'] += 1

		return lang_stats

if __name__ == '__main__':
	import tempfile
	import argparse
	import tqdm

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_name', type=str, required=True)
	parser.add_argument('--lang', type=bool, default=False)

	args = parser.parse_args()

	# create a temp file for test_list.txt and insert "text"
	with tempfile.NamedTemporaryFile(mode='w+') as f:
		f.write(args.dataset_name)
		f.seek(0)
		if args.lang:
			dataset = LangStatsTDM(
				root_data_path='s3://laion-west-audio/webdataset_tar/', 
				dataset_list=f.name,
				exclude_list='/fsx/knoriy/CLASP/config/exclude_list.txt',
				# exclude_list='/fsx/knoriy/CLASP/logs/get_data_stats.txt',
				batch_size = 64,
				num_workers=12,
			)
		else:
			dataset = GetStatsTDM(
				root_data_path='s3://laion-west-audio/webdataset_tar/', 
				dataset_list=f.name,
				# exclude_list='/fsx/knoriy/CLASP/config/exclude_list.txt',
				batch_size = 64,
				num_workers=12,
			)

	dataset.setup()
	
	if args.lang:
		total_stats = {}
	else:
		total_stats = {'time':0, 'male':0, 'female':0, 'unknown':0}

	for dl in [dataset.train_dataloader, dataset.val_dataloader, dataset.test_dataloader]:
		print(dl.__name__)
		try:
			p_bar = tqdm.tqdm(dl())
			if args.lang:
				for i, val in enumerate(p_bar):
					for key in val.keys():
						if key not in total_stats:
							total_stats[key] = {'time':0, 'male':0, 'female':0, 'unknown':0}

						total_stats[key]['time'] += val[key]['time']
						total_stats[key]['male'] += val[key]['male']
						total_stats[key]['female'] += val[key]['female']
						total_stats[key]['unknown'] += val[key]['unknown']
						p_bar.set_description(f"Lang: {key}")
					p_bar.update()
			else:
				for i, val in enumerate(p_bar):
					total_stats['time'] += val['time']
					total_stats['male'] += val['male']
					total_stats['female'] += val['female']
					total_stats['unknown'] += val['unknown']
					p_bar.set_description(f"Lang: {key}")
					p_bar.update()
		except AttributeError:
			pass

	if args.lang:
		for key in total_stats.keys():
			total_stats[key]['time'] = ((total_stats[key]['time']/60)/60)
	else:
		total_stats['time'] = ((total_stats['time']/60)/60)
	
	if args.lang:
		for lang in total_stats.keys():
			print(f"Total Stats: {lang}, {total_stats[lang]['time']:.1f}  {total_stats[lang]['male']:,}  {total_stats[lang]['female']:,}  {total_stats[lang]['unknown']:,}")
	else:
		print(f"Total Stats: {total_stats}")