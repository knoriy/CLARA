import os
import json
import random


def get_tar_path_from_dataset_name(
	dataset_names:list[str],
	dataset_types:list[str],
	islocal:bool,
	dataset_path:str=None,
	proportion:int=1,
):
	"""
	Get tar path from dataset name and type
	"""
	output = []
	for n in dataset_names:
		for s in dataset_types:
			tmp = []
			sizefilepath_ = f"./json_files/{n}/{s}/sizes.json"
			if not os.path.exists(sizefilepath_):
				continue
			sizes = json.load(open(sizefilepath_, "r"))
			for k in sizes.keys():
				if islocal:
					if dataset_path==None:
						raise ValueError(f'dataset_path must be provided if is_local = True')
					if not os.path.exists(dataset_path):
						raise ValueError(f'{dataset_path} Does not exist')
					tmp.append(f"{dataset_path}/{n}/{s}/{k}")
				else:
					tmp.append(
						f"pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/{n}/{s}/{k} -"
					)
			if proportion!=1:
				tmp = random.sample(tmp, int(proportion * len(tmp)))
			output.append(tmp)
	return sum(output, [])

def get_tar_path_s3(base_s3_path:str, 
		train_valid_test:list[str], 
		dataset_names:list[str]=[''], 
		exclude:list[str]=[], 
		cache_path:str='', 
		recache:bool=False,
	):
	if os.path.isfile(cache_path) and not recache:
		with open(cache_path) as f:
			print("Loading Cache")
			return json.load(f)

	# create cmd for collecting url spesific dataset, 
	# if `dataset_names` is not given it will search the full base_s3_path
	cmds = [f'aws s3 ls s3://{os.path.join(base_s3_path, name, "")} --recursive | grep /.*.tar' for name in dataset_names]
	# urls are collected
	urls = [os.popen(cmd).read() for cmd in cmds]
	# cleaning the urls to conform with webdataset
	final_urls = [i.split(' ')[-1] for url in urls for i in url.split('\n')]
	final_urls = [f'pipe:aws s3 --cli-connect-timeout 0 cp s3://{os.path.join(base_s3_path, *i.split("/")[1:])} -' for i in final_urls]
	# Spliting url by state e.g. train, test and valud
	final_urls = {state:[url for url in final_urls if state in url 
		and all(exclude_name not in url for exclude_name in exclude)] for state in train_valid_test}

	if cache_path:
		with open(cache_path, 'w') as f:
			json.dump(final_urls, f)

	return final_urls

if __name__ == '__main__':
	urls = get_tar_path_s3(
		's-laion-audio/webdataset_tar/', 
		['train', 'test', 'valid'],
		['EmoV_DB'], 
		cache_path='./url_cache.json',
		recache=True,
		)

	print(urls)


