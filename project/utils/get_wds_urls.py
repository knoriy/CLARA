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
		dataset_names:list[str]=None, 
		train_valid_test:list[str]=['train'], 
		cache_path:str='', 
		recache:bool=False,
	):
	if os.path.isfile(cache_path) and not recache:
		with open(cache_path) as f:
			print("Loading Cache")
			return json.load(f)
	
	if dataset_names:
		cmds = [f'aws s3 ls s3://{os.path.join(base_s3_path, name, "")} --recursive | grep /.*.tar' for name in dataset_names]
	else:
		cmds = [f'aws s3 ls s3://{os.path.join("s-laion-audio/webdataset_tar/", "")} --recursive | grep /.*.tar']

	urls = [os.popen(cmd).read() for cmd in cmds]
	final_urls = [i.split(' ')[-1] for url in urls for i in url.split('\n')]
	final_urls = [f'pipe:aws s3 --cli-connect-timeout 0 cp s3://{os.path.join(base_s3_path, i)}' for i in final_urls]
	
	tmp= {}
	for state in train_valid_test:
		tmp[state] = [url for url in final_urls if state in url] 

	if cache_path:
		with open(cache_path, 'w') as f:
			json.dump(tmp, f)

	return tmp


if __name__ == '__main__':
	urls = get_tar_path_s3(
		's-laion-audio/webdataset_tar/', 
		['LJSpeech'], 
		cache_path='./url_cache.json',
		recache=True,
		)

	
	print(urls)

	# import boto3 
	# client = boto3.client('s3')
	# print(client.list_objects(Bucket='s-laion-audio'))

