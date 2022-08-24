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
					tmp.append(f"{dataset_path}/{n}/{s}/{k}")
				else:
					tmp.append(
						f"pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/{n}/{s}/{k} -"
					)
			if proportion!=1:
				tmp = random.sample(tmp, int(proportion * len(tmp)))
			output.append(tmp)
	return sum(output, [])


if __name__ == '__main__':
	urls = get_tar_path_from_dataset_name(
		['LJSpeech'], 
		['train', 'test', 'valid'],
		False, 
		)
	print(urls)

	# import boto3 
	# client = boto3.client('s3')
	# print(client.list_objects(Bucket='s-laion-audio'))

