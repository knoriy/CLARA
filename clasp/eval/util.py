
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import json
from datamodule import *
from utils import get_lists

def get_dataset(task, dataset_name, root_cfg_path, root_data_path='s3://laion-west-audio/webdataset_tar/', batch_size=1, num_workers=0):
    ##########
	# Sounds
	##########

	if task == 'sounds':
		if dataset_name == 'esc50':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/esc-50/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/esc-50/minor_classes.json")) as f:
				classes = json.load(f)
			dataset = ESC50TDM(
						root_data_path=root_data_path,
						classes=classes,
						batch_size = batch_size,
						num_workers = num_workers,
					)
		elif dataset_name == 'audioset':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/audioset/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/audioset/classes.json")) as f:
				classes = json.load(f)
			dataset = AudioSetTDM(
						root_data_path=root_data_path,
						classes=classes,
						batch_size = batch_size,
						num_workers = num_workers,
					)
		elif dataset_name == 'us8k':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/us8k/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/us8k/classes.json")) as f:
				classes = json.load(f)
			dataset = Urbansound8KTDM(
						root_data_path=root_data_path,
						batch_size = batch_size,
						num_workers = num_workers,
					)
		elif dataset_name == 'fsd50k':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/fsd50k/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/fsd50k/classes.json")) as f:
				classes = json.load(f)
			dataset = FSD50KTDM(
						root_data_path=root_data_path,
						classes=classes,
						batch_size = batch_size,
						num_workers = num_workers,
					)

	##########
	# Gender
	##########
	elif task == 'gender':
		dataset = VoxCelebTDM(
					test_urls=['s3://s-laion/knoriy/VoxCeleb_gender/'],
					batch_size = batch_size,
					num_workers = num_workers,
				)
		templates = get_lists(os.path.join(root_cfg_path , "classification/gender/templates.txt"))
		with open(os.path.join(root_cfg_path , "classification/gender/classes.json")) as f:
			classes = json.load(f)

	##########
	# Emotion
	##########
	elif task == 'emotion':
		templates_path = os.path.join(root_cfg_path , f"classification/{task}/{dataset_name}/templates.txt")
		classes_path = os.path.join(root_cfg_path , f"classification/{task}/{dataset_name}/classes.json")

		if dataset_name == 'emns':
			dataset = EMNSTDM(
						root_data_path=root_data_path,
						batch_size = batch_size,
						num_workers = num_workers,
					)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif dataset_name == 'emov-db':
			dataset = EmovDBTDM(
				root_data_path=root_data_path,
				batch_size = batch_size,
				num_workers = num_workers,
			)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif dataset_name == 'crema-d':
			Warning("CREMA-D is not supported yet")
			dataset = CremaDTDM(
				root_data_path=root_data_path,
				batch_size = batch_size,
				num_workers = num_workers,
			)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif dataset_name == 'ravdess':
			dataset = RavdessTDM(
						root_data_path=root_data_path,
						batch_size = batch_size,
						num_workers = num_workers,
					)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)

	##########
	# age
	##########
	elif task == 'age':
		dataset = CommonVoiceTDM(
					test_urls=['s3://s-laion-audio/webdataset_tar/common_voice/test/'],
					batch_size = batch_size,
					num_workers = num_workers,
				)

		templates = get_lists(os.path.join(root_cfg_path , "classification/age/common_voice/templates.txt"))
		with open(os.path.join(root_cfg_path , "classification/age/common_voice/classes.json")) as f:
			classes = json.load(f)

	else:
		raise ValueError(f"Task {task} not supported")

	return dataset, templates, classes