
import logging
log = logging.getLogger(__name__)

import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.cli import LightningCLI

from typing import Dict, Set, Optional

from td_datamodules import MultilingualTorchDataDataModule
from clasp import PLCLASP
from eval.zeroshot import zeroshot_eval
from utils import get_lists

class Trainer(pl.Trainer):
	def zeroshot(self, 
		model, 
		dataloaders, 
		zeroshot_templates, 
		datamodule: Optional[LightningDataModule] = None,
		ckpt_path: Optional[str] = None,
	) -> tuple[float, float]:
		Trainer._log_api_event("zeroshot")
		log.detail(f"{self.__class__.__name__}: trainer validate stage")
		templates = get_lists(zeroshot_templates)

		classes = ["hello world", "how are you?", "some random thing", "it's a beautiful day", "i love you", "goodbye", "i hate you", "today is not the day"]
		
		return zeroshot_eval(model, classes, templates, dataloaders)

class MyLightningCLI(LightningCLI):

	def add_arguments_to_parser(self, parser):
		# parser.add_lightning_class_args(DataParallelStrategy, "ddp_Strategy")
		# parser.set_defaults({"ddp_Strategy.find_unused_parameters": "False"})
		pass

	@staticmethod 
	def subcommands() -> Dict[str, Set[str]]: 
		"""Defines the list of available subcommands and the arguments to skip.""" 
		return { 
			"fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"}, 
			"validate": {"model", "dataloaders", "datamodule"}, 
			"test": {"model", "dataloaders", "datamodule"}, 
			"predict": {"model", "dataloaders", "datamodule"}, 
			"zeroshot": {"model", "dataloaders", "zeroshot_templates"}, 
		}

if __name__ == '__main__':
	cli = MyLightningCLI(PLCLASP, MultilingualTorchDataDataModule, trainer_class=Trainer, save_config_callback=None)
