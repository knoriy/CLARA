import os
import tempfile
import shutil
import pytorch_lightning as pl

class DeleteTempFilesCallback(pl.Callback):

    def _delete_temp_files(self, *args, **kwargs):
        file_path = os.path.join(tempfile.gettempdir(), "CLASP")
        if os.path.exists(file_path):
            shutil.rmtree(file_path)

    def on_train_end(self, *args, **kwargs):
        self._delete_temp_files(*args, **kwargs)

    def on_exception(self, *args, **kwargs):
        self._delete_temp_files(*args, **kwargs)

    def teardown(self, *args, **kwargs):
        self._delete_temp_files(*args, **kwargs)