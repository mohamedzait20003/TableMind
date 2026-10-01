"""
Checkpoint saving, loading and Google Drive transfer.

Each stage saves locally, then uploads to persistent storage (a Drive folder
on Colab, a local folder elsewhere); the next stage downloads from there.
"""

import os
import sys
import torch
import shutil
import torch.nn as nn
from typing import Any, Dict, Optional

DRIVE_MOUNT = '/content/drive'


def in_colab() -> bool:
    """True when running inside Google Colab."""
    return 'google.colab' in sys.modules


def get_storage_dir(
    drive_subdir: str = 'TableMind/checkpoints',
    local_dir: str = './storage'
) -> str:
    """
    Return the persistent checkpoint folder, mounting Google Drive on Colab.

    Args:
        drive_subdir: Folder under "MyDrive" used on Colab
        local_dir: Folder used outside Colab

    Returns:
        Path of the (created) storage folder
    """
    if in_colab():
        from google.colab import drive
        if not os.path.ismount(DRIVE_MOUNT):
            drive.mount(DRIVE_MOUNT)
        path = os.path.join(DRIVE_MOUNT, 'MyDrive', drive_subdir)
    else:
        path = local_dir
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(
    path: str,
    model: nn.Module,
    memory_store=None,
    trainer=None,
    config: Optional[Dict[str, Any]] = None,
    include_optimizer: bool = False
) -> str:
    """
    Save model weights, memory store contents and training state.

    Args:
        path: Output file (parent folders are created)
        model: Model whose state_dict is saved
        memory_store: KeyValueMemoryStore to save (keys and values)
        trainer: BaseTrainer whose history/counters are saved
        config: Configuration dict stored alongside the weights
        include_optimizer: Also save optimizer/scheduler state (for resuming;
            roughly triples the file size)

    Returns:
        The checkpoint path
    """
    checkpoint = {'model_state_dict': model.state_dict()}
    if memory_store is not None:
        checkpoint['memory_store'] = memory_store.state_dict()
    if trainer is not None:
        checkpoint['trainer_state'] = trainer.state_dict(include_optimizer)
    if config is not None:
        checkpoint['config'] = config

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    memory_store=None,
    trainer=None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load a checkpoint written by `save_checkpoint`.

    Args:
        path: Checkpoint file
        model: Model to load weights into
        memory_store: KeyValueMemoryStore to restore, if saved
        trainer: BaseTrainer to restore training state into, if saved
        device: map_location for the tensors

    Returns:
        The raw checkpoint dict (e.g. to read its 'config')
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if memory_store is not None and 'memory_store' in checkpoint:
        memory_store.load_state_dict(checkpoint['memory_store'])
    if trainer is not None and 'trainer_state' in checkpoint:
        trainer.load_state_dict(checkpoint['trainer_state'])
    print(f"Checkpoint loaded: {path}")
    return checkpoint


def upload_checkpoint(local_path: str, storage_dir: str) -> str:
    """
    Copy a local checkpoint into the storage folder (Drive on Colab).

    Returns:
        Path of the uploaded copy
    """
    destination = os.path.join(storage_dir, os.path.basename(local_path))
    if os.path.abspath(destination) != os.path.abspath(local_path):
        shutil.copy2(local_path, destination)
    print(f"Uploaded: {destination}")
    return destination


def download_checkpoint(
    name: str, storage_dir: str, local_dir: str = './checkpoints'
) -> str:
    """
    Copy a checkpoint from the storage folder to local disk (faster to load).

    Args:
        name: Checkpoint file name, e.g. 'memory_stage.pt'
        storage_dir: Storage folder from `get_storage_dir`
        local_dir: Local destination folder

    Returns:
        Local path of the checkpoint
    """
    source = os.path.join(storage_dir, name)
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"{source} not found - run the previous stage's notebook first"
        )
    os.makedirs(local_dir, exist_ok=True)
    destination = os.path.join(local_dir, name)
    if os.path.abspath(source) != os.path.abspath(destination):
        shutil.copy2(source, destination)
    return destination
