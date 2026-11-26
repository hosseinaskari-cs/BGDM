"""Fetch original BGDM release assets and verify their checksums."""

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from .tasks import Task


ASSETS = {
    "brats": (
        ("1pwaDNuDIuAJzCLkrVwj4DIxegAlwsLbd", "62caca4479b854ef077e0e5decadc2feae09a7a7e1c75b6ee1bc94c5e05d4829"),
        ("1RRdop20U6bwfWWgO0DzHdb98EJR5igAs", "64e2db74762d8ffc797a474ee70952703ec3242f5c406d99a2a8c2384ba142a7"),
    ),
    "fastmri-knee": (
        ("1AgV6Smd_NiwJnUSZc-PAeeu69fqlawUh", "0f6adebf4ce4000f2eabbb119fc8460466a4f78000555ede3d96f86c3e6e2dff"),
        ("1Hom4MdfUrvb1jTuA3ExCG5z0s_AaFour", "3d7d28494a8a4a0dac8ea0d912f778d11802466f32522786570b288aa48d3451"),
    ),
    "fastmri-brain": (
        ("1lFOYJEDk_ksThgJJvcRcaEm3AL7R-3nk", "ad5acb9064ca5d96b1c165b4e61ea5297ce02caf149ab13042af777ee99c2b3f"),
        ("19U3JGAL0KqJtZ0PlB1yDCc2m5s5SHWFg", "f2e2421c47dd1c4272913745d72adeb1e95683cd56b8fac130731ea1570fad80"),
    ),
    "lidc": (
        ("11xh4kCYx4Xsbgb1T5m0DmDgqa63uBOiv", "a8ab4abf33f3a294e9993a382671dbfaa9226271bcbabb482b07a898625fdc9f"),
        ("1xocXIAsKZXj3cMiL7zOrd79I-4Kw80Mn", "1f007821fa88bae76e6e608d1ac7549b3c3146abd60f1c128f898e9d736bdccf"),
    ),
}


def default_asset_dir() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "bgdm"


def checkpoint_path(root: Path, task: Task) -> Path:
    return root / "checkpoints" / task.checkpoint_folder / task.checkpoint


def dataset_path(root: Path, task: Task) -> Path:
    return root / "test_data" / task.dataset


def download_asset(destination: Path, file_id: str, expected_hash: str) -> None:
    if destination.is_file():
        digest = hashlib.sha256()
        with destination.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Checksum mismatch for existing file {destination}. Use a different --assets-dir.")
        print(f"Verified {destination}")
        return

    from tqdm import tqdm

    source = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(source, headers={"User-Agent": "BGDM reference asset downloader"})
        with urlopen(request, timeout=45) as response:
            if response.headers.get_content_type() == "text/html":
                raise OSError("Google Drive returned a web page instead of the original file.")
            size = response.headers.get("Content-Length")
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
                temporary = Path(file.name)
                digest = hashlib.sha256()
                with tqdm(total=int(size) if size else None, unit="B", unit_scale=True, desc=destination.name) as bar:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        file.write(chunk)
                        digest.update(chunk)
                        bar.update(len(chunk))
        if digest.hexdigest() != expected_hash:
            raise OSError(f"Downloaded file does not match the published asset checksum: {destination.name}")
        # Linking never replaces a file created by another process.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            download_asset(destination, file_id, expected_hash)
            return
        print(f"Saved {destination}")
    except OSError as error:
        raise OSError(f"{error} Manual download: https://drive.google.com/file/d/{file_id}/view") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_task(root: Path, task: Task, *, with_data: bool) -> None:
    weights, dataset = ASSETS[task.name]
    download_asset(checkpoint_path(root, task), *weights)
    if with_data:
        download_asset(dataset_path(root, task), *dataset)
