"""Unpacking the ImageNet train archive, on a tiny archive of the same shape: one inner tar per class."""

import io
import os
import tarfile

import pytest

from corread import imagenet


def tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def write_train_archive(path: str, classes: dict[str, dict[str, bytes]]) -> None:
    with open(path, 'wb') as f:
        f.write(tar_bytes({wnid + '.tar': tar_bytes(images) for wnid, images in classes.items()}))


def test_unpacks_and_uses_up_the_archive(tmp_path):
    archive = str(tmp_path / 'train.tar')
    write_train_archive(archive, {'n01': {'a.JPEG': b'1', 'b.JPEG': b'2'}, 'n02': {'c.JPEG': b'3'}})
    dest = str(tmp_path / 'train')
    imagenet._unpack_train(archive, dest, workers=2, keep_archive=False)
    assert sorted(os.listdir(dest)) == ['n01', 'n02']
    assert open(os.path.join(dest, 'n01', 'b.JPEG'), 'rb').read() == b'2'
    assert not os.path.exists(archive)


def test_keeps_the_archive_when_asked(tmp_path):
    archive = str(tmp_path / 'train.tar')
    write_train_archive(archive, {'n01': {'a.JPEG': b'1'}})
    imagenet._unpack_train(archive, str(tmp_path / 'train'), workers=1, keep_archive=True)
    assert os.path.isfile(archive)


@pytest.mark.parametrize('name', ['../escape.JPEG', 'sub/a.JPEG', '/abs.JPEG'])
def test_refuses_paths_out_of_the_class_folder(tmp_path, name):
    archive = str(tmp_path / 'train.tar')
    write_train_archive(archive, {'n01': {name: b'x'}})
    with pytest.raises(ValueError, match='Unexpected entry'):
        imagenet._unpack_train(archive, str(tmp_path / 'train'), workers=1, keep_archive=True)
    assert not os.path.exists(tmp_path / 'escape.JPEG')


def test_missing_archives_are_named(tmp_path):
    with pytest.raises(FileNotFoundError, match=r'ILSVRC2012_img_val\.tar'):
        imagenet.prepare_imagenet(str(tmp_path))
