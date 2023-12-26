#!/usr/bin/env python3
import io
import pathlib as p
import tarfile
from typing import Any, Sequence

import onnx
import onnxruntime
import torch
import torch.nn as nn
from torch import Tensor

from sodium.device import DeviceSupport
from sodium.torch_ex.types import TarWriteModeLiterals, TarReadonlyModeLiterals
from sodium.types import PathType


def check_file_is_safe(path: PathType,
                       arcname: str | None = None,
                       exists_ok: bool = True) -> p.Path | None:
    """
        Check File is Safe.
    :param path:
    :param arcname:
    :param exists_ok:
    :return:
    """
    path = p.Path(path)

    if path.exists():
        if path.is_file():

            # Auto Merging.
            return path

        elif path.is_dir():
            if arcname is None:
                raise IsADirectoryError("Path already used for directory")

            else:
                arcname = arcname.strip()  # trimming space char

                if arcname != "":
                    path = path.joinpath(arcname)
                    return check_file_is_safe(path, arcname, exists_ok)

                else:
                    raise Exception("Arcname is empty string")

        else:
            raise FileExistsError("Path already used with another format")

    if exists_ok:
        raise FileNotFoundError(f"Path file '{path}' is not found")

    return path


def check_file_is_readable(path: PathType,
                           arcname: str | None = None,
                           ignore_errors: bool = False) -> p.Path | None:
    """
        Check File is Readable.
    :param path:
    :param arcname:
    :param ignore_errors:
    :return:
    """

    try:
        path = check_file_is_safe(path, arcname)
        with open(path, "rb") as fstream:
            if not fstream.readable():
                raise Exception("File is not give read permission")

        return path

    except Exception as e:

        if not ignore_errors:
            raise RuntimeError(e)

        return None


def check_file_is_writable(path: PathType,
                           arcname: str | None = None,
                           ignore_errors: bool = False) -> p.Path | None:
    """
        Check File is Writable.
    :param path:
    :param arcname:
    :param ignore_errors:
    :return:
    """

    try:
        path = check_file_is_safe(path, arcname, exists_ok=False)
        with open(path, "wb") as fstream:
            if not fstream.writable():
                raise Exception("File is not give write permission")

        return path

    except Exception as e:

        if not ignore_errors:
            raise RuntimeError(e)

        return None


def torch_jit_model_save(model: nn.Module, path: PathType):
    """
        Torch JIT Model Save.
    :param model:
    :param path:
    :return:
    """
    path = check_file_is_writable(path, "model.pt")

    with torch.jit.optimized_execution(should_optimize=True):
        model_scripted = torch.jit.script(model, optimize=None)
        torch.jit.save(model_scripted, path)


def torch_jit_model_load(path: PathType,
                         map_location: Any | None = None,
                         device: torch.device = DeviceSupport.pref()) -> nn.Module:
    """
        Torch JIT Model Loader.
    :param path:
    :param map_location:
    :param device:
    :return:
    """
    if map_location is None:
        map_location = torch.device('cpu')

    path = check_file_is_readable(path, "model.pt")

    model = torch.jit.load(path, map_location=map_location)
    return model.to(device)


def torch_jit_model_save_tar(model: nn.Module,
                             path: PathType,
                             arcname: str = "model.pt",
                             mode: TarWriteModeLiterals = "w"):
    """
        Torch JIT Model Save with TAR Format.
    :param model:
    :param path:
    :param arcname:
    :param mode:
    :return:
    """
    path = check_file_is_writable(path, arcname)

    buffer = io.BytesIO()
    with tarfile.open(path, mode) as tarstream:
        with torch.jit.optimized_execution(should_optimize=True):
            scripted = torch.jit.script(model, optimize=None)
            torch.jit.save(scripted, buffer)
            buffer.seek(0)

            tarinfo = tarfile.TarInfo()
            tarinfo.name = "model.pt"
            tarinfo.size = buffer.getbuffer().nbytes

            tarstream.addfile(tarinfo, buffer)


def torch_jit_model_save_tgz(model: nn.Module, path: PathType, arcname: str = "model.pt.tar.gz"):
    """
        Torch JIT Model Save with TAR.GZ Format.
    :param model:
    :param path:
    :param arcname:
    :return:
    """
    torch_jit_model_save_tar(model, path, arcname, "w:gz")


def torch_jit_model_save_bz2(model: nn.Module, path: PathType, arcname: str = "model.pt.tar.bz2"):
    """
        Torch JIT Model Save with TAR.BZ2 Format.
    :param model:
    :param path:
    :param arcname:
    :return:
    """
    torch_jit_model_save_tar(model, path, arcname, "w:bz2")


def torch_jit_model_save_lzma(model: nn.Module, path: PathType, arcname: str = "model.pt.tar.xz"):
    """
        Torch JIT Model Save with TAR.XZ Format.
    :param model:
    :param path:
    :param arcname:
    :return:
    """
    torch_jit_model_save_tar(model, path, arcname, "w:xz")


def torch_jit_model_load_tar(path: PathType,
                             map_location: Any | None = None,
                             arcname: str = "model.pt",
                             mode: TarReadonlyModeLiterals = "r",
                             device: torch.device = DeviceSupport.pref()) -> nn.Module:
    """
        Torch JIT Model Loader with TAR Format.
    :param path:
    :param map_location:
    :param arcname:
    :param mode:
    :param device:
    :return:
    """
    if map_location is None:
        map_location = torch.device('cpu')

    path = check_file_is_readable(path, arcname)

    with tarfile.open(path, mode) as tarstream:
        with torch.jit.optimized_execution(should_optimize=True):
            tarinfo = tarstream.getmember("model.pt")
            buffer = tarstream.extractfile(tarinfo)

            if buffer is None:
                raise Exception("Couldn't extract file model pytorch")

            model = torch.jit.load(buffer, map_location=map_location)
            return model.to(device)


def torch_jit_model_load_tgz(path: PathType,
                             map_location: Any | None = None,
                             arcname: str = "model.pt.tar.gz") -> nn.Module:
    """
        Torch JIT Model Loader with TAR.TGZ Format.
    :param path:
    :param map_location:
    :param arcname:
    :return:
    """

    return torch_jit_model_load_tar(path, map_location, arcname, "r:gz")


def torch_jit_model_load_bz2(path: PathType,
                             map_location: Any | None = None,
                             arcname: str = "model.pt.tar.bz2") -> nn.Module:
    """
        Torch JIT Model Loader with TAR.BZ2 Format.
    :param path:
    :param map_location:
    :param arcname:
    :return:
    """

    return torch_jit_model_load_tar(path, map_location, arcname, "r:bz2")


def torch_jit_model_load_lzma(path: PathType,
                              map_location: Any | None = None,
                              arcname: str = "model.pt.tar.xz") -> nn.Module:
    """
        Torch JIT Model Loader with TAR.XZ Format.
    :param path:
    :param map_location:
    :param arcname:
    :return:
    """

    return torch_jit_model_load_tar(path, map_location, arcname, "r:xz")


def torch_onnx_model_save(model: nn.Module,
                          path: PathType,
                          dummy: Tensor | None = None,
                          input_names: Sequence[str] | None = None,
                          output_names: Sequence[str] | None = None,
                          export_params: bool = True,
                          verbose: bool = False,
                          device: torch.device = DeviceSupport.pref()):
    """
        Torch JIT Model Save with ONNX Format.
    :param model:
    :param path:
    :param dummy:
    :param input_names:
    :param output_names:
    :param export_params:
    :param verbose:
    :param device:
    :return:
    """
    path = check_file_is_writable(path, "model.onnx")

    if dummy is None:
        dummy = torch.randn((1, 3, 224, 244), device=device)

    # Link to Compatible Device
    model = model.to(device)
    dummy = dummy.to(device)

    torch.onnx.export(model,
                      dummy,
                      str(path),
                      export_params=export_params,
                      verbose=verbose,
                      input_names=input_names,
                      output_names=output_names)


def torch_onnx_model_load(path: PathType) -> onnxruntime.InferenceSession:
    """
        Torch JIT Model Loader with TAR Format.
    :param path:
    :return:
    """
    path = check_file_is_readable(path, "model.onnx")

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    ort_session = onnxruntime.InferenceSession(path, providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ])

    return ort_session
