# -*- coding: utf-8 -*-
"""Query the hardware information and software information of torchpilot runtime."""

import ctypes
import logging
import platform
import re
import subprocess  # noqa: S404
from typing import Any
from typing import Dict

import psutil  # noqa: I900
from tabulate import tabulate


logger = logging.getLogger(__name__)

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36


def _convert_smver2cores(major, minor) -> int:
    """Return the number of CUDA cores per multiprocessor.

    There is no way to retrieve that via the API, so it needs to be hard-coded.

    See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    """
    return {
        (1, 0): 8,  # Tesla
        (1, 1): 8,
        (1, 2): 8,
        (1, 3): 8,
        (2, 0): 32,  # Fermi
        (2, 1): 48,
        (3, 0): 192,  # Kepler
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,  # Maxwell
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,  # Pascal
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,  # Volta
        (7, 2): 64,
        (7, 5): 64,  # Turing
        (8, 0): 64,  # Ampere
        (8, 6): 64,
    }.get((major, minor), 0)


def _scale_size(bytes_data, suffix="B") -> str:
    """Scale bytes_data to its proper format.

    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes_data < factor:
            return f"{bytes_data:.2f}{unit}{suffix}"
        bytes_data /= factor
    return ""


def _get_platform_info() -> Dict[str, Any]:
    """Get platform info."""
    uname = platform.uname()
    info = {
        "System": uname.system,
        "Node Name": uname.node,
        "Release": uname.release,
        "Version": uname.version,
    }
    return info


def _get_cpu_info() -> Dict[str, Any]:
    """Get cpu info."""
    attrs = [
        "Architecture",
        "CPU op-mode(s)",
        "Byte Order",
        "CPU(s)",
        "On-line CPU(s) list",
        "Thread(s) per core",
        "Core(s) per socket",
        "Socket(s)",
        "NUMA node(s)",
        "Model name",
        "CPU max MHz",
        "Virtualization",
        "L1d cache",
        "L1i cache",
        "L2 cache",
        "L3 cache",
    ]
    info = {}
    try:
        with subprocess.Popen(  # noqa: S607
            "lscpu",
            shell=True,  # noqa: S602, 607
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            try:
                outs_bytes, _ = proc.communicate(timeout=15)
                outs_str = bytes.decode(outs_bytes)
                outs_str_list = outs_str.split("\n")
                for item_str in outs_str_list:
                    item_attr = item_str.split(":")[0]
                    if item_attr in attrs:
                        if "cache" in item_attr:
                            info[item_attr] = (
                                f"{item_str.split()[-2]} {item_str.split()[-1]}"
                            )
                        else:
                            info[item_attr] = item_str.split()[-1]
            except Exception:  # pylint: disable=broad-except
                proc.kill()
                outs_bytes, _ = proc.communicate()
                return info
    except Exception:  # pylint: disable=broad-except
        return info
    return info


def _get_gpu_info() -> Dict[str, Any]:
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-statements
    """Get gpu info."""
    info = {}
    try:
        with subprocess.Popen(  # noqa: S607
            "cat /proc/driver/nvidia/version",
            shell=True,  # noqa: S602, 607
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            try:
                outs, _ = proc.communicate(timeout=15)
                outs = bytes.decode(outs)  # type: ignore[assignment]
                outs = outs.split("\n")  # type: ignore[arg-type, assignment]
                res = re.search(
                    r".*  (\d+).(\d+).(\d+) ", outs[0], flags=0
                )  # type: ignore[call-overload]
                version = res.group(0).strip().split(" ")[-1]
                info["NVIDIA GPU Driver Version"] = version
            except Exception:  # pylint: disable=broad-except
                proc.kill()
                outs, _ = proc.communicate()
                return {}
    except Exception:  # pylint: disable=broad-except
        return {}

    # 1. GPU Architecture Info

    libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll")
    cuda = None
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except Exception:  # pylint: disable=broad-except # noqa: S110
            pass
    if cuda is None:
        return info

    n_gpus = ctypes.c_int()
    name = b" " * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    free_mem = ctypes.c_size_t()
    total_mem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return info
    result = cuda.cuDeviceGetCount(ctypes.byref(n_gpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return info
    info["GPU NUMs"] = n_gpus.value
    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return info

    if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
        info["GPU NAME"] = name.split(b"\0", 1)[0].decode()
    if (
        cuda.cuDeviceComputeCapability(
            ctypes.byref(cc_major), ctypes.byref(cc_minor), device
        )
        == CUDA_SUCCESS
    ):
        info["Compute Capability"] = f"{cc_major.value}.{cc_minor.value}"
    if (
        cuda.cuDeviceGetAttribute(
            ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
        )
        == CUDA_SUCCESS
    ):
        info["SMs"] = cores.value
        info["CUDA Cores/SM"] = _convert_smver2cores(cc_major.value, cc_minor.value)
        info["CUDA Cores"] = cores.value * _convert_smver2cores(
            cc_major.value, cc_minor.value
        )

        if (
            cuda.cuDeviceGetAttribute(
                ctypes.byref(threads_per_core),
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                device,
            )
            == CUDA_SUCCESS
        ):
            info["Max Number of Threads per SM"] = threads_per_core.value
            info["Concurrent threads"] = cores.value * threads_per_core.value
    if (
        cuda.cuDeviceGetAttribute(
            ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device
        )
        == CUDA_SUCCESS
    ):
        info["GPU clock /MHz"] = f"{clockrate.value / 1000.0:.4f}"
    if (
        cuda.cuDeviceGetAttribute(
            ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device
        )
        == CUDA_SUCCESS
    ):
        info["GPU Memory clock /MHz"] = f"{clockrate.value / 1000.0:.4f}"

    try:
        result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
    except AttributeError:
        result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
    if result != CUDA_SUCCESS:  # pylint: disable=no-else-return
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        return info
    else:  # pylint: disable=no-else-return
        try:
            result = cuda.cuMemGetInfo_v2(
                ctypes.byref(free_mem), ctypes.byref(total_mem)
            )
        except AttributeError:
            result = cuda.cuMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
        if result == CUDA_SUCCESS:
            info["Total GPU Memory /MiB"] = total_mem.value / 1024**2
            info["Free GPU Memory /MiB"] = free_mem.value / 1024**2
        else:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
        cuda.cuCtxDetach(context)
    return info


def _get_mem_info() -> Dict[str, Any]:
    """Get mem info."""
    info = {}
    svmem = psutil.virtual_memory()
    info["Total"] = _scale_size(svmem.total)
    info["Available"] = _scale_size(svmem.available)
    info["Used"] = _scale_size(svmem.used)
    info["Percentage /%"] = svmem.percent

    swap = psutil.swap_memory()
    info["SWAP Total"] = _scale_size(swap.total)
    info["SWAP Free"] = _scale_size(swap.free)
    info["SWAP Used"] = _scale_size(swap.used)
    info["SWAP Percentage /%"] = swap.percent
    return info


def _format_info(header: str, log_dict: Dict[str, str]) -> None:
    log_list = []
    for item_name, item_value in log_dict.items():
        log_list.append([f"{item_name}: {item_value}"])

    logger.info("\n%s", tabulate(log_list, headers=[header], tablefmt="psql"))


def show_hardware_info() -> None:
    """Get platform&cpu&gpu&men info."""
    _format_info("platform", _get_platform_info())
    _format_info("cpu", _get_cpu_info())
    _format_info("gpu", _get_gpu_info())
    _format_info("mem", _get_mem_info())
