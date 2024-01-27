#!/usr/bin/env python3

"""
Play around with Ultra HDR (embedded in JPEG).

* First, build this: https://github.com/google/libultrahdr
* Make sure FFMpeg is installed

Some related issues:
https://github.com/ImageMagick/ImageMagick/issues/6377
https://github.com/libvips/libvips/issues/3799

Note: Alternatives to Ultra HDR are: AVIF
"""

from typing import Optional, Tuple
import argparse
import os
import shutil
import subprocess
import struct


ultra_hdr_app_path: Optional[str] = None
ffmpeg_path: Optional[str] = None


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", help="input file, jpeg")
    arg_parser.add_argument("output_file", help="output file, jpeg")
    args = arg_parser.parse_args()

    _setup_paths()

    yuv_file = args.output_file + ".yuv"
    _make_yuv_p010(args.input_file, yuv_file)
    _make_jpeg_from_yuv_p010(args.input_file, yuv_file, args.output_file)
    os.remove(yuv_file)  # not needed anymore


def _setup_paths():
    global ultra_hdr_app_path, ffmpeg_path

    ultra_hdr_build_path = "../libultrahdr/build"  # hardcoded currently...
    assert os.path.exists(ultra_hdr_build_path)
    ultra_hdr_app_path = ultra_hdr_build_path + "/ultrahdr_app"
    assert os.path.exists(ultra_hdr_app_path), "libultrahdr not built yet?"

    ffmpeg_path = shutil.which("ffmpeg")
    assert ffmpeg_path is not None, "ffmpeg not installed?"


def _make_yuv_p010(input_jpeg_file: str, output_yuv_p010_file: str):
    """Make YUV P010 from JPEG"""
    args = [ffmpeg_path, "-i", input_jpeg_file, "-vf", "format=p010", output_yuv_p010_file]
    print("$", " ".join(args))
    subprocess.check_call(args)


def _make_jpeg_from_yuv_p010(input_jpeg_sdr_file: str, input_yuv_p010_file: str, output_jpeg_hdr_file: str):
    """Make JPEG from YUV P010"""
    w, h = _get_size_from_jpeg(input_jpeg_sdr_file)
    args = [
        ultra_hdr_app_path,
        "-m",
        "0",
        "-p",
        input_yuv_p010_file,
        "-i",
        input_jpeg_sdr_file,
        "-w",
        str(w),
        "-h",
        str(h),
    ]
    print("$", " ".join(args))
    subprocess.check_call(args)
    shutil.move("out.jpeg", output_jpeg_hdr_file)


def _get_size_from_jpeg(input_jpeg_file: str) -> Tuple[int, int]:
    # https://stackoverflow.com/questions/8032642/how-can-i-obtain-the-image-size-using-a-standard-python-class-without-using-an
    with open(input_jpeg_file, "rb") as fhandle:
        size = 2
        ftype = 0
        while not 0xC0 <= ftype <= 0xCF:
            fhandle.seek(size, 1)
            byte = fhandle.read(1)
            while ord(byte) == 0xFF:
                byte = fhandle.read(1)
            ftype = ord(byte)
            size = struct.unpack(">H", fhandle.read(2))[0] - 2
        # We are at a SOFn block
        fhandle.seek(1, 1)  # Skip `precision' byte.
        height, width = struct.unpack(">HH", fhandle.read(4))
        return width, height


if __name__ == "__main__":
    main()
