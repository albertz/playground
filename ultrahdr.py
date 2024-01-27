#!/usr/bin/env python3

"""
Play around with Ultra HDR (embedded in JPEG).

High-dynamic range (HDR) is to extend the usual color range
(Standard Dynamic Range (SDR))
and usually also extends the common 8bit color depth to 10bit or more.
https://en.wikipedia.org/wiki/High_dynamic_range

Some modern displays (~2021) (e.g. MacBook M1, some OLED TVs) support HDR,
but it is still a rare feature.

There are multiple formats for HDR images, e.g.:
- OpenEXR
- AVIF
- JPEG XT (https://en.wikipedia.org/wiki/JPEG_XT)
  embedded in JPEG XL (https://en.wikipedia.org/wiki/JPEG_XL)
- JPEG XR (https://en.wikipedia.org/wiki/JPEG_XR)
- Ultra HDR (used here)
  embedded in standard JPEG

Ultra HDR uses the JPEG multi-picture format (MPF).
It stores the normal SDR JPEG image as the first image,
so all existing JPEG decoders can display the normal image.
Then it stores a HDR gain map embedded in MPF
which can be used to reconstruct the HDR image.

Currently, (end of 2023), Google Chrome stable (end of 2023) supports this format.

Currently, (end of 2023), Google Pixel phones can capture Ultra HDR images
(e.g. when they use night mode).

About the Ultra HDR format:
https://developer.android.com/media/platform/hdr-image-format

> This document defines the behavior of a new file format
> that encodes a logarithmic range gain map image in a JPEG image file.
> Legacy readers that don't support the new format read and display
> the conventional low dynamic range image from the image file.
> Readers that support the format combine the primary image
> with the gain map and render a high dynamic range image on compatible displays.

To use the simple script here, for preparation:

* First, build this: https://github.com/google/libultrahdr
* Make sure FFMpeg is installed

Some related issues:
https://github.com/ImageMagick/ImageMagick/issues/6377
https://github.com/libvips/libvips/issues/3799

Demo: https://github.com/albertz/playground/wiki/HDR-demo

Note: Alternatives to Ultra HDR are: AVIF
"""

from typing import Optional, Tuple
import argparse
import os
import shutil
import subprocess
import struct
import atexit


ultra_hdr_app_path: Optional[str] = None
ffmpeg_path: Optional[str] = None


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", help="input file, jpeg")
    arg_parser.add_argument("output_file", help="output file, jpeg")
    args = arg_parser.parse_args()

    _setup_paths()

    w, h = _get_size_from_jpeg(args.input_file)
    print("Input size:", w, h)
    # Need to crop to even size, otherwise Ultra HDR encoding fails.
    w -= w % 2
    h -= h % 2
    print("Output (cropped) size:", w, h)

    yuv_file = args.output_file + ".yuv"
    _make_yuv_p010(args.input_file, yuv_file, (w, h))
    atexit.register(os.remove, yuv_file)  # yuv file not needed anymore

    jpeg_cropped_file = args.output_file + ".cropped.jpeg"
    _make_jpeg_cropped(args.input_file, jpeg_cropped_file, (w, h))
    atexit.register(os.remove, jpeg_cropped_file)  # cropped jpeg file not needed anymore

    _make_hdr_jpeg_from_yuv_p010_and_sdr_jpeg(jpeg_cropped_file, yuv_file, args.output_file, (w, h))


def _setup_paths():
    global ultra_hdr_app_path, ffmpeg_path

    ultra_hdr_build_path = "../libultrahdr/build"  # hardcoded currently...
    assert os.path.exists(ultra_hdr_build_path)
    ultra_hdr_app_path = ultra_hdr_build_path + "/ultrahdr_app"
    assert os.path.exists(ultra_hdr_app_path), "libultrahdr not built yet?"

    ffmpeg_path = shutil.which("ffmpeg")
    assert ffmpeg_path is not None, "ffmpeg not installed?"


def _make_yuv_p010(input_jpeg_file: str, output_yuv_p010_file: str, size: Tuple[int, int]):
    """Make YUV P010 from JPEG"""
    args = [
        ffmpeg_path,
        "-i",
        input_jpeg_file,
        "-filter:v",
        f"crop={size[0]}:{size[1]}:0:0,format=p010",
        output_yuv_p010_file,
    ]
    print("$", " ".join(args))
    subprocess.check_call(args)


def _make_jpeg_cropped(input_jpeg_file: str, output_jpeg_file: str, size: Tuple[int, int]):
    """Make JPEG cropped"""
    args = [ffmpeg_path, "-i", input_jpeg_file, "-filter:v", f"crop={size[0]}:{size[1]}:0:0", output_jpeg_file]
    print("$", " ".join(args))
    subprocess.check_call(args)


def _make_hdr_jpeg_from_yuv_p010_and_sdr_jpeg(
    input_jpeg_sdr_file: str, input_yuv_p010_file: str, output_jpeg_hdr_file: str, size: Tuple[int, int]
):
    """Make HDR JPEG (Ultra HDR) from YUV P010 and SDR JPEG"""
    args = [
        ultra_hdr_app_path,
        "-m",
        "0",
        "-p",
        input_yuv_p010_file,
        "-i",
        input_jpeg_sdr_file,
        "-w",
        str(size[0]),
        "-h",
        str(size[1]),
    ]
    print("$", " ".join(args))
    subprocess.check_call(args)
    shutil.move("out.jpeg", output_jpeg_hdr_file)


def _get_size_from_jpeg(input_jpeg_file: str) -> Tuple[int, int]:
    # https://stackoverflow.com/questions/8032642/how-can-i-obtain-the-image-size-using-a-standard-python-class-without-using-an
    with open(input_jpeg_file, "rb") as fhandle:
        size = 2
        ftype = 0
        while not 0xC0 <= ftype <= 0xCF or ftype in (0xC4, 0xC8, 0xCC):
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
