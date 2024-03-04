"""
Testing https://twitter.com/francoisfleuret/status/1764315206853755148
"""

import torch


def get():
    tmp_img = torch.zeros(1, 3, 8 * 128, 8 * 128)
    print(f"{tmp_img.untyped_storage().data_ptr()=}")
    img = torch.nn.functional.avg_pool2d(tmp_img, 8)
    return img


def main():
    keep = []

    with torch.no_grad():
        for i in range(10_000):
            img = get()
            print(
                f"{i=}, {img.shape=}, {img.is_contiguous()=}, {img.nbytes=},"
                f" {img.untyped_storage().nbytes()=}, {img.storage_offset()=}, {img.untyped_storage().data_ptr()=}"
            )
            keep.append(img)


if __name__ == "__main__":
    main()
