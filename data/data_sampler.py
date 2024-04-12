#    Copyright 2024 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from .sampler import GridSampler, ChunkedGridSampler
from .aggregator import WeightedSoftmaxAggregator, ChunkedWeightedSoftmaxAggregator
from .slicer import slicer

import numpy as np
import tifffile
import torch
import zarr
import copy
import shutil
import os


def load_tiff(path):

    im = tifffile.imread(path)
    # im = im.transpose((1, 0, 2))

    return im


class CustomGridSampler(GridSampler):
    def __getitem__(self, idx):
        indices = self.indices[idx]
        patch_indices = np.zeros(len(indices) * 2, dtype=int).reshape(-1, 2)
        for axis in range(len(indices)):
            patch_indices[axis][0] = indices[axis]
            patch_indices[axis][1] = indices[axis] + self.patch_size[axis]
        if self.image is not None and not isinstance(self.image, dict):
            slices = self.get_slices(self.image, patch_indices)
            patch = self.image[slicer(self.image, slices)]

            if self.transforms:

                patch = patch.transpose((1, 2, 0))
                # print(patch.shape)
                patch = self.transforms(image=patch)["image"]

            return patch, patch_indices
        elif self.image is not None and isinstance(self.image, dict):
            patch_dict = {}
            for key in self.image.keys():
                slices = self.get_slices(self.image[key], patch_indices)
                patch_dict[key] = self.image[key][slicer(self.image[key], slices)]
            return patch_dict, patch_indices
        else:
            return patch_indices

    def set_test_transforms(self, transforms):

        self.transforms = transforms


class CustomChunkedGridSampler(ChunkedGridSampler):
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration
        chunk_id = np.argmax(self.length > idx)
        patch_id = idx - self.length[chunk_id - 1]
        chunk_id -= 1  # -1 to remove the [0] appended at the start of self.length

        patch_indices = copy.copy(self.chunk_sampler[chunk_id].__getitem__(patch_id))
        patch_indices += self.chunk_sampler_offset[chunk_id].reshape(-1, 1)
        # self.patch_index += 1
        if self.image is not None and not isinstance(self.image, dict):
            slices = self.get_slices(self.image, patch_indices)
            patch = self.image[slicer(self.image, slices)]

            if self.transforms:

                patch = patch.transpose((1, 2, 0))
                # print(patch.shape)
                patch = self.transforms(image=patch)["image"]

            return patch, (patch_indices, chunk_id)
        elif self.image is not None and isinstance(self.image, dict):
            patch_dict = {}
            for key in self.image.keys():
                slices = self.get_slices(self.image[key], patch_indices)
                patch_dict[key] = self.image[key][slicer(self.image[key], slices)]
            return patch_dict, (patch_indices, chunk_id)
        else:
            return (patch_indices, chunk_id)

    def set_test_transforms(self, transforms):

        self.transforms = transforms


def get_patch_handler(
    path,
    patch_size=512,
    sliding_window_overlap=128,
    batch_size=16,
    num_workers=8,
    transform=None,
    chunked=False,
    zarr_path=None,
    chunk_multiplier=16,
):
    # load image
    img = load_tiff(path)
    x, y = img.shape[0], img.shape[1]

    print(img.shape)

    img = img.transpose((2, 0, 1))
    print(img.shape)

    # init GridSampler
    if chunked:
        print("Using chunked grid sampler")
        grid_sampler = CustomChunkedGridSampler(
            img,
            image_size=(x, y),
            patch_size=(patch_size, patch_size),
            patch_overlap=(sliding_window_overlap, sliding_window_overlap),
            chunk_size=(patch_size * chunk_multiplier, patch_size * chunk_multiplier),
        )
    else:
        grid_sampler = CustomGridSampler(
            img,
            image_size=(x, y),
            patch_size=(patch_size, patch_size),
            patch_overlap=(sliding_window_overlap, sliding_window_overlap),
        )
    grid_sampler.set_test_transforms(transform)

    # init Aggregator
    if chunked:
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        softmax_pred = zarr.open(
            zarr_path,
            mode="w",
            shape=(x, y),
            chunks=(patch_size * chunk_multiplier, patch_size * chunk_multiplier),
            dtype=np.uint8,
        )
    else:
        softmax_pred = np.zeros((6, x, y), dtype=np.float32)
    if chunked:
        softmax_aggregator = ChunkedWeightedSoftmaxAggregator(
            softmax_pred,
            image_size=(x, y),
            patch_size=(patch_size, patch_size),
            chunk_size=(patch_size * chunk_multiplier, patch_size * chunk_multiplier),
            patch_overlap=(sliding_window_overlap, sliding_window_overlap),
        )
    else:
        softmax_aggregator = WeightedSoftmaxAggregator(
            softmax_pred, image_size=(x, y), patch_size=(patch_size, patch_size)
        )

    patch_loader = torch.utils.data.DataLoader(
        grid_sampler, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return patch_loader, softmax_aggregator
