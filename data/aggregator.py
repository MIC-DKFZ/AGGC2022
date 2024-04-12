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

import numpy as np
from .sampler import BasicGridSampler, GridSampler, AdaptiveGridSampler
from .slicer import slicer
from scipy.ndimage.filters import gaussian_filter

# from utils import normalize
from collections import defaultdict
import copy
from skimage import transform as ski_transform
import concurrent.futures
import gc


class Aggregator:
    def __init__(self, image=None, image_size=None):
        """
        Aggregator to assemble an image with continuous content from patches. The content of overlapping patches is averaged.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param image: A numpy-style zero-initialized image (Numpy, Zarr, Dask, ...) of a continuous data type. If none then a zero-initialized Numpy image of data type np.float32 is created internally.
        :param image_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        """
        self.image_size = np.asarray(image_size)
        self.set_image(image)
        # self.patch_size = np.asarray(patch_size)
        # self.set_weights(self.patch_size, None)
        self.weight_map = np.zeros(self.image_size, dtype=np.uint8)

    def set_image(self, image):
        if image is not None:
            self.image = image
        else:
            # self.image = np.zeros(self.image_size, dtype=np.float32)
            self.image = np.zeros(self.image_size, dtype=np.float32)

    # def set_weights(self, size, weights):
    # self.weight_map = np.zeros(self.image_size, dtype=np.uint8)
    # self.weight_patch = np.ones(size, dtype=np.uint8)

    def append(self, patch, patch_indices):
        """
        Appends a patch to the image.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_indices: The patch indices in the format of (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...).
        """
        slices = self.get_slices(self.image, patch_indices)
        weight_patch = np.ones(patch.shape, dtype=np.uint8)
        self.image[slicer(self.image, slices)] += patch.astype(self.image.dtype) * weight_patch.astype(self.image.dtype)
        weight_map_patch = self.weight_map[slicer(self.weight_map, patch_indices)]
        # weight_map_patch[...] += self.weight_patch
        weight_map_patch[...] += weight_patch

    def get_output(
        self, patch_size=False, inplace=False
    ):  # Add inplace option with image instead of sel.image for result storage
        """
        Computes and returns the final aggregated output image based on all provided patches. The content of overlapping patches is averaged.
        In case the image is a larger-than-RAM image and if the image format supports chunk-loading then defining patch_size enables a chunk-based computation.
        :param patch_size: The shape of patch that should be used for aggregation without batch and channel dimensions. Only required if a chunk-based computation is desired.
        The patch size can be different to the patch size of any previous patchification processes like that of the GridSampler.
        :return: The final aggregated output image.
        """
        if isinstance(patch_size, bool) and not patch_size:
            self.image = self.image / self.weight_map.astype(self.image.dtype)
            self.image = np.nan_to_num(self.image)
        else:
            sampler = GridSampler(image_size=self.image_size, patch_size=patch_size)
            for patch_indices in sampler:
                slices = self.get_slices(self.image, patch_indices)
                image_patch = self.image[slicer(self.image, slices)]
                weight_map_patch = self.weight_map[slicer(self.weight_map, patch_indices)]
                image_patch = image_patch / weight_map_patch.astype(self.image.dtype)
                image_patch = np.nan_to_num(image_patch)
                self.image[slicer(self.image, slices)] = image_patch
        return self.image

    def get_slices(self, image, patch_indices):
        num_image_dims = len(image.shape) - len(self.image_size)
        slices = [None] * num_image_dims
        slices.extend([index_pair.tolist() for index_pair in patch_indices])
        return slices


class WeightedAggregator(Aggregator):
    def __init__(self, image=None, image_size=None, patch_size=None, weights="gaussian"):
        """
        Weighted aggregator to assemble an image with continuous content from patches. The content of overlapping patches is gaussian-weighted by default.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param image: A numpy-style zero-initialized image (Numpy, Zarr, Dask, ...) of a continuous data type. If none then a zero-initialized Numpy image of data type np.float32 is created internally.
        :param image_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :weights: A weight map of size patch_size that should be used for weighting the importance of each value in a patch. Default is a gaussian weight map.
        """
        super().__init__(image, image_size)
        self.patch_size = np.asarray(patch_size)
        self.set_weights(self.patch_size, weights)

    def set_weights(self, size, weights):
        if weights == "gaussian":
            self.weight_map = np.zeros(self.image_size, dtype=np.uint16)
            self.weight_patch = self.gaussian_weights(size, center_value=255, dtype=np.uint16)
        else:
            self.weight_patch = weights

    def gaussian_weights(self, size, center_value, dtype):
        sigma_scale = 1.0 / 8
        sigmas = size * sigma_scale
        center_coords = size // 2
        tmp = np.zeros(size)
        tmp[tuple(center_coords)] = 1
        gaussian_weights = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
        # gaussian_weights = np.rint(normalize(gaussian_weights) * center_value).astype(dtype)
        # gaussian_weights += 1
        gaussian_weights[gaussian_weights == 0] = np.min(gaussian_weights[gaussian_weights != 0])
        # import SimpleITK as sitk
        # sitk.WriteImage(sitk.GetImageFromArray(gaussian_weights), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/tmp.nii.gz")
        return gaussian_weights


class WeightedSoftmaxAggregator(WeightedAggregator):
    def __init__(self, image=None, image_size=None, patch_size=None, weights="gaussian", low_memory_mode=False):
        """
        Weighted aggregator to assemble an image with continuous content from patches. Returns the maximum class at each position of the image. The content of overlapping patches is gaussian-weighted by default.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param image: A numpy-style zero-initialized image (Numpy, Zarr, Dask, ...) of a continuous data type. If none then a zero-initialized Numpy image of data type np.float32 is created internally.
        :param image_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param weights: A weight map of size patch_size that should be used for weighting the importance of each value in a patch. Default is a gaussian weight map.
        :param low_memory_mode: Reduces memory consumption by more than 50% in comparison to the normal WeightedAggregator and Aggregator. However, the prediction quality is slightly reduced.
        """
        self.low_memory_mode = low_memory_mode
        super().__init__(image, image_size, patch_size, weights)

    def set_weights(self, size, weights):
        if weights == "gaussian" and not self.low_memory_mode:
            self.weight_patch = self.gaussian_weights(size, center_value=255, dtype=np.uint16)
        elif weights == "gaussian" and self.low_memory_mode:
            self.weight_patch = self.gaussian_weights(size, center_value=63, dtype=np.uint8)
        else:
            self.weight_patch = weights

    def append(self, patch, patch_indices):
        """
        Appends a patch to the image.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_indices: The patch indices in the format of (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...).
        """

        if self.low_memory_mode:
            patch = np.rint(patch * 500).astype(self.image.dtype)  # 255
        slices = self.get_slices(self.image, patch_indices)
        self.image[slicer(self.image, slices)] += patch.astype(self.image.dtype) * self.weight_patch.astype(
            self.image.dtype
        )

    def get_output(self, patch_size=False, output=None):
        """
        Computes and returns the final aggregated output image based on all provided patches. The content of overlapping patches is averaged.
        In case the image is a larger-than-RAM image and if the image format supports chunk-loading then defining patch_size enables a chunk-based computation.
        :param patch_size: The shape of patch that should be used for aggregation without batch and channel dimensions. Only required if a chunk-based computation is desired.
        The patch size can be different to the patch size of any previous patchification processes like that of the GridSampler.
        :return: The final aggregated output image.
        """
        if output is None:
            output = np.zeros(self.image.shape, dtype=np.uint16)

        self.image = np.array(self.image)
        if isinstance(patch_size, bool) and not patch_size:
            output = self.image.argmax(axis=0)
        else:
            sampler = GridSampler(image_size=self.image_size, patch_size=patch_size)
            for patch_indices in sampler:
                slices = self.get_slices(self.image, patch_indices)
                image_patch = self.image[slicer(self.image, slices)]
                image_patch = image_patch.argmax(axis=0)
                output[slicer(output, slices)[1:]] = image_patch
        return output


class ChunkedWeightedSoftmaxAggregator(WeightedSoftmaxAggregator):
    def __init__(
        self, image=None, image_size=None, patch_size=None, patch_overlap=None, chunk_size=None, weights="gaussian"
    ):
        """
        Weighted aggregator to assemble an image with continuous content from patches. Returns the maximum class at each position of the image. The content of overlapping patches is gaussian-weighted by default.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param image: A numpy-style zero-initialized image (Numpy, Zarr, Dask, ...) of a continuous data type. If none then a zero-initialized Numpy image of data type np.float32 is created internally.
        :param image_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param weights: A weight map of size patch_size that should be used for weighting the importance of each value in a patch. Default is a gaussian weight map.
        :param low_memory_mode: Reduces memory consumption by more than 50% in comparison to the normal WeightedAggregator and Aggregator. However, the prediction quality is slightly reduced.
        """
        super().__init__(image, image_size, patch_size, weights, low_memory_mode=False)
        self.patch_overlap = patch_overlap
        self.chunk_size = chunk_size
        self.compute_indices()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def compute_indices(self):
        self.grid_sampler = GridSampler(
            image_size=self.image_size, patch_size=self.chunk_size, patch_overlap=self.chunk_size - self.patch_size
        )
        
        self.chunk_sampler = []
        self.chunk_sampler_offset = []
        self.chunk_indices = list(self.grid_sampler)
        self.chunk_patches_dicts = defaultdict(dict)

        for chunk_id, chunk_indices in enumerate(self.chunk_indices):
            chunk_size = copy.copy(chunk_indices[:, 1] - chunk_indices[:, 0])
            sampler = BasicGridSampler(
                image_size=chunk_size, patch_size=self.patch_size, patch_overlap=self.patch_overlap
            )
            sampler_offset = copy.copy(chunk_indices[:, 0])
            self.chunk_sampler.append(sampler)
            self.chunk_sampler_offset.append(sampler_offset)

            # The edges of non-border chunks need to be cropped as they have no overlap patch within the chunk
            for axis in range(len(self.image_size)):
                if 0 < chunk_indices[axis][0]:
                    chunk_indices[axis][0] += int(self.patch_size[axis] // 2)
                if chunk_indices[axis][1] < self.image_size[axis]:
                    chunk_indices[axis][1] -= int(self.patch_size[axis] // 2)

            for patch_indices in sampler:
                patch_indices_key = patch_indices + sampler_offset.reshape(-1, 1)
                patch_indices_key = patch_indices_key.astype(np.int64).tobytes()
                self.chunk_patches_dicts[chunk_id][patch_indices_key] = None

    def append(self, patch, patch_indices):
        """
        Appends a patch to the image.
        :param patch: The patch data in a numpy-style format (Numpy, Zarr, Dask, ...) with or without batch and channel dimensions.
        :param patch_indices: The patch indices in the format of (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin, ...).
        """

        patch_indices, chunk_id = patch_indices
        # patch_indices, chunk_id = patch_indices.numpy(), chunk_id.item()
        # import IPython

        # IPython.embed()
        # print(chunk_id)
        patch_indices_key = patch_indices.astype(np.int64).tobytes()
        if patch_indices_key not in self.chunk_patches_dicts[chunk_id]:
            unhashed_keys = [
                np.array(np.frombuffer(key, dtype=np.int64), dtype=int).reshape(-1, 2)
                for key in self.chunk_patches_dicts[chunk_id].keys()
            ]
            raise RuntimeError(
                "patch_indices_key not in self.chunk_patches_dicts[chunk_id]! patch_indices: {}. Offset for chunk_id {}"
                " is{}. unhashed_keys: {}".format(
                    patch_indices, chunk_id, self.chunk_sampler_offset[chunk_id], unhashed_keys
                )
            )
        self.chunk_patches_dicts[chunk_id][patch_indices_key] = patch
        if self.is_chunk_complete(chunk_id):
            # print("chunk_id: ", chunk_id)
            # self.process_chunk(chunk_id)
            self.executor.submit(self.process_chunk, chunk_id)

    def is_chunk_complete(self, chunk_id):
        # Check if all self.chunk_patches_dicts[chunk_id] values are not None
        for value in self.chunk_patches_dicts[chunk_id].values():
            if value is None:
                return False
        return True

    def process_chunk(self, chunk_id):
        # print("Test", flush=True)
        # If they are all not None, create a softmax array of size chunk_size with number classes as channels
        num_channels = self.chunk_patches_dicts[chunk_id][list(self.chunk_patches_dicts[chunk_id].keys())[0]].shape[0]
        image_chunk_softmax = np.zeros((num_channels, *self.chunk_size), dtype=np.float32)
        # Weight each patch during insertion
        sampler_offset = self.chunk_sampler_offset[chunk_id].reshape(-1, 1)
        for patch_indices_key, patch in self.chunk_patches_dicts[chunk_id].items():
            patch_indices = np.array(np.frombuffer(patch_indices_key, dtype=np.int64), dtype=int).reshape(-1, 2)
            patch_indices -= sampler_offset
            slices = self.get_slices(image_chunk_softmax, patch_indices)
            image_chunk_softmax[slicer(image_chunk_softmax, slices)] += patch.astype(
                image_chunk_softmax.dtype
            ) * self.weight_patch.astype(image_chunk_softmax.dtype)
        # Argmax the softmax chunk
        image_chunk = image_chunk_softmax.argmax(axis=0).astype(np.uint16)
        # Crop the chunk
        crop_indices = self.chunk_indices[chunk_id] - sampler_offset
        image_chunk = image_chunk[slicer(image_chunk, crop_indices)]
        # Write the chunk into the global image
        crop_indices = self.chunk_indices[chunk_id]
        self.image[slicer(self.image, crop_indices)] = image_chunk
        # Set all self.chunk_patches_dicts[chunk_id] values to None
        for key in self.chunk_patches_dicts[chunk_id].keys():
            self.chunk_patches_dicts[chunk_id][key] = None
        gc.collect()
        # print("Finished saving chunk ", chunk_id, flush=True)

    def get_output(self, patch_size=False, output=None):
        """
        Computes and returns the final aggregated output image based on all provided patches. The content of overlapping patches is averaged.
        In case the image is a larger-than-RAM image and if the image format supports chunk-loading then defining patch_size enables a chunk-based computation.
        :param patch_size: The shape of patch that should be used for aggregation without batch and channel dimensions. Only required if a chunk-based computation is desired.
        The patch size can be different to the patch size of any previous patchification processes like that of the GridSampler.
        :return: The final aggregated output image.
        """
        return self.image


class ResizeChunkedWeightedSoftmaxAggregator(ChunkedWeightedSoftmaxAggregator):
    def __init__(
        self,
        image=None,
        image_size=None,
        patch_size=None,
        patch_overlap=None,
        chunk_size=None,
        weights="gaussian",
        spacing=None,
    ):
        """
        Weighted aggregator to assemble an image with continuous content from patches. Returns the maximum class at each position of the image. The content of overlapping patches is gaussian-weighted by default.
        Can be used in conjunction with the GridSampler during inference to assemble the image-predictions from the patch-predictions.
        Is mainly intended to be used with the GridSampler, but can technically be used with any sampler.
        :param image: A numpy-style zero-initialized image (Numpy, Zarr, Dask, ...) of a continuous data type. If none then a zero-initialized Numpy image of data type np.float32 is created internally.
        :param image_size: The image size that was used for patchification without batch and channel dimensions. Always required.
        :param patch_size: The shape of the patch without batch and channel dimensions. Always required.
        :param weights: A weight map of size patch_size that should be used for weighting the importance of each value in a patch. Default is a gaussian weight map.
        :param low_memory_mode: Reduces memory consumption by more than 50% in comparison to the normal WeightedAggregator and Aggregator. However, the prediction quality is slightly reduced.
        """
        super().__init__(image, image_size, patch_size, patch_overlap, chunk_size, weights)
        self.patch_overlap = patch_overlap
        self.chunk_size = np.asarray(chunk_size)
        self.spacing = spacing
        # self.source_size = np.asarray(source_size)
        self.compute_indices()
        # self.set_weights(self.source_size, weights)
        # self.size_conversion_factor = size_conversion_factor

    def process_chunk(self, chunk_id):
        # print("chunk_id: ", chunk_id)
        # If they are all not None, create a softmax array of size chunk_size with number classes as channels
        patch_shape = self.chunk_patches_dicts[chunk_id][list(self.chunk_patches_dicts[chunk_id].keys())[0]].shape
        num_channels = patch_shape[0]
        image_chunk_softmax = np.zeros((num_channels, *self.chunk_size), dtype=np.float32)
        # Weight each patch during insertion
        sampler_offset = self.chunk_sampler_offset[chunk_id].reshape(-1, 1)
        for patch_indices_key, patch in self.chunk_patches_dicts[chunk_id].items():
            patch_indices = np.array(np.frombuffer(patch_indices_key, dtype=np.uint16), dtype=int).reshape(-1, 2)
            patch_indices -= sampler_offset
            # patch_indices = np.rint(patch_indices * size_conversion_factor).astype(np.int32)
            slices = self.get_slices(image_chunk_softmax, patch_indices)
            image_chunk_softmax[slicer(image_chunk_softmax, slices)] += patch.astype(
                image_chunk_softmax.dtype
            ) * self.weight_patch.astype(image_chunk_softmax.dtype)
        # Argmax the softmax chunk
        image_chunk = image_chunk_softmax.argmax(axis=0).astype(np.uint16)
        crop_indices = self.chunk_indices[chunk_id] - sampler_offset
        image_chunk = image_chunk[slicer(image_chunk, crop_indices)]

        # Write the chunk into the global image
        crop_indices = self.chunk_indices[chunk_id]
        # print("crop_indices 2: ", crop_indices)
        self.image[slicer(self.image, crop_indices)] = image_chunk
        # Set all self.chunk_patches_dicts[chunk_id] values to None
        for key in self.chunk_patches_dicts[chunk_id].keys():
            self.chunk_patches_dicts[chunk_id][key] = None




if __name__ == "__main__":
    from sampler import ChunkedGridSampler
    import zarr
    from tqdm import tqdm

    image_size = (500, 500, 500)
    patch_size = (128, 128, 128)
    patch_overlap = (64, 64, 64)
    chunk_size = (384, 384, 384)

    result = zarr.open("tmp.zarr", mode="w", shape=image_size, chunks=chunk_size, dtype=np.uint8)

    grid_sampler = ChunkedGridSampler(
        image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size
    )
    aggregrator = ChunkedWeightedSoftmaxAggregator(
        image=result, image_size=image_size, patch_size=patch_size, patch_overlap=patch_overlap, chunk_size=chunk_size
    )

    print(len(grid_sampler))

    counter = 0
    for i, indices in enumerate(tqdm(grid_sampler)):
        # print("Iteration: {}, indices: {}".format(i, indices))
        patch = np.zeros((8, *patch_size), dtype=np.float32)
        chunk_id = indices[1]
        patch[chunk_id, ...] = 1
        aggregrator.append(patch, indices)
    print("")
