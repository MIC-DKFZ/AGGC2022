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




import os
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from tqdm import tqdm
import gc
from data.data_sampler import get_patch_handler
from data.test_augs import get_norm_only
from omegaconf import OmegaConf
from ensemble import Ensemble

from hrnet.hrnet import get_seg_model
from hrnet_ocr_ms.hrnet_ocr_ms import get_seg_model as get_hrnet_ocr_ms_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define test inference")
    parser.add_argument("--model", type=str, help="Name of the model", default="hrnet_ocr_ms")
    parser.add_argument(
        "--weights",
        type=str,
        default="./checkpoint/best_epoch_11__wF1_0.6670.ckpt",
        help=(
            "path to the checkpoint file, if --ensemble is used this needs to be the directory containing all"
            " checkpoints"
        ),
    )
    parser.add_argument(
        "--config_file",
        default="./checkpoint/hparams_hrnet_ocr_ms.yaml",
        help="Path to the model config file",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing the WSIs. If a path to a single file is given only this file will be predicted",
    )
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--test_aug", default='only_norm', help="only_norm / None")
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="will do ensembling with all models in the directory passed with --weights",
    )
    parser.add_argument("--debug", action="store_true", help="will only predict the first WSI in the directory")
    parser.add_argument("--reverse_WSI_list", action="store_true", help="starts processing at the end of the WSI list")

    args = parser.parse_args()
    model_name = args.model
    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    chunked = args.chunked
    bs = args.batch_size
    num_workers = args.num_workers
    do_ensembling = args.ensemble

    zarr_dir = os.path.join(save_dir, "zarr_files")
    if chunked:
        os.makedirs(zarr_dir, exist_ok=True)

    # load data
    if data_dir.endswith(".tiff"):
        WSIs = [data_dir]
    else:
        WSIs = glob.glob(os.path.join(data_dir, "*.tiff"))

    # load model
    weights_path = args.weights
    config_file = args.config_file
    if model_name == "hrnet":
        # load Lars model
        # path_hparams = os.path.join(path, "hparams.yaml")
        # path_ckpt = os.path.join(path, "best_epoch_11__wF1_0.6202.ckpt")

        print("Loading trained model...")
        config = OmegaConf.load(config_file)
        model = get_seg_model(config)
        model.load_weights(weights_path)
        model.eval().to("cuda")
        print("Done")

    elif model_name == "hrnet_ocr_ms":
        if do_ensembling:
            # path_hparams = os.path.join(path, "hparams.yaml")
            # ensemble_dir = os.path.join(path, "checkpoint_2022-07-13_11-29-46")
            model = Ensemble(weights_path)
            model.eval().to("cuda")
        else:
            # load Lars model
            # path_hparams = os.path.join(path, "hparams.yaml")
            # path_ckpt = os.path.join(path, "checkpoint_2022-07-11_15-19-36/best_epoch_11__wF1_0.6670.ckpt")

            print("Loading trained model...")
            config = OmegaConf.load(config_file)
            model = get_hrnet_ocr_ms_model(config)
            model.load_weights(weights_path)
            model.eval().to("cuda")
            print("Done")

    else:
        raise NotImplementedError

    # get test transforms
    if args.test_aug:
        if args.test_aug == "only_norm":
            test_aug = get_norm_only()
        else:
            raise NotImplementedError
    else:
        test_aug = None

    # run inference
    sliding_window_overlap = 256
    if args.reverse_WSI_list:
        WSIs.reverse()
    for path in WSIs:

        save_file_name = os.path.join(save_dir, path.split("/")[-1])
        save_file_name=save_file_name.replace('.tiff','.tif')
        if os.path.isfile(save_file_name):
            continue

        """if "Train_20" not in path:
            continue"""

        patch_loader, softmax_aggregator = get_patch_handler(
            path,
            patch_size=512,
            sliding_window_overlap=sliding_window_overlap,
            batch_size=bs,
            num_workers=num_workers,
            transform=test_aug,
            chunked=chunked,
            zarr_path=os.path.join(zarr_dir, path.split("/")[-1].replace(".tiff", ".zarr")),
        )

        print("Start Inference on {}".format(path.split("/")[-1]))
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader, disable=False):

                input_tensor = patches_batch[0].to("cuda")
                if chunked:
                    locations = patches_batch[1]
                else:
                    locations = patches_batch[1].numpy()

                if do_ensembling:
                    logits = model(input_tensor)
                else:
                    if model_name == "efnet_Lukas":
                        y = model(input_tensor)

                        input_tensor_aug_90 = torch.rot90(input_tensor, k=1, dims= (2,3))
                        input_tensor_aug_180 = torch.rot90(input_tensor, k=2, dims= (2,3))
                        input_tensor_aug_270 = torch.rot90(input_tensor, k=3, dims= (2,3))

                        y_aug_90 = model(input_tensor_aug_90)
                        y_aug_180 = model(input_tensor_aug_180)
                        y_aug_270 = model(input_tensor_aug_270)

                        y = 0.25 * y + 0.25 * y_aug_90 + 0.25 * y_aug_180 + 0.25 * y_aug_270

                        logits = torch.stack([i.unsqueeze(dim=1).unsqueeze(dim=2).expand(6,512,512) for i in y])
                    else:
                        logits = model(input_tensor)["out"]
                logits = logits.to("cpu").numpy()

                if chunked:
                    [
                        softmax_aggregator.append(p, (locations[0][i].numpy(), locations[1][i].item()))
                        for i, p in enumerate(logits)
                    ]
                else:
                    [softmax_aggregator.append(p, locations[i]) for i, p in enumerate(logits)]
                gc.collect()

            print("Inference done")

            output_tensor = softmax_aggregator.get_output()
            print("Aggregation finished")
            if chunked:
                output_tensor = np.asarray(output_tensor)
            print("Prediction shape", output_tensor.shape)

            # downscale prediction by factor 10
            new_size = (
                np.ceil(output_tensor.shape[0] / 10).astype(np.uint16),
                np.ceil(output_tensor.shape[1] / 10).astype(np.uint16),
            )

            output_tensor = (
                F.interpolate(torch.from_numpy(output_tensor).byte().unsqueeze(dim=0).unsqueeze(dim=0), size=new_size)
                .squeeze()
                .numpy()
            )

            print("final downscaled shape", output_tensor.shape)

            # save as tiff
            tifffile.imwrite(save_file_name, output_tensor)

            print("Prediction saved")

            if args.debug:
                break
