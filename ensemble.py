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


from omegaconf import OmegaConf
import glob
import os

import torch
import torch.nn as nn


# Lars hrnet ocr ms model
from hrnet_ocr_ms.hrnet_ocr_ms import get_seg_model as get_hrnet_ocr_ms_model


def load_model(path_ckpt):
    from torchvision.models import efficientnet_v2_m
    import torch

    model = efficientnet_v2_m()
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=6, bias=True)
    pretrained_dict = torch.load(path_ckpt, map_location={"cuda:0": "cpu"})

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict["state_dict"]
    pretrained_dict = {
        k.replace("model.", "").replace("module.", "").replace("backbone.", ""): v
        for k, v in pretrained_dict.items()
    }

    model.load_state_dict(pretrained_dict)
    model.eval().to("cuda")

    print("Done")
    return model


def load_models(ckpt_dir):
    checkpoint_paths = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    print("Loading {} models for ensembling".format(len(checkpoint_paths)))
    models = []
    for p in checkpoint_paths:
        model = load_model(p)
        models.append(model)

    return models


class Ensemble(nn.Module):
    def __init__(self, ckpt_dir):
        super(Ensemble, self).__init__()

        models = load_models(ckpt_dir)
        self.models = models

    def forward(self, input_tensor):
        out = None
        for m in self.models:
            if out is None:
                y = m(input_tensor)

                input_tensor_aug_90 = torch.rot90(input_tensor, k=1, dims= (2,3))
                input_tensor_aug_180 = torch.rot90(input_tensor, k=2, dims= (2,3))
                input_tensor_aug_270 = torch.rot90(input_tensor, k=3, dims= (2,3))

                y_aug_90 = m(input_tensor_aug_90)
                y_aug_180 = m(input_tensor_aug_180)
                y_aug_270 = m(input_tensor_aug_270)

                y = 0.25 * y + 0.25 * y_aug_90 + 0.25 * y_aug_180 + 0.25 * y_aug_270

                out = torch.stack([i.unsqueeze(dim=1).unsqueeze(dim=2).expand(6,512,512) for i in y])
            else:
                y = m(input_tensor)

                input_tensor_aug_90 = torch.rot90(input_tensor, k=1, dims= (2,3))
                input_tensor_aug_180 = torch.rot90(input_tensor, k=2, dims= (2,3))
                input_tensor_aug_270 = torch.rot90(input_tensor, k=3, dims= (2,3))

                y_aug_90 = m(input_tensor_aug_90)
                y_aug_180 = m(input_tensor_aug_180)
                y_aug_270 = m(input_tensor_aug_270)

                y = 0.25 * y + 0.25 * y_aug_90 + 0.25 * y_aug_180 + 0.25 * y_aug_270

                out += torch.stack([i.unsqueeze(dim=1).unsqueeze(dim=2).expand(6,512,512) for i in y])

        out_avg = out / len(self.models)

        return out_avg
