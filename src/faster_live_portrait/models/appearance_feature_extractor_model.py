# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: motion_extractor_model.py
import pdb
import numpy as np
from .base_model import BaseModel
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict


class AppearanceFeatureExtractorModel(BaseModel):
    """
    AppearanceFeatureExtractorModel
    """

    def __init__(self, **kwargs):
        super(AppearanceFeatureExtractorModel, self).__init__(**kwargs)
        self.predict_type = kwargs.get("predict_type", "trt")
        print(self.predict_type)

    def input_process(self, *data):
        img = data[0].astype(np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[None]

    def output_process(self, *data):
        # Return the first element directly (now potentially a tensor)
        return data[0]

    def predict_trt(self, *data):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            input_data = data[i]
            if not isinstance(input_data, torch.Tensor):
                 input_data = torch.from_numpy(input_data).to(device=self.device,
                                                               dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            # Ensure tensor is on the correct device (might already be)
            feed_dict[inp['name']] = input_data.to(device=self.device)

        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            # Keep the output as a GPU tensor
            outs.append(preds_dict[out["name"]])
        nvtx.range_pop()
        # Return list of GPU tensors
        return outs

    def predict(self, *data):
        data = self.input_process(*data)
        if self.predict_type == "trt":
            preds = self.predict_trt(data)
        else:
            preds = self.predictor.predict(data)
        outputs = self.output_process(*preds)
        return outputs
