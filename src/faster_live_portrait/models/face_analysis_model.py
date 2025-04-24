# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: face_analysis_model.py
import pdb

import numpy as np
from insightface.app.common import Face
import cv2
from .predictor import get_predictor
from ..utils import face_align
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict


def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]),
                      reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2] + face['bbox'][0]) / 2 - face_center[0]) ** 2 + (
                (face['bbox'][3] + face['bbox'][1]) / 2 - face_center[1]) ** 2) ** 0.5)
    return faces


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceAnalysisModel:
    def __init__(self, **kwargs):
        self.model_paths = kwargs.get("model_path", [])
        self.predict_type = kwargs.get("predict_type", "trt")
        self.device = torch.cuda.current_device()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        grid_sample_plugin_path=kwargs.get("grid_sample_plugin_path", None)

        assert self.model_paths
        self.face_det = get_predictor(predict_type=self.predict_type, model_path=self.model_paths[0],
                                       grid_sample_plugin_path=grid_sample_plugin_path)
        self.face_det.input_spec()
        self.face_det.output_spec()
        self.face_pose = get_predictor(predict_type=self.predict_type, model_path=self.model_paths[1],
                                       grid_sample_plugin_path=grid_sample_plugin_path)
        self.face_pose.input_spec()
        self.face_pose.output_spec()

        # face det
        self.input_mean = 127.5
        self.input_std = 128.0
        # print(self.output_names)
        # assert len(outputs)==10 or len(outputs)==15
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self.input_size = (512, 512)
        if len(self.face_det.outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(self.face_det.outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(self.face_det.outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(self.face_det.outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

        self.lmk_dim = 2
        self.lmk_num = 212 // self.lmk_dim

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect_face(self, *data):
        img = data[0]  # BGR mode
        det_img = img
        input_height_orig = img.shape[0]
        input_width_orig = img.shape[1]

        det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        det_img_transposed = np.transpose(det_img_rgb, (2, 0, 1))
        det_img_cont = np.ascontiguousarray(det_img_transposed)

        input_tensor_gpu = torch.from_numpy(det_img_cont).to(self.device)

        mean_tensor = torch.tensor([self.input_mean], device=self.device).view(1, -1, 1, 1)
        std_tensor = torch.tensor([self.input_std], device=self.device).view(1, -1, 1, 1)
        input_tensor_gpu = input_tensor_gpu.unsqueeze(0)
        input_tensor_normalized = (input_tensor_gpu - mean_tensor) / std_tensor

        if self.predict_type == "trt":
            inp = self.face_det.inputs[0]
            target_dtype = numpy_to_torch_dtype_dict[inp['dtype']]
            input_tensor = input_tensor_normalized.to(target_dtype)
        else:
            input_tensor = input_tensor_normalized.float()

        if self.predict_type == "trt":
            feed_dict = {self.face_det.inputs[0]['name']: input_tensor}
            preds_dict = self.face_det.predict(feed_dict, self.cudaStream)
            output_keys = ["448", "471", "494", "451", "474", "497"]
            if self.use_kps:
                output_keys.extend(["454", "477", "500"])
            if not all(key in preds_dict for key in output_keys):
                 raise KeyError(f"Missing expected output keys from TRT engine. Expected: {output_keys}, Got: {list(preds_dict.keys())}")
            outputs_np = [preds_dict[key].cpu().numpy() for key in output_keys]
        else:
            outputs_np = self.face_det.predict(input_tensor)

        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_height = det_img_cont.shape[1]
        input_width = det_img_cont.shape[2]
        fmc = self.fmc

        for idx, stride in enumerate(self._feat_stride_fpn):
            if idx >= len(outputs_np) or idx + fmc >= len(outputs_np):
                 print(f"Warning: Output index out of bounds at stride {stride}. Skipping.")
                 continue
            if self.use_kps and idx + fmc * 2 >= len(outputs_np):
                 print(f"Warning: KPS output index out of bounds at stride {stride}. Skipping KPS processing for this stride.")
                 continue

            scores = outputs_np[idx]
            bbox_preds = outputs_np[idx + fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = None
            if self.use_kps:
                kps_preds = outputs_np[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            expected_score_elements = K * self._num_anchors
            expected_bbox_elements = K * self._num_anchors * 4
            expected_kps_elements = K * self._num_anchors * 10

            if scores.size != expected_score_elements:
                print(f"Warning: Score shape mismatch at stride {stride}. Expected {expected_score_elements}, Got {scores.size}. Reshaping.")
                try:
                     scores = scores.reshape(-1)
                except ValueError:
                     print("Error: Cannot reshape scores. Skipping stride.")
                     continue
            if bbox_preds.size != expected_bbox_elements:
                 print(f"Warning: Bbox shape mismatch at stride {stride}. Expected {expected_bbox_elements}, Got {bbox_preds.size}. Reshaping.")
                 try:
                     bbox_preds = bbox_preds.reshape(-1, 4)
                 except ValueError:
                     print("Error: Cannot reshape bbox_preds. Skipping stride.")
                     continue
            if self.use_kps and kps_preds is not None and kps_preds.size != expected_kps_elements:
                 print(f"Warning: KPS shape mismatch at stride {stride}. Expected {expected_kps_elements}, Got {kps_preds.size}. Reshaping.")
                 try:
                     kps_preds = kps_preds.reshape(-1, 10)
                 except ValueError:
                     print("Error: Cannot reshape kps_preds. Skipping KPS this stride.")
                     kps_preds = None

            anchor_centers = anchor_centers.reshape((-1, 2))
            pos_inds = np.where(scores >= self.det_thresh)[0]
            if len(pos_inds) == 0:
                continue

            if np.max(pos_inds) >= anchor_centers.shape[0]:
                 print(f"Warning: pos_inds index out of bounds for anchor_centers at stride {stride}. Skipping.")
                 continue

            bboxes = distance2bbox(anchor_centers[pos_inds], bbox_preds[pos_inds])
            pos_scores = scores[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(bboxes)

            if self.use_kps and kps_preds is not None:
                 if np.max(pos_inds) >= kps_preds.shape[0]:
                      print(f"Warning: pos_inds index out of bounds for kps_preds at stride {stride}. Skipping KPS.")
                 else:
                     kpss = distance2kps(anchor_centers[pos_inds], kps_preds[pos_inds])
                     kpss = kpss.reshape((kpss.shape[0], -1, 2))
                     kpss_list.append(kpss)

        if not scores_list:
             det = np.empty((0, 5), dtype=np.float32)
             kpss_final = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
             return det, kpss_final

        # Ensure scores is 1D after concatenation
        scores = np.concatenate(scores_list).flatten()
        bboxes = np.vstack(bboxes_list)

        det_scale = 1
        bboxes_scaled = bboxes / det_scale

        if self.use_kps and kpss_list:
            kpss_stacked = np.vstack(kpss_list)
            kpss_scaled = kpss_stacked / det_scale
        else:
            kpss_scaled = None

        pre_det = np.hstack((bboxes_scaled, scores[:, np.newaxis])).astype(np.float32, copy=False)

        order = scores.argsort()[::-1]
        pre_det_ordered = pre_det[order, :]
        keep = self.nms(pre_det_ordered)
        det = pre_det_ordered[keep, :]

        kpss_final = None
        if self.use_kps and kpss_scaled is not None:
            if kpss_scaled.shape[0] == scores.shape[0]:
                 kpss_ordered = kpss_scaled[order, :, :]
                 kpss_final = kpss_ordered[keep, :, :]
            else:
                 print(f"Warning: Mismatch between KPS count ({kpss_scaled.shape[0]}) and score count ({scores.shape[0]}) before NMS. KPS data might be incorrect.")
                 kpss_final = None
        return det, kpss_final

    def estimate_face_pose(self, *data):
        """
        检测脸部关键点
        :param data:
        :return:
        """
        img, face = data
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        input_size = (192, 192)
        _scale = input_size[0] / (max(w, h) * 1.5)
        aimg, M = face_align.transform(img, center, input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
        aimg = np.transpose(aimg, (2, 0, 1))
        if self.predict_type == "trt":
            feed_dict = {}
            inp = self.face_pose.inputs[0]
            det_img_torch = torch.from_numpy(aimg[None]).to(device=self.device,
                                                            dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            feed_dict[inp['name']] = det_img_torch
            preds_dict = self.face_pose.predict(feed_dict, self.cudaStream)
            outs = []
            for i, out in enumerate(self.face_pose.outputs):
                outs.append(preds_dict[out["name"]].cpu().numpy())
            pred = outs[0]
        else:
            pred = self.face_pose.predict(aimg[None])[0]
        pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = face_align.trans_points(pred, IM)
        face["landmark"] = pred
        return pred

    def predict(self, *data, **kwargs):
        img = data[0]  # Assume first argument is the image

        bboxes, kpss = self.detect_face(img)

        if bboxes.shape[0] == 0:
            return []

        # Prepare batch for pose estimation
        face_batch = []
        transforms = []
        input_size_pose = (192, 192)
        ret = [] # Keep track of original Face objects

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if self.use_kps and kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            ret.append(face)

            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = input_size_pose[0] / (max(w, h) * 1.5)
            aimg, M = face_align.transform(img, center, input_size_pose[0], _scale, rotate)
            aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
            aimg = np.transpose(aimg, (2, 0, 1))
            face_batch.append(aimg)
            transforms.append(cv2.invertAffineTransform(M))

        if not face_batch:
            return []

        face_batch_np = np.stack(face_batch)

        if self.predict_type == "trt":
            feed_dict = {}
            inp = self.face_pose.inputs[0]
            face_batch_torch = torch.from_numpy(face_batch_np).to(device=self.device,
                                                                   dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            feed_dict[inp['name']] = face_batch_torch
            preds_dict = self.face_pose.predict(feed_dict, self.cudaStream)
            landmarks_batch = preds_dict[self.face_pose.outputs[0]["name"]].cpu().numpy()
        else:
            landmarks_batch = self.face_pose.predict(face_batch_np)[0]

        # Postprocess landmarks for each face
        for i in range(landmarks_batch.shape[0]):
            pred = landmarks_batch[i]
            pred = pred.reshape((-1, 2))
            if self.lmk_num < pred.shape[0]:
                pred = pred[self.lmk_num * -1:, :]
            pred[:, 0:2] += 1
            pred[:, 0:2] *= (input_size_pose[0] // 2)

            IM = transforms[i]
            pred = face_align.trans_points(pred, IM)
            ret[i]["landmark"] = pred

        # Sort faces and return landmarks
        ret = sort_by_direction(ret, 'large-small', None)
        outs = [x.landmark for x in ret]

        return outs

    def __del__(self):
        del self.face_det
        del self.face_pose
