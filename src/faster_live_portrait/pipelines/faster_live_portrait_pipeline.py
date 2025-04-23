# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: faster_live_portrait_pipeline.py

# TODO: cleanup the code further, it was written for source video and driving image -- we need source image and driving image

import copy
import os.path
import traceback
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import torch
import logging
import torch.nn.functional as F

from .. import models
from ..utils.crop import crop_image, parse_bbox_from_landmark, crop_image_by_bbox, paste_back, paste_back_pytorch
from ..utils.utils import resize_to_limit, prepare_paste_back, get_rotation_matrix, calc_lip_close_ratio, \
    calc_eye_close_ratio, transform_keypoint, concat_feat
from ..utils import utils
# from ..utils.animal_landmark_runner import XPoseRunner
from ..utils.utils import make_abs_path
from .profile import prof


class FasterLivePortraitPipeline:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_dict = {}
        self.init(**kwargs)

    def init(self, **kwargs):
        self.init_vars(**kwargs)
        self.init_models(**kwargs)

    def update_cfg(self, args_user):
        update_ret = False
        for key in args_user:
            if key in self.cfg.infer_params:
                if self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                logging.info("update infer cfg {} from {} to {}".format(key, self.cfg.infer_params[key], args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
            elif key in self.cfg.crop_params:
                if self.cfg.crop_params[key] != args_user[key]:
                    update_ret = True
                logging.info("update crop cfg {} from {} to {}".format(key, self.cfg.crop_params[key], args_user[key]))
                self.cfg.crop_params[key] = args_user[key]
            else:
                if key in self.cfg.infer_params and self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                logging.info("add {}:{} to infer cfg".format(key, args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
        return update_ret

    def clean_models(self, **kwargs):
        """
        clean model
        :param kwargs:
        :return:
        """
        for key in list(self.model_dict.keys()):
            del self.model_dict[key]
        self.model_dict = {}

    def init_models(self, **kwargs):
        grid_sample_plugin_path = self.cfg.get("grid_sample_plugin_path", None)
        logging.info("load Human Model >>>")
        self.is_animal = False
        for model_name in self.cfg.models:
            logging.info(f"loading model: {model_name}")
            logging.info(self.cfg.models[model_name])
            self.model_dict[model_name] = getattr(models, self.cfg.models[model_name]["name"])(
                **self.cfg.models[model_name], grid_sample_plugin_path=grid_sample_plugin_path)

    def init_vars(self, **kwargs):
        self.mask_crop = torch.from_numpy(cv2.imread(self.cfg.infer_params.mask_crop_path, cv2.IMREAD_COLOR)).to(self.device)
        self.frame_id = 0
        self.src_lmk_pre = None
        self.R_d_0 = None
        self.x_d_0_info = None
        self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
        self.exp_smooth = utils.OneEuroFilter(4, 0.3)

        self.source_path = None
        self.src_infos = []
        self.src_imgs = []
        self.is_source_video = False

    def calc_combined_eye_ratio(self, c_d_eyes_i, source_lmk):
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_d_eyes_i = np.array(c_d_eyes_i).reshape(1, 1)
        combined_eye_ratio_tensor = np.concatenate([c_s_eyes, c_d_eyes_i], axis=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, c_d_lip_i, source_lmk):
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_d_lip_i = np.array(c_d_lip_i).reshape(1, 1)  # 1x1
        combined_lip_ratio_tensor = np.concatenate([c_s_lip, c_d_lip_i], axis=1)  # 1x2
        return torch.from_numpy(combined_lip_ratio_tensor).to(self.device)

    def prepare_source(self, source_path, **kwargs):
        try:
            if utils.is_video(source_path):
                self.is_source_video = True
            else:
                self.is_source_video = False

            if self.is_source_video:
                src_imgs_bgr = []
                src_vcap = cv2.VideoCapture(source_path)
                while True:
                    ret, frame = src_vcap.read()
                    if not ret:
                        break
                    src_imgs_bgr.append(frame)
                src_vcap.release()
            else:
                img_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
                src_imgs_bgr = [img_bgr]

            self.src_imgs = []
            self.src_infos = []
            self.source_path = source_path

            for ii, img_bgr in tqdm(enumerate(src_imgs_bgr), total=len(src_imgs_bgr)):
                img_bgr = resize_to_limit(img_bgr, self.cfg.infer_params.source_max_dim,
                                          self.cfg.infer_params.source_division)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                src_faces = []
                if self.is_animal:
                    with torch.no_grad():
                        img_rgb_pil = Image.fromarray(img_rgb)
                        lmk = self.model_dict["xpose"].run(
                            img_rgb_pil,
                            'face',
                            'animal_face',
                            0,
                            0
                        )
                    if lmk is None:
                        continue
                    self.src_imgs.append(img_rgb)
                    src_faces.append(lmk)
                else:
                    src_faces = self.model_dict["face_analysis"].predict(img_bgr)
                    if len(src_faces) == 0:
                        logging.info("No face detected in the this image.")
                        continue
                    self.src_imgs.append(img_rgb)
                    # 如果是实时，只关注最大的那张脸
                    if kwargs.get("realtime", False):
                        src_faces = src_faces[:1]

                crop_infos = []
                for i in range(len(src_faces)):
                    # NOTE: temporarily only pick the first face, to support multiple face in the future
                    lmk = src_faces[i]
                    # crop the face
                    ret_dct = crop_image(
                        img_rgb,  # ndarray
                        lmk,  # 106x2 or Nx2
                        dsize=self.cfg.crop_params.src_dsize,
                        scale=self.cfg.crop_params.src_scale,
                        vx_ratio=self.cfg.crop_params.src_vx_ratio,
                        vy_ratio=self.cfg.crop_params.src_vy_ratio,
                    )
                    if self.is_animal:
                        ret_dct["lmk_crop"] = lmk
                    else:
                        lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                        ret_dct["lmk_crop"] = lmk
                        ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize

                    # update a 256x256 version for network input
                    ret_dct["img_crop_256x256"] = cv2.resize(
                        ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
                    )
                    crop_infos.append(ret_dct)

                src_infos = [[] for _ in range(len(crop_infos))]
                for i, crop_info in enumerate(crop_infos):
                    source_lmk = crop_info['lmk_crop']
                    img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']
                    pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(
                        img_crop_256x256)
                    x_s_info = {
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll,
                        "t": t,
                        "exp": exp,
                        "scale": scale,
                        "kp": kp
                    }
                    src_infos[i].append(copy.deepcopy(x_s_info))
                    x_c_s = kp
                    R_s = get_rotation_matrix(pitch, yaw, roll)
                    f_s = self.model_dict["app_feat_extractor"].predict(img_crop_256x256)
                    x_s = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
                    src_infos[i].extend([source_lmk.copy(), R_s.copy(), f_s.copy(), x_s.copy(), x_c_s.copy()])
                    if not self.is_animal:
                        flag_lip_zero = self.cfg.infer_params.flag_normalize_lip  # not overwrite
                        if flag_lip_zero:
                            # let lip-open scalar to be 0 at first
                            # 似乎要调参？
                            c_d_lip_before_animation = [0.05]
                            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                                c_d_lip_before_animation, source_lmk.copy())
                            if combined_lip_ratio_tensor_before_animation[0][
                                0] < self.cfg.infer_params.lip_normalize_threshold:
                                flag_lip_zero = False
                                src_infos[i].append(None)
                                src_infos[i].append(flag_lip_zero)
                            else:
                                lip_delta_before_animation = self.model_dict['stitching_lip_retarget'].predict(
                                    concat_feat(x_s, combined_lip_ratio_tensor_before_animation))
                                src_infos[i].append(lip_delta_before_animation.copy())
                                src_infos[i].append(flag_lip_zero)
                        else:
                            src_infos[i].append(None)
                            src_infos[i].append(flag_lip_zero)
                    else:
                        src_infos[i].append(None)
                        src_infos[i].append(False)

                    ######## prepare for pasteback ########
                    if self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop and self.cfg.infer_params.flag_stitching:
                        mask_ori_float = prepare_paste_back(self.mask_crop, crop_info['M_c2o'],
                                                            dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                        mask_ori_float = torch.from_numpy(mask_ori_float).to(self.device)
                        src_infos[i].append(mask_ori_float)
                    else:
                        src_infos[i].append(None)
                    M = torch.from_numpy(crop_info['M_c2o']).to(self.device)
                    src_infos[i].append(M)
                self.src_infos.append(src_infos[:])
            logging.info(f"finish process source:{source_path} >>>>>>>>")
            return len(self.src_infos) > 0
        except Exception as e:
            logging.exception(f"Error preparing source {source_path}: {e}")
            traceback.print_exc()
            return False

    def prepare_source_np(self, src_image_np):
        try:
            self.is_source_video = False
            self.src_imgs = []
            self.src_infos = []
            self.source_path = "numpy_source"

            img_bgr = src_image_np
            img_bgr = resize_to_limit(img_bgr, self.cfg.infer_params.source_max_dim,
                                      self.cfg.infer_params.source_division)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            src_faces = []
            if self.is_animal:
                 logging.warning("Animal face detection from NumPy array not implemented yet.")
                 return False
            else:
                src_faces = self.model_dict["face_analysis"].predict(img_bgr)
                if len(src_faces) == 0:
                    logging.info("No face detected in the source image.")
                    return False
                self.src_imgs.append(img_rgb)
                src_faces = src_faces[:1]

            crop_infos = []
            for i in range(len(src_faces)):
                lmk = src_faces[i]
                ret_dct = crop_image(
                    img_rgb, lmk,
                    dsize=self.cfg.crop_params.src_dsize, scale=self.cfg.crop_params.src_scale,
                    vx_ratio=self.cfg.crop_params.src_vx_ratio, vy_ratio=self.cfg.crop_params.src_vy_ratio,
                )
                if self.is_animal:
                    ret_dct["lmk_crop"] = lmk
                else:
                    lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                    ret_dct["lmk_crop"] = lmk
                    ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize

                ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
                crop_infos.append(ret_dct)

            if not crop_infos:
                 return False

            src_info_face = []
            crop_info = crop_infos[0]
            source_lmk = crop_info['lmk_crop']
            img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']
            pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(img_crop_256x256)
            x_s_info = {"pitch": pitch, "yaw": yaw, "roll": roll, "t": t, "exp": exp, "scale": scale, "kp": kp}

            src_info_face.append(copy.deepcopy(x_s_info))
            x_c_s = kp
            R_s = get_rotation_matrix(pitch, yaw, roll)
            f_s = self.model_dict["app_feat_extractor"].predict(img_crop_256x256)
            x_s = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
            src_info_face.extend([source_lmk.copy(), R_s.copy(), f_s.copy(), x_s.copy(), x_c_s.copy()])

            lip_delta_before_animation = None
            flag_lip_zero = False
            if not self.is_animal:
                flag_lip_zero = self.cfg.infer_params.flag_normalize_lip
                if flag_lip_zero:
                    c_d_lip_before_animation = [0.05]
                    combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk.copy())
                    if combined_lip_ratio_tensor_before_animation[0][0] < self.cfg.infer_params.lip_normalize_threshold:
                        flag_lip_zero = False
                    else:
                        lip_delta_before_animation = self.model_dict['stitching_lip_retarget'].predict(concat_feat(x_s, combined_lip_ratio_tensor_before_animation))

            src_info_face.append(lip_delta_before_animation.copy() if lip_delta_before_animation is not None else None)
            src_info_face.append(flag_lip_zero)

            ######## prepare for pasteback ########
            mask_ori_float = None
            M = None
            if self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop and self.cfg.infer_params.flag_stitching:
                 mask_ori_float = prepare_paste_back(self.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                 mask_ori_float = torch.from_numpy(mask_ori_float).to(self.device)
                 M = torch.from_numpy(crop_info['M_c2o']).to(self.device)

            src_info_face.append(mask_ori_float)
            src_info_face.append(M)

            self.src_infos.append([src_info_face])

            return True
        except Exception as e:
            logging.error(f"Error processing source numpy array: {e}")
            traceback.print_exc()
            return False


    def prepare_source_tensor(self, src_image_tensor):
        try:
            self.is_source_video = False
            self.src_imgs = []
            self.src_infos = []
            self.source_path = "tensor_source"

            # print(type(src_image_tensor))
            # src_image_tensor_temp = src_image_tensor * 255.0
            # src_image_tensor_temp = src_image_tensor_temp.to(torch.uint8)
            # Convert to RGB if needed and keep a copy for face detection
            # with prof("CPU TO GPU MOVEMENT"):
            #     if src_image_tensor.shape[0] == 3:
            #         src_image_bgr = src_image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
            #         src_image_bgr = src_image_bgr.astype(np.uint8)
            #         src_image_bgr = cv2.cvtColor(src_image_bgr, cv2.COLOR_RGB2BGR)
            #     else:
            #         src_image_bgr = src_image_tensor.cpu().numpy() * 255.0
            #         src_image_bgr = src_image_bgr.astype(np.uint8)

            # print(src_image_bgr)
            src_image_bgr = src_image_tensor
            # print(type(src_image_bgr))
            # src_image_bgr = cv2.cvtColor(src_image_bgr, cv2.COLOR_RGB2BGR) # Removed: Input is likely already BGR

            logging.info("Getting face landmarks for source image...")
            # Get face landmarks using BGR image
            with prof("face_analysis.predict"):
                src_faces = self.model_dict["face_analysis"].predict(src_image_bgr)
            if len(src_faces) == 0:
                logging.error("No face detected in the source image.")
                return False

            with prof("landmark.predict"):
            # Process first face only
                lmk = src_faces[0]
                logging.info("Getting landmark predictions...")
                lmk = self.model_dict["landmark"].predict(src_image_bgr, lmk)
            
            with prof("post landmark.predict"):
                # Convert to tensor and move to device as float32
                lmk = torch.from_numpy(lmk).float().to(self.device)
                
                # Prepare source info
                src_info_face = []
                
                # Get motion info using RGB image
                logging.info("Getting motion info...")
                src_image_rgb = cv2.cvtColor(src_image_bgr, cv2.COLOR_BGR2RGB)
                # Resize to 256x256 for motion extractor
                src_image_rgb_256 = cv2.resize(src_image_rgb, (256, 256))
            
            with prof("motion_extractor.predict"):
                pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(src_image_rgb_256)
            
            with prof("post motion_extractor.predict"):
                x_s_info = {"pitch": pitch, "yaw": yaw, "roll": roll, "t": t, "exp": exp, "scale": scale, "kp": kp}
                
                src_info_face.append(copy.deepcopy(x_s_info))
                x_c_s = torch.from_numpy(kp).float().to(self.device)
                R_s = torch.from_numpy(get_rotation_matrix(pitch, yaw, roll)).float().to(self.device)
                
                logging.info("Getting appearance features...")
                f_s = torch.from_numpy(self.model_dict["app_feat_extractor"].predict(src_image_rgb_256)).float().to(self.device)
                x_s = torch.from_numpy(transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)).float().to(self.device)
                
                src_info_face.extend([lmk, R_s, f_s, x_s, x_c_s])
                
                # Add None for lip delta and flag
                src_info_face.append(None)
                src_info_face.append(False)
                
                # Add None for mask and M since we're not using pasteback
                src_info_face.append(None)
                src_info_face.append(None)
                
                self.src_infos.append([src_info_face])
                self.src_imgs.append(src_image_tensor)
                
                logging.info("Source preparation completed successfully")
                return True
        except Exception as e:
            logging.error(f"Error processing source tensor: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def extract_driving_info_tensor(self, dri_image_tensor):
        try:
            # Ensure input is 512x512 and float32
            # if dri_image_tensor.shape[1] != 512 or dri_image_tensor.shape[2] != 512:
            #     dri_image_tensor = F.interpolate(dri_image_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
            # dri_image_tensor = dri_image_tensor.float()

            # Convert to RGB if needed and keep a copy for face detection
            # if dri_image_tensor.shape[0] == 3:
            #     dri_image_bgr = dri_image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
            #     dri_image_bgr = dri_image_bgr.astype(np.uint8)
            #     dri_image_bgr = cv2.cvtColor(dri_image_bgr, cv2.COLOR_RGB2BGR)
            # else:
            #     dri_image_bgr = dri_image_tensor.cpu().numpy() * 255.0
            #     dri_image_bgr = dri_image_bgr.astype(np.uint8)

            dri_image_bgr = dri_image_tensor

            logging.info("Getting face landmarks for driving image...")
            # Get face landmarks using BGR image
            src_face = self.model_dict["face_analysis"].predict(dri_image_bgr)
            if len(src_face) == 0:
                logging.error("No face detected in the driving image.")
                return None

            lmk = src_face[0]
            logging.info("Getting landmark predictions...")
            lmk = self.model_dict["landmark"].predict(dri_image_bgr, lmk)
            lmk = torch.from_numpy(lmk).float().to(self.device)

            # Get motion info using RGB image
            logging.info("Getting motion info...")
            dri_image_rgb = cv2.cvtColor(dri_image_bgr, cv2.COLOR_BGR2RGB)
            # Resize to 256x256 for motion extractor
            dri_image_rgb_256 = cv2.resize(dri_image_rgb, (256, 256))
            pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(dri_image_rgb_256)
            x_d_i_info = {
                "pitch": pitch, "yaw": yaw, "roll": roll, "t": t, "exp": exp, "scale": scale, "kp": kp
            }
            R_d_i = torch.from_numpy(get_rotation_matrix(pitch, yaw, roll)).float().to(self.device)

            # Calculate lip ratio
            logging.info("Calculating lip ratio...")
            input_lip_ratio = torch.from_numpy(calc_lip_close_ratio(lmk.cpu().numpy()[None])).float().to(self.device)

            logging.info("Driving info extraction completed successfully")
            return x_d_i_info, R_d_i, input_lip_ratio

        except Exception as e:
            logging.error(f"Error extracting driving info from tensor: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def animate_image(self, src_image_tensor, dri_image_tensor):
        try:
            logging.info("Starting source preparation...")
            if not self.prepare_source_tensor(src_image_tensor):
                logging.error("Source preparation failed.")
                return None

            logging.info("Starting driving info extraction...")
            driving_info = self.extract_driving_info_tensor(dri_image_tensor)
            if driving_info is None:
                logging.error("Driving info extraction failed.")
                return None
            x_d_i_info, R_d_i, input_lip_ratio = driving_info

            if not self.src_infos or not self.src_infos[0]:
                logging.error("Source info not found after preparation.")
                return None
            src_info_list = self.src_infos[0]
            img_src_rgb = self.src_imgs[0]

            R_d_0 = R_d_i.clone()
            x_d_0_info = copy.deepcopy(x_d_i_info)

            self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
            self.exp_smooth = utils.OneEuroFilter(4, 0.3)

            logging.info("Starting animation process...")
            try:
                out_crop_rgb_np, out_org_rgb_np = self._run(
                    src_info_list, x_d_i_info, x_d_0_info, R_d_i, R_d_0,
                    realtime=False,
                    input_eye_ratio=None,
                    input_lip_ratio=input_lip_ratio,
                )
            except Exception as e:
                logging.error(f"Error during _run: {str(e)}")
                logging.error(traceback.format_exc())
                return None

            output_rgb_np = None
            if out_crop_rgb_np is not None:
                output_rgb_np = out_crop_rgb_np
            else:
                logging.error("Animation failed to produce an output.")
                return None

            logging.info("Animation completed successfully")
            return output_rgb_np

        except Exception as e:
            logging.error(f"Error in animate_image: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def retarget_eye(self, kp_source, eye_close_ratio):
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp+2)
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)
        delta = self.model_dict['stitching_eye_retarget'].predict(feat_eye)
        return delta

    def retarget_lip(self, kp_source, lip_close_ratio):
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)
        delta = self.model_dict['stitching_lip_retarget'].predict(feat_lip)
        return delta

    def stitching(self, kp_source, kp_driving):
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        bs, num_kp = kp_source.shape[:2]

        kp_driving_new = kp_driving.copy()

        delta = self.model_dict['stitching'].predict(concat_feat(kp_source, kp_driving_new))

        delta_exp = delta[..., :3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
        delta_tx_ty = delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty

        return kp_driving_new

    def _run(self, src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio, input_lip_ratio, **kwargs):
        out_crop, out_org = None, None
        for j in range(len(src_info)):
            x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M = \
                src_info[j]

            # Convert numpy arrays to tensors and move to device
            delta_new = torch.from_numpy(x_s_info['exp']).float().to(self.device)
            scale_new = torch.from_numpy(x_s_info['scale']).float().to(self.device)
            t_new = torch.from_numpy(x_s_info['t']).float().to(self.device)

            # Only handle lip animation
            if self.cfg.infer_params.animation_region in ["lip"]:
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = torch.from_numpy(x_d_i_info['exp'][:, lip_idx, :]).float().to(self.device)

            t_new[..., 2] = 0  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_s + delta_new) + t_new

            # Apply lip retargeting
            if self.cfg.infer_params.flag_lip_retargeting:
                c_d_lip_i = input_lip_ratio
                combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                lip_delta = self.retarget_lip(x_s, combined_lip_ratio_tensor)
                x_d_i_new = x_d_i_new + lip_delta.reshape(-1, x_s.shape[1], 3)

            x_d_i_new = x_s + (x_d_i_new - x_s) * self.cfg.infer_params.driving_multiplier
            out_crop = self.model_dict["warping_spade"].predict(f_s, x_s, x_d_i_new)

        return out_crop.to(dtype=torch.uint8).cpu().numpy(), None

    def run(self, image, img_src, src_info, **kwargs):
        img_bgr = image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        I_p_pstbk = torch.from_numpy(img_src).to(self.device).float()
        realtime = kwargs.get("realtime", False)
        if self.cfg.infer_params.flag_crop_driving_video:
            if self.src_lmk_pre is None:
                src_face = self.model_dict["face_analysis"].predict(img_bgr)
                if len(src_face) == 0:
                    return None, None, None, None
                lmk = src_face[0]
                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                self.src_lmk_pre = lmk.copy()
            else:
                lmk = self.model_dict["landmark"].predict(img_rgb, self.src_lmk_pre)
                self.src_lmk_pre = lmk.copy()

            ret_bbox = parse_bbox_from_landmark(
                lmk,
                scale=self.cfg.crop_params.dri_scale,
                vx_ratio_crop_video=self.cfg.crop_params.dri_vx_ratio,
                vy_ratio=self.cfg.crop_params.dri_vy_ratio,
            )["bbox"]
            global_bbox = [
                ret_bbox[0, 0],
                ret_bbox[0, 1],
                ret_bbox[2, 0],
                ret_bbox[2, 1],
            ]
            ret_dct = crop_image_by_bbox(
                img_rgb,
                global_bbox,
                lmk=lmk,
                dsize=kwargs.get("dsize", 512),
                flag_rot=False,
                borderValue=(0, 0, 0),
            )
            lmk_crop = ret_dct["lmk_crop"]
            img_crop = ret_dct["img_crop"]
            img_crop = cv2.resize(img_crop, (256, 256))
        else:
            if self.src_lmk_pre is None:
                src_face = self.model_dict["face_analysis"].predict(img_bgr)
                if len(src_face) == 0:
                    return None, None, None, None
                lmk = src_face[0]
                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                self.src_lmk_pre = lmk.copy()
            else:
                lmk = self.model_dict["landmark"].predict(img_rgb, self.src_lmk_pre)
                self.src_lmk_pre = lmk.copy()
            lmk_crop = lmk.copy()
            img_crop = cv2.resize(img_rgb, (256, 256))

        input_eye_ratio = calc_eye_close_ratio(lmk_crop[None])
        input_lip_ratio = calc_lip_close_ratio(lmk_crop[None])
        pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(img_crop)
        x_d_i_info = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "t": t,
            "exp": exp,
            "scale": scale,
            "kp": kp
        }
        R_d_i = get_rotation_matrix(pitch, yaw, roll)
        x_d_i_info["R"] = R_d_i
        x_d_i_info_copy = copy.deepcopy(x_d_i_info)
        for key in x_d_i_info_copy:
            x_d_i_info_copy[key] = x_d_i_info_copy[key].astype(np.float32)
        dri_motion_info = [x_d_i_info_copy, copy.deepcopy(input_eye_ratio.astype(np.float32)),
                           copy.deepcopy(input_lip_ratio.astype(np.float32))]
        if kwargs.get("first_frame", False) or self.R_d_0 is None:
            self.frame_id = 0
            self.R_d_0 = R_d_i.copy()
            self.x_d_0_info = copy.deepcopy(x_d_i_info)
            # realtime smooth
            self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
            self.exp_smooth = utils.OneEuroFilter(4, 0.3)
        R_d_0 = self.R_d_0.copy()
        x_d_0_info = copy.deepcopy(self.x_d_0_info)
        out_crop, I_p_pstbk = self._run(src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio,
                                        input_lip_ratio, I_p_pstbk, **kwargs)
        return img_crop, out_crop, I_p_pstbk, dri_motion_info

    def run_with_pkl(self, dri_motion_info, img_src, src_info, **kwargs):
        I_p_pstbk = torch.from_numpy(img_src).to(self.device).float()
        realtime = kwargs.get("realtime", False)

        input_eye_ratio = dri_motion_info[1]
        input_lip_ratio = dri_motion_info[2]
        x_d_i_info = dri_motion_info[0]
        R_d_i = x_d_i_info["R"] if "R" in x_d_i_info else x_d_i_info["R_d"]

        if kwargs.get("first_frame", False) or self.R_d_0 is None:
            self.frame_id = 0
            self.R_d_0 = R_d_i.copy()
            self.x_d_0_info = copy.deepcopy(x_d_i_info)
            # realtime smooth
            self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
            self.exp_smooth = utils.OneEuroFilter(4, 0.3)
        R_d_0 = self.R_d_0.copy()
        x_d_0_info = copy.deepcopy(self.x_d_0_info)
        out_crop, I_p_pstbk = self._run(src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio,
                                        input_lip_ratio, I_p_pstbk, **kwargs)
        return out_crop, I_p_pstbk

    def __del__(self):
        self.clean_models()
