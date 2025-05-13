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
        for key, user_value in args_user.items():
            if key in self.cfg.infer_params:
                if self.cfg.infer_params[key] != user_value:
                    update_ret = True
                    logging.info(f"update infer_params.{key} from {self.cfg.infer_params[key]} to {user_value}")
                    self.cfg.infer_params[key] = user_value
            # TODO: add crop_params update, Currently only updating infer_params
        return update_ret
    
    def set_cfg_param(self, key, value):
        container = None
        if key in self.cfg.infer_params:
            container = self.cfg.infer_params
        elif key in self.cfg.crop_params:
            container = self.cfg.crop_params
        else:
            container = self.cfg.infer_params  # fallback

        if container.get(key) != value:
            print(f"update cfg {key} from {container.get(key)} to {value}")
            container[key] = value
            return True
        return False

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
        self.mask_crop = cv2.imread(self.cfg.infer_params.mask_crop_path, cv2.IMREAD_COLOR)
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
        # source_lmk might be numpy array or tensor
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        
        # Ensure c_d_lip_i is compatible for concatenation
        if isinstance(c_s_lip, torch.Tensor):
            # Convert c_d_lip_i to tensor on the same device as c_s_lip
            c_d_lip_i_tensor = torch.tensor(c_d_lip_i, device=c_s_lip.device, dtype=c_s_lip.dtype).reshape(1, 1)
            # Use torch.cat for tensors
            combined_lip_ratio_tensor = torch.cat([c_s_lip, c_d_lip_i_tensor], dim=1)  # 1x2
        else:
            # Keep original numpy operations
            c_d_lip_i_np = np.array(c_d_lip_i).reshape(1, 1)  # 1x1
        # [c_s,lip, c_d,lip,i]
            combined_lip_ratio_tensor = np.concatenate([c_s_lip, c_d_lip_i_np], axis=1)  # 1x2
            
        return combined_lip_ratio_tensor

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
            self.source_path = "tensor_source"
            assert src_image_tensor.shape[2] == 3 and src_image_tensor.shape[0] == 512 and src_image_tensor.shape[1] == 512, "src_image_tensor must be a 3x512x512 tensor"
            src_image_bgr = src_image_tensor
            src_image_rgb = cv2.cvtColor(src_image_bgr, cv2.COLOR_BGR2RGB)

            logging.info("Getting face landmarks for source image...")
            src_faces = self.model_dict["face_analysis"].predict(src_image_bgr)
            if len(src_faces) == 0:
                logging.error("No face detected in the source image.")
                return None

            lmk = src_faces[0]
            logging.debug("Getting landmark predictions...")
            ret_dct = crop_image(
                    src_image_rgb, lmk,
                    dsize=self.cfg.crop_params.src_dsize, scale=self.cfg.crop_params.src_scale,
                    vx_ratio=self.cfg.crop_params.src_vx_ratio, vy_ratio=self.cfg.crop_params.src_vy_ratio,
                )
            lmk = self.model_dict["landmark"].predict(src_image_rgb, lmk)

            lmk = torch.from_numpy(lmk).float().to(self.device)
            
            src_info_face = []            
            logging.debug("Getting motion info...")
            src_image_rgb_256 = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            
            pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(src_image_rgb_256)
            
            x_s_info = {"pitch": pitch, "yaw": yaw, "roll": roll, "t": t, "exp": exp, "scale": scale, "kp": kp}
            src_info_face.append(x_s_info)
            
            x_c_s = torch.from_numpy(kp).float().to(self.device)
            
            R_s_np = get_rotation_matrix(pitch, yaw, roll)
            R_s = torch.from_numpy(R_s_np).float().to(self.device)
            
            logging.info("Getting appearance features...")
            f_s = self.model_dict["app_feat_extractor"].predict(src_image_rgb_256)
            
            if not isinstance(f_s, torch.Tensor):
                f_s = torch.from_numpy(f_s)
            f_s = f_s.float().to(self.device)
            
            x_s_np = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
            x_s = torch.from_numpy(x_s_np).float().to(self.device)

            mask_ori_float = None
            M = None
            if self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop and self.cfg.infer_params.flag_stitching:
                 mask_ori_float = prepare_paste_back(self.mask_crop, ret_dct['M_c2o'], dsize=(src_image_bgr.shape[1], src_image_bgr.shape[0]))
                 mask_ori_float = torch.from_numpy(mask_ori_float).to(self.device)
                 M = torch.from_numpy(ret_dct['M_c2o']).to(self.device)
            
            src_info_face.extend([lmk, R_s, f_s, x_s, x_c_s])
            src_info_face.append(None)
            src_info_face.append(False)
            src_info_face.append(mask_ori_float)
            src_info_face.append(M)
                
            logging.info("Source preparation completed successfully")
            return [src_info_face], src_image_rgb
        except Exception as e:
            logging.error(f"Error processing source tensor: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def extract_driving_info_tensor(self, dri_image_tensor):
        try:
            assert dri_image_tensor.shape[2] == 3 and dri_image_tensor.shape[0] == 512 and dri_image_tensor.shape[1] == 512, "dri_image_tensor must be a 3x512x512 tensor"
            dri_image_bgr = dri_image_tensor
            dri_image_rgb = cv2.cvtColor(dri_image_bgr, cv2.COLOR_BGR2RGB)

            logging.info("Getting face landmarks for driving image...")
            # Get face landmarks using BGR image
            src_face = self.model_dict["face_analysis"].predict(dri_image_bgr)
            if len(src_face) == 0:
                logging.error("No face detected in the driving image.")
                return None

            lmk = src_face[0]
            logging.info("Getting landmark predictions...")
            lmk = self.model_dict["landmark"].predict(dri_image_rgb, lmk)
            # lmk = torch.from_numpy(lmk).float().to(self.device)

            # Get motion info using RGB image
            logging.info("Getting motion info...")
            # Resize to 256x256 for motion extractor
            dri_image_rgb_256 = cv2.resize(dri_image_rgb, (256, 256))
            pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(dri_image_rgb_256)
            x_d_i_info = {
                "pitch": pitch, "yaw": yaw, "roll": roll, "t": t, "exp": exp, "scale": scale, "kp": kp
            }
            # R_d_i = torch.from_numpy(get_rotation_matrix(pitch, yaw, roll)).float().to(self.device)
            R_d_i = None

            # Calculate lip ratio
            logging.info("Calculating lip ratio...")
            input_lip_ratio = calc_lip_close_ratio(lmk[None])

            logging.info("Driving info extraction completed successfully")
            return x_d_i_info, R_d_i, input_lip_ratio

        except Exception as e:
            logging.error(f"Error extracting driving info from tensor: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def animate_image(self, src_image_tensor, dri_image_tensor):
        try:
            logging.debug("Starting source preparation...")
            src_info_list, I_p_pstbk = self.prepare_source_tensor(src_image_tensor)

            if not src_info_list:
                logging.error("Source preparation failed.")
                return None

            logging.debug("Starting driving info extraction...")
            driving_info = self.extract_driving_info_tensor(dri_image_tensor)
            if driving_info is None:
                logging.error("Driving info extraction failed.")
                return None
            x_d_i_info, R_d_i, input_lip_ratio = driving_info

            logging.debug("Starting animation process...")
            try:
                I_p_pstbk = torch.from_numpy(I_p_pstbk).to(self.device).float()
                out_crop_rgb_np, out_org_rgb_np = self._run(
                    src_info_list, x_d_i_info, None, R_d_i, None,
                    realtime=False,
                    input_eye_ratio=None,
                    input_lip_ratio=input_lip_ratio,
                    I_p_pstbk=I_p_pstbk
                )
            except Exception as e:
                logging.error(f"Error during _run: {str(e)}")
                logging.error(traceback.format_exc())
                return None

            output_rgb_np = None
            if self.cfg.infer_params.flag_pasteback and out_org_rgb_np is not None:
                output_rgb_np = out_org_rgb_np
            elif out_crop_rgb_np is not None:
                output_rgb_np = out_crop_rgb_np
            else:
                logging.info("Animation failed to produce an output.")
                return None

            logging.debug("Animation completed successfully")
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
        
        # Ensure delta is a tensor on the correct device
        if not isinstance(delta, torch.Tensor):
            delta = torch.from_numpy(delta)
        delta = delta.to(device=kp_source.device, dtype=kp_source.dtype) # Match kp_source device and dtype
        
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

    def _run(self, src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio, input_lip_ratio, I_p_pstbk, **kwargs):
        out_crop, out_org = None, None
        for j in range(len(src_info)):
            x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M = \
                src_info[j]

            delta_new_np = x_s_info['exp']
            scale_new_np = x_s_info['scale']
            t_new_np = x_s_info['t']

            if self.cfg.infer_params.animation_region in ["lip"]:
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new_np[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]

            t_new_np[..., 2] = 0 

            delta_new = torch.from_numpy(delta_new_np).float().to(self.device)
            scale_new = torch.from_numpy(scale_new_np).float().to(self.device)
            t_new = torch.from_numpy(t_new_np).float().to(self.device)

            x_d_i_new = scale_new * (x_c_s @ R_s + delta_new) + t_new

            if self.cfg.infer_params.flag_lip_retargeting:
                c_d_lip_i = input_lip_ratio
                combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                lip_delta = self.retarget_lip(x_s, combined_lip_ratio_tensor)
                x_d_i_new = x_d_i_new + lip_delta.reshape(-1, x_s.shape[1], 3)

            x_d_i_new = x_s + (x_d_i_new - x_s) * self.cfg.infer_params.driving_multiplier
            out_crop = self.model_dict["warping_spade"].predict(f_s, x_s, x_d_i_new)

            if self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop and self.cfg.infer_params.flag_stitching:
                # TODO: pasteback is slow, considering optimize it using multi-threading or GPU
                # I_p_pstbk = paste_back(out_crop, crop_info['M_c2o'], I_p_pstbk, mask_ori_float)
                I_p_pstbk = paste_back_pytorch(out_crop, M, I_p_pstbk, mask_ori_float)

        return out_crop.to(dtype=torch.uint8).cpu().numpy(), I_p_pstbk.to(dtype=torch.uint8).cpu().numpy()

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
