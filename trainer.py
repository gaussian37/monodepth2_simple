# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # 입력 영상의 height, width의 크기가 32의 배수가 될 수 있도록 설정함        
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # [0, 1, 2, 3]
        # scale을 정의하는 값으로 1/(2**0), 1/(2**1), 1/(2**2) 1/(2**3) 까지의 scale을 가지도록 정의합니다.
        self.num_scales = len(self.opt.scales) 
        
        # [0, -1, 1]
        # 연속된 3개의 Frame을 처리하기 위한 Frame의 상대적인 차이를 정의함
        # t번째 Frame은 0, t-1번째 Frame은 -1, t+1번째 Frame은 1로 정의하였음
        self.num_input_frames = len(self.opt.frame_ids) 
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        
        # pose network가 사용할 입력의 갯수로 (t-1, t), (t, t+1) 과 같이 2개의 Frame으로 이루어진 쌍을
        # 사용할 예정이므로 num_pose_frames = 2로 정의하였음
        self.num_pose_frames = 2 # if self.opt.pose_model_input == "pairs" else self.num_input_frames

        # 스테레오 카메라가 아닌 경우 pose network를 사용해야 함으로 아래 조건을 True 로 지정함
        self.use_pose_net = True

        # Depth Estimation 모델의 Encoder를 선언함
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Depth Estimation 모델의 Decoder를 선언함
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # 단안 카메라를 사용하였으므로 pose network를 사용하고
        # Depth Estimation Network 와 별개의 Encoder를 사용하여 pose network를 구성하였음
        # num_pose_frames = 2
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        # Pose Network의 Decoder를 선언함
        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())
        
        # optimizer 및 scheduler
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # 기존 weight 사용
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # dataset, dataloader 관련
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # SSIM Loss 적용
        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        # scales : [0, 1, 2, 3]
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            # BackprojectDepth : depth image -> point cloud로 변환하기 위함 (layers.py)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            
            # BackprojectDepth : depth image -> point cloud로 변환하기 위함 (layers.py)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # process_batch를 통하여 models와 inputs를 이용하여 output과 loss를 계산함
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # inputs 중에서 frame_id=0, scale=0인 이미지만 Depth Encoder에 입력으로 넣는다.
        # ("color_aug", <frame_id>, <scale>)        
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        
        # outputs[('disp', scales)].shape : (B, C=1, H//(2**scales), W//(2**scales))
        # outputs[('disp'), 0].shape : (B, C=1, H, W)
        # outputs[('disp'), 1].shape : (B, C=1, H/2, W/2)
        # outputs[('disp'), 2].shape : (B, C=1, H/4, W/4)
        # outputs[('disp'), 3].shape : (B, C=1, H/8, W/8)
        outputs = self.models["depth"](features)
        
        # pose 예측 결과를 outputs (dict)에 추가함
        # outputs += ("axisangle", 0, f_i), ("translation", 0, f_i), ("cam_T_cam", 0, f_i)
        outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                # t-1, t Frame 순서로 정렬
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                # t, t+1 Frame 순서로 정렬
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            # Pose Encdoer는 연속된 두 Frame을 입력으로 받아서 feature를 출력함
            pose_inputs = self.models["pose_encoder"](torch.cat(pose_inputs, 1))

            # Pose Encdoer의 출력 feature를 Pose Decoder에 입력하여 
            # 최종적으로 카메라의 axisangle, translation을 예측함
            # axisangle : (B, 2, 1, 3)
            # translation : (B, 2, 1, 3)
            axisangle, translation = self.models["pose"](pose_inputs)
            
            # 0 -> f_i(-1 또는 +1)로의 변화된 카메라 pose의 axisangle, translation을 예측함
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # cam to cam R|t 행렬을 예측함
            # 출력은 (4 x 4) 크기의 행렬을 가지며 [R|t; 0, 0, 0, 1] 형태를 가짐
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            # 입력 해상도 (scale = 0)로 변경
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                # I_{t}의 깊이 정보인 D_t (depth)를 3D 좌표로 backprojection함
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                # 3D 좌표 (t) -> 3D 좌표 (t') -> uv 좌표 (t')
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                # inputs[("color", frame_id, source_scale)] : I_{t'}의 원본 해상도 이미지
                # outputs[("sample", frame_id, scale)] : uv 좌표 (t')
                
                # outputs[("color", frame_id, scale)] : I_{t' -> t}
                # => sample 좌표 (uv 좌표)를 이용하여 I_{t'}를 I_{t}에 맞게 sampling 하여 생성 (I_{t'->t})하고
                #    최종적으로 I_{t'->t}와 I_{t}의 reprojection loss가 낮아지도록 학습하므로 학습 완료 시 I_{t' -> t}와 I_{t}는 유사해짐
                #    F.grid_sample 연산에 따라 기존 이미지 크기를 벗어난 sample 좌표들은 가장 외곽의 값으로 대체됨
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        # scale 별 Loss를 구합니다. [0, 1, 2, 3]
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            # outputs[('disp', scales)].shape : (B, C=1, H//(2**scales), W//(2**scales))
            disp = outputs[("disp", scale)]
            # (2**scale) 만큼 다운사이즈된 I_{t}
            color = inputs[("color", 0, scale)]
            # scale 변화가 없는 I_{t}
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                # outputs[("color", frame_id, scale)]는 source_scale의 크기를 가짐
                # outputs[("color", frame_id, scale)]에서 frame_id는 -1 또는 1을 가지며 scale은 0, 1, 2, 3을 가짐
                # outputs[("color", frame_id, scale)]의 크기는 source_scale을 가지나 scale 만큼 downsampling된 이미지로 부터 생성된 것을 의미함
                pred = outputs[("color", frame_id, scale)]
                
                # pred.shape : (B, 1, source_height, source_width)
                # target.shape : (B, 1, source_height, source_width)
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            # I_{t}와 I_{t-1}, I_{t+1} 과 구한 reprojection loss
            reprojection_loss = torch.cat(reprojection_losses, 1)

            ############# auto-masking을 위한 identity_reprojection_loss를 구함 #########
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                # pred는 scale 변화가 없는 I_{t'}
                pred = inputs[("color", frame_id, source_scale)]
                # target은 scale 변화가 없는 I_{t}
                
                # I_{t'} (pred) 와 I_{t} (target) 간의 reprojection loss를 구하며 이를 identity_reprojection_loss 라고 함
                # auto masking을 적용하기 위하여 사용함
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            # I_{t}와 I_{t-1}, I_{t+1} 과 구한 identity reprojection loss
            identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001
            ###########################################################################

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            
            # identity_reprojection_loss와 reprojection_loss 중 가장 작은 값을 선택합니다.
            # 만약 identity_reprojection_loss 중에서 가장 작은 부분이 선택된다면 두 Frame 간 static 한 픽셀에 의해 loss가 가장 작아서 선택 된 것으로 가정하며
            # static한 픽셀은 차이가 거의 없어 Loss가 0에 가까워지므로 auto-masking이 됩니다.
            # reprojection_loss 중에서 가장 작은 값이 선택된다면 두 Frame 간 차이가 있는 것으로 가정하며, occluded pixel 문제 또한 처리된 것으로 판단합니다.
            to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            # norm_disp : d^{*}_t 에 해당하며 norm_disp와 color 이미지를 통하여 smoothness loss를 구합니다. 
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
