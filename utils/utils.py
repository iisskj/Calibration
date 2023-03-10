import os
import sys
import cv2
import numpy as np
import json

import torch

import kornia
"""
NOTE:
left-right warp
depth-extrincs warp
flow warp
"""


def downsamp_K(cam_in, scale):  #cam_in:b x 3 x 3
  cam_out = cam_in.clone()
  cam_out[:, :2, :] = cam_out[:, :2, :] * scale
  return cam_out


def carla_depth_decode(ori):
  # normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
  # in_meters = 1000 * normalized
  ori = ori.astype(np.float32)
  h, w = ori.shape[:2]
  depth = np.ones([h, w, 1])
  # opencv -> H x W x 3 -> BGR
  depth = (ori[:, :, 2] * 256.0 + ori[:, :, 1]) * 256.0 + ori[:, :, 0]
  depth = depth / (256 * 256 * 256 - 1) * 1000  # to meters
  return depth.squeeze()  # [h,w,1]


# NOTE:
# refer to deep-video-mvs
def warp_frame_depth(img_src: torch.Tensor, depth_dst: torch.Tensor, trans_src2dst: torch.Tensor, camera_matrix_dst: torch.Tensor,camera_matrix_src: torch.Tensor, normalize_points: bool = False, sampling_mode: str = 'bilinear'):
  # unproject source points to camera frame
  points_3d_dst: torch.Tensor = kornia.geometry.depth.depth_to_3d(depth_dst, camera_matrix_dst, normalize_points)  # Bx3xHxW

  # transform points from source to destination
  points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

  # apply transformation to the 3d points
  points_3d_src = kornia.geometry.linalg.transform_points(trans_src2dst[:, None], points_3d_dst)  # BxHxWx3
  points_3d_src[:, :, :, 2] = torch.relu(points_3d_src[:, :, :, 2])

  # project back to pixels
  camera_matrix_tmp: torch.Tensor = camera_matrix_src[:, None, None]  # Bx1x1xHxW
  points_2d_src: torch.Tensor = kornia.geometry.camera.perspective.project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

  # normalize points between [-1 / 1]
  height, width = depth_dst.shape[-2:]
  points_2d_src_norm: torch.Tensor = kornia.geometry.conversions.normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

  output = torch.nn.functional.grid_sample(img_src, points_2d_src_norm, align_corners=True, mode=sampling_mode)
  return output


def warp_frame_depth_correct(img_src: torch.Tensor,
                             depth_dst: torch.Tensor,
                             trans_src2dst: torch.Tensor,
                             camera_matrix: torch.Tensor,
                             correct_vector: torch.Tensor,
                             normalize_points: bool = False,
                             sampling_mode: str = 'bilinear'):
  # unproject source points to camera frame
  points_3d_dst: torch.Tensor = kornia.geometry.depth.depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

  #correct = torch.from_numpy(np.array([1, -1, -1]).astype(np.float32)).cuda()

  # transform points from source to destination
  points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3
  points_3d_dst = points_3d_dst * correct_vector

  # apply transformation to the 3d points
  points_3d_src = kornia.geometry.linalg.transform_points(trans_src2dst[:, None], points_3d_dst)  # BxHxWx3
  points_3d_src = points_3d_src * correct_vector

  points_3d_src[:, :, :, 2] = torch.relu(points_3d_src[:, :, :, 2])

  # project back to pixels
  camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
  points_2d_src: torch.Tensor = kornia.geometry.camera.perspective.project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

  # normalize points between [-1 / 1]
  height, width = depth_dst.shape[-2:]
  points_2d_src_norm: torch.Tensor = kornia.geometry.conversions.normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2
  points_2d_src_norm[points_2d_src_norm > 1.0] = 1.0
  points_2d_src_norm[points_2d_src_norm < -1.0] = -1.0
  output = torch.nn.functional.grid_sample(img_src, points_2d_src_norm, align_corners=True, mode=sampling_mode)
  return output


def get_non_differentiable_rectangle_depth_estimation(reference_pose_torch, measurement_pose_torch, previous_depth_torch, full_K_torch, half_K_torch, original_width, original_height, correct_vector):
  # reference_pose_torch = cur pose
  # measurement_pose_torch = previous_pose
  #
  batch_size, _, _ = reference_pose_torch.shape
  half_width = int(original_width / 2)
  half_height = int(original_height / 2)

  trans = torch.bmm(torch.inverse(reference_pose_torch), measurement_pose_torch).cuda()
  # trans = torch.inverse(trans)
  points_3d_src = kornia.geometry.depth.depth_to_3d(previous_depth_torch, full_K_torch, normalize_points=False)
  points_3d_src = points_3d_src.permute(0, 2, 3, 1)

  points_3d_src = points_3d_src * correct_vector

  points_3d_dst = kornia.geometry.linalg.transform_points(trans[:, None], points_3d_src)

  points_3d_dst = points_3d_dst * correct_vector

  points_3d_dst = points_3d_dst.view(batch_size, -1, 3)

  z_values = points_3d_dst[:, :, -1]
  z_values = torch.relu(z_values)
  sorting_indices = torch.argsort(z_values, descending=True)
  z_values = torch.gather(z_values, dim=1, index=sorting_indices)

  sorting_indices_for_points = torch.stack([sorting_indices] * 3, dim=-1)
  points_3d_dst = torch.gather(points_3d_dst, dim=1, index=sorting_indices_for_points)

  projections = torch.round(kornia.geometry.camera.perspective.project_points(points_3d_dst, full_K_torch.unsqueeze(1))).long()
  is_valid_below = (projections[:, :, 0] >= 0) & (projections[:, :, 1] >= 0)
  is_valid_above = (projections[:, :, 0] < original_width) & (projections[:, :, 1] < original_height)
  is_valid = is_valid_below & is_valid_above

  depth_hypothesis = torch.zeros(size=(batch_size, 1, original_height, original_width)).to(previous_depth_torch.device)
  for projection_index in range(0, batch_size):
    valid_points_zs = z_values[projection_index][is_valid[projection_index]]
    valid_projections = projections[projection_index][is_valid[projection_index]]
    i_s = valid_projections[:, 1]
    j_s = valid_projections[:, 0]
    ij_combined = i_s * original_width + j_s
    _, ij_combined_unique_indices = np.unique(ij_combined.cpu().numpy(), return_index=True)
    ij_combined_unique_indices = torch.from_numpy(ij_combined_unique_indices).long()
    i_s = i_s[ij_combined_unique_indices]
    j_s = j_s[ij_combined_unique_indices]
    valid_points_zs = valid_points_zs[ij_combined_unique_indices]
    torch.index_put_(depth_hypothesis[projection_index, 0], (i_s, j_s), valid_points_zs)
  return depth_hypothesis


def get_camera_data(camfile):
  camdata = []  # for each frame, record [num, left_para, right_para]
  with open(camfile, 'r') as f:
    d = f.readlines()
    num = len(d) // 4  # 4 rows for 1 frame
    for i in range(num):
      fn = int(d[4 * i].strip().split(' ')[-1])
      L_ori = d[4 * i + 1].strip()
      R_ori = d[4 * i + 2].strip()
      #camdata.append([str(fn).zfill(4), L_ori, R_ori])
      camdata.append([str(fn).zfill(4), parse_extinct_para(L_ori), parse_extinct_para(R_ori)])
  return camdata


def parse_extinct_para(ori_str):
  ori = ori_str.split(' ')
  assert (len(ori) == 17 and (ori[0] == 'L' or ori[0] == 'R'))
  para = np.zeros((4, 4))
  for i in range(1, len(ori)):
    para[(i - 1) // 4][(i - 1) % 4] = float(ori[i])
  return para


if __name__ == '__main__':
  # K = np.array([[1050, 0, 479.5], [0, 1050, 269.5], [0, 0, 1]]).astype(np.float32)  # intrinsics
  h, w = 1080, 1920
  hfov = 60 / 180 * np.pi
  K = np.array([[1050, 0, (w - 1) / 2], [0, 1050, (h - 1) / 2], [0, 0, 1]]).astype(np.float32)  # intrinsics
  f = w / 2 / (np.tan(hfov / 2))
  baseline = 1
  correct = torch.from_numpy(np.array([1, -1, -1]).astype(np.float32)).cuda()

  root_dir = '../../datasets/multi_view_huawei_data/huawei_SimpleParking/huawei_parking01/'
  prev_id = 129
  cur_id = prev_id + 5
  prev_left_name = os.path.join(root_dir, 'pinhole', 'ph_rgb8_' + str(prev_id) + '.jpg')
  cur_left_name = os.path.join(root_dir, 'pinhole', 'ph_rgb8_' + str(cur_id) + '.jpg')
  prev_disp_gt_name = os.path.join(root_dir, 'pinhole', 'ph_depth8_' + str(prev_id) + '.npz')
  cur_disp_gt_name = os.path.join(root_dir, 'pinhole', 'ph_depth8_' + str(cur_id) + '.npz')
  cam_file_name = os.path.join(root_dir, 'camera_poses.json')
  prev_left = cv2.imread(prev_left_name)
  cur_left = cv2.imread(cur_left_name)
  prev_depth_gt = np.load(prev_disp_gt_name)['arr_0']
  cur_depth_gt = np.load(cur_disp_gt_name)['arr_0']
  prev_depth_gt = carla_depth_decode(prev_depth_gt)
  cur_depth_gt = carla_depth_decode(cur_depth_gt)

  prev_left = prev_left.transpose((2, 0, 1)).astype(np.float32)
  cur_left = cur_left.transpose((2, 0, 1)).astype(np.float32)
  prev_left = torch.from_numpy(prev_left).unsqueeze(0)
  cur_left = torch.from_numpy(cur_left).unsqueeze(0)

  cur_depth_gt = torch.from_numpy(cur_depth_gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
  prev_depth_gt = torch.from_numpy(prev_depth_gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)

  cam_data = get_camera_data(cam_file_name)
  prev_pose = cam_data[prev_id][1].astype(np.float32)
  cur_pose = cam_data[cur_id][1].astype(np.float32)
  print(prev_pose)
  print(cur_pose)
  trans = np.matmul(np.linalg.inv(prev_pose), cur_pose)
  print(trans)

  prev_pose = torch.from_numpy(prev_pose).unsqueeze(0)
  cur_pose = torch.from_numpy(cur_pose).unsqueeze(0)
  #print(downsamp_K(torch.from_numpy(K).unsqueeze(0), 0.5))

  # prev_pose = torch.inverse(prev_pose)
  # cur_pose = torch.inverse(cur_pose)
  transformation = torch.bmm(torch.inverse(prev_pose), cur_pose)
  #transformation = torch.inverse(transformation)
  #output = warp_frame_depth(prev_left, cur_depth_gt, transformation, torch.from_numpy(K).unsqueeze(0))
  output = warp_frame_depth_correct(prev_left, cur_depth_gt, transformation, torch.from_numpy(K).unsqueeze(0))
  print(transformation)
  save_out = torch.cat([prev_left, cur_left, output], dim=3)
  save_out = save_out.numpy().squeeze()
  save_out = save_out.transpose((1, 2, 0)).astype(np.uint8)
  cv2.imwrite("test_repo_cur_gt.png", save_out)

  depth_pro = get_non_differentiable_rectangle_depth_estimation(cur_pose, prev_pose, prev_depth_gt, torch.from_numpy(K).unsqueeze(0), None, w, h, correct)
  print(depth_pro.shape, torch.max(depth_pro), torch.min(depth_pro))
  print(cur_depth_gt.shape, torch.max(cur_depth_gt), torch.min(cur_depth_gt))

  output2 = warp_frame_depth_correct(prev_left, depth_pro, transformation, torch.from_numpy(K).unsqueeze(0), correct)
  save_out = torch.cat([cur_left, output2], dim=3)
  save_out = save_out.numpy().squeeze()
  save_out = save_out.transpose((1, 2, 0)).astype(np.uint8)
  cv2.imwrite("test_repo_cur_gt-new.png", save_out)

  diff = torch.abs(output - cur_left).numpy().squeeze()
  diff = diff.transpose((1, 2, 0)).astype(np.uint8)
  cv2.imwrite("test_repo_diff.png", diff)

  save_out = torch.cat([cur_depth_gt, depth_pro], dim=3).numpy().squeeze()
  save_out = np.log(save_out + 1.0)
  save_out = ((save_out - save_out.min()) / (save_out.max() - save_out.min()) * 255).astype(np.uint8)
  cv2.imwrite("test_repo_cur_gt-dep.png", save_out)

# train_data = SceneflowTemporalDataloader(rootdir='../../datasets/SceneFlow/', frames=10, stage='training')
# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, num_workers=2, shuffle=False)