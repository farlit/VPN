import os
import cv2
import json
from typing import Any, Dict, List
import torch
import torch.distributed as dist
from collections import defaultdict
from torchvision import transforms
from PIL import Image
import numpy as np
import copy
import math
import random

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
    max_length: int = 512,
    pad_id: int = 0,
):
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure."""
    if instruction_sensor_uuid not in observations[0]:
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
            if len(token) < max_length:
                token += [pad_id] * (max_length - len(token))
            observations[i][instruction_sensor_uuid] = token
        else:
            break
    return observations

def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    dist.all_gather(gather_t, tensor)
    return gather_t

def dis_to_con(path, amount=0.25):
    starts = path[:-1]
    ends = path[1:]
    new_path = [path[0]]
    for s, e in zip(starts,ends):
        vec = np.array(e) - np.array(s)
        ratio = amount/np.linalg.norm(vec[[0,2]])
        unit = vec*ratio
        times = int(1/ratio)
        for i in range(times):
            if i != times - 1:
                location = np.array(new_path[-1])+unit
                new_path.append(location.tolist())
        new_path.append(e)
    
    return new_path

def get_camera_orientations12():
    base_angle_deg = 30
    base_angle_rad = math.pi / 6
    orient_dict = {}
    for k in range(1,12):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict

def load_scan2image(floors_dir):
    scan_to_images = defaultdict(list)
    for floor_name in sorted(os.listdir(floors_dir), key=lambda x: int(x.split("_")[-1])):
        floor_path = os.path.join(floors_dir, floor_name)
        for filename in os.listdir(floor_path):
            if filename.endswith(".png"):
                parts = filename.split("_")
                scan_name = "_".join(parts[:-1])  # 去掉最后一部分的楼层编号
                full_path = os.path.join(floor_path, filename)
                img = cv2.imread(full_path)
                if img is None:
                    print(f"[Warning] 图像加载失败: {full_path}")
                else:
                    scan_to_images[scan_name].append(img)
    # for scan, img_list in scan_to_images.items():
    #     print(f"Scan {scan} has {len(img_list)} images:")
    return scan_to_images

def load_traj2locs(r2r_dir):

    traj2scanvps = {}
    splits = ['train', 'val_seen', 'val_unseen', 'aug']
    for split in splits:
        if split != 'aug':
            scanvps = json.load(open(r2r_dir.format(split)))
        else:
            dir_path = os.path.dirname(r2r_dir)
            new_path = os.path.join(dir_path, 'prevalent_aug_train_enc.json')
            scanvps = json.load(open(new_path))
        for episode in scanvps:
            path_id = episode['path_id']
            traj2scanvps[path_id] = {
                'scan': episode['scan'],
                'vps': episode['path']
            }

    return traj2scanvps

def extract_visual_instr_tokens(
    observations,
    node2pix ,
    scan2images,
    traj2scanvps,
    device,
    mode
):
    vis_instr_imgs = []
    rotation_ops = {0: lambda x: x,
                    90: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
                    180: lambda x: cv2.rotate(x, cv2.ROTATE_180),
                    270: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    for i, observation in enumerate(observations):
        traj_id = observation['instruction']['trajectory_id']
        scanvps = traj2scanvps[traj_id]
        scan = scanvps['scan']
        vps = scanvps['vps']
        pre_ind = -1
        prev_location = None
        radius = 9
        img_dict = {}
        angle = random.choice([0, 90, 180, 270])
        for k, vp in enumerate(vps):
            location = node2pix[scan][vp]
            now_ind = location[2]
            # print(location)
            if now_ind != pre_ind:
                if pre_ind != -1:
                    if digit_points:
                        img = center_crop_img(img, digit_points)
                    if mode == 'train':
                        img = rotation_ops[angle](img)
                    img_dict[pre_ind] = img
                digit_points = []
                img = scan2images[scan][now_ind].copy()
            curr_point = np.array([location[0][0], location[0][1]])

            # cv2.circle(img, (location[0][0], location[0][1]),
            #             radius=radius, color=(255, 255, 255),
            #             thickness=-1,
            #             lineType=cv2.LINE_AA)

            # cv2.putText(img,
            #             text=str(k+1),org=(location[0][0] - 7,
            #             location[0][1] + 7),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.65,
            #             color=(0, 0, 255),
            #             thickness=1,
            #             lineType=cv2.LINE_AA)

            digit_points.append((location[0][0], location[0][1]))
            if prev_location is not None and now_ind == pre_ind:
                prev_point = np.array([prev_location[0][0], prev_location[0][1]])
                direction = curr_point - prev_point
                length = np.linalg.norm(direction)
                if length > 1e-5:
                    unit_dir = direction / length
                    start_point = (prev_point + unit_dir * radius).astype(int)
                    end_point = (curr_point - unit_dir * radius).astype(int)

                    cv2.arrowedLine(img,
                                    tuple(start_point),
                                    tuple(end_point),
                                    color=(0, 255, 0),
                                    thickness=2,
                                    line_type=cv2.LINE_AA,
                                    tipLength=0.15)

            pre_ind = now_ind
            prev_location = location
        if digit_points:
            img = center_crop_img(img, digit_points)
        if mode == 'train':
            img = rotation_ops[angle](img)
        img_dict[pre_ind] = img
        vis_instr_imgs.append(img_dict)

    batch_images = []
    batch_indexes = []
    batch_floors = []
    for i, img_dict in enumerate(vis_instr_imgs):
        last_two = list(img_dict.items())[-2:]
        img_dict = dict(last_two)
        for flo_id, img in img_dict.items():
            crop_img = crop_non_black(img)
            rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (224, 224))
            norm_img = torch.from_numpy(rgb_img).permute(2, 0, 1).cuda()
            batch_images.append(norm_img)
            batch_indexes.append(i)

            # save_path = f"{i}_{flo_id}.png"
            # success = cv2.imwrite(save_path, crop_img)
            # print(observations[i]['instruction'])
            # print(f"{i}_{flo_id}.png")
            # print()

            batch_floors.append(flo_id)
        del observations[i]['instruction']

    # assert len(batch_images) == len(batch_indexes) == len(batch_floors)
    batch_images = torch.stack(batch_images, 0).to(device)
    batch_images = transform_img(batch_images)

    # indices = torch.randperm(batch_images.size(0)).to(batch_images.device)
    # shuffled_images = batch_images[indices]
    # batch_images = shuffled_images

    batch_indexes = torch.LongTensor(batch_indexes).to(device)
    batch_floors = torch.LongTensor(batch_floors).to(device)

    return observations, batch_images, batch_indexes, batch_floors

transform_img = torch.nn.Sequential(transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])).cuda()

def crop_non_black(img):
    # 找出所有非黑色像素的位置（注意是 BGR）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)  # 返回非零点坐标
    if coords is None:
        return img  # 全黑图，直接返回原图
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]
    return cropped

def center_crop_img(img, digit_points):
    xs = [pt[0] for pt in digit_points]
    ys = [pt[1] for pt in digit_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    side = max(w, h) + 60
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(cx + side // 2, img.shape[1])
    y2 = min(cy + side // 2, img.shape[0])
    img = img[y1:y2, x1:x2]
    return img