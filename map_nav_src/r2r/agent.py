import json
import os
import cv2
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad

from torchvision import transforms
from PIL import Image


class GMapNavAgent(Seq2SeqAgent):

    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        self.transform_img = torch.nn.Sequential(transforms.ConvertImageDtype(torch.float),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])).cuda()
        # buffer
        self.scanvp_cands = {}

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool_)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _visual_instruction_variable(self, obs):
        batch_size = len(obs)
        vis_instr_imgs = []
        if self.args.prompt_enhance and self.env.name == 'aug':
            angle_list = []
            rotation_ops = {0: lambda x: x,
                            90: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
                            180: lambda x: cv2.rotate(x, cv2.ROTATE_180),
                            270: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)}
        for i, ob in enumerate(obs):
            scan = ob['scan']
            scan_imgs = self.scan_to_images[scan]
            path_imgs = ob['gt_path']
            pre_ind = -1
            prev_location = None
            radius = 9
            flag = False
            img_dict = {}
            if self.args.prompt_enhance and self.env.name == 'aug':
                angle = random.choice([0, 90, 180, 270])  # 随机选择旋转角度
                angle_list.append(angle)
            for k, vp in enumerate(path_imgs):
                location = self.node2pix[scan][vp]
                now_ind = location[2]
                # print(location)
                if now_ind != pre_ind:
                    if pre_ind != -1:
                        if digit_points:
                            img = self.center_crop_img(img, digit_points)
                        if self.args.prompt_enhance and self.env.name == 'aug':
                            img = rotation_ops[angle](img)
                        img_dict[pre_ind] = img
                    digit_points = []
                    img = scan_imgs[now_ind].copy()
                #     if self.args.prompt_enhance and self.env.name == 'aug':
                #         img_height, img_width = img.shape[:2]
                #         img = rotation_ops[angle](img)
                # if self.args.prompt_enhance and self.env.name == 'aug':
                #     rotated_x, rotated_y = self.rotate_point((location[0][0], location[0][1]), angle, img_width, img_height)
                #     location[0][0], location[0][1] = rotated_x, rotated_y
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
                img = self.center_crop_img(img, digit_points)
            if self.args.prompt_enhance and self.env.name == 'aug':
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
                crop_img = self.crop_non_black(img)

                # lower_green = np.array([0, 250, 0])
                # upper_green = np.array([10, 255, 10])
                # mask = cv2.inRange(crop_img, lower_green, upper_green)
                # mask_3ch = cv2.merge([mask, mask, mask])
                # crop_img = cv2.bitwise_and(crop_img, mask_3ch)

                rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                # norm_img = self.transform_img(rgb_img)
                rgb_img = cv2.resize(rgb_img, (224, 224))
                norm_img = torch.from_numpy(rgb_img).permute(2, 0, 1).cuda()
                batch_images.append(norm_img)
                batch_indexes.append(i)

                # save_path = f"{i}_{flo_id}.png"
                # success = cv2.imwrite(save_path, crop_img)
                # print(obs[i]['instructions'])
                # if self.args.prompt_enhance and self.env.name == 'aug':
                #     print(angle_list[i])
                # print(f"{i}_{flo_id}.png")
                # print()

                batch_floors.append(flo_id)
        # assert len(batch_images) == len(batch_indexes) == len(batch_floors)
        batch_images = torch.stack(batch_images, 0)
        batch_images = self.transform_img(batch_images)

        # indices = torch.randperm(batch_images.size(0)).to(batch_images.device)
        # shuffled_images = batch_images[indices]
        # batch_images = shuffled_images

        batch_indexes = torch.LongTensor(batch_indexes).cuda()
        batch_floors = torch.LongTensor(batch_floors).cuda()
        return {
            'batch_vis_instrs': batch_images, 'batch_indexes': batch_indexes, 'batch_floors': batch_floors
        }

    def rotate_point(self, point, angle, img_width, img_height):
        """与cv2.rotate()完全匹配的点旋转"""
        x, y = point
        if angle == 90:  # ROTATE_90_CLOCKWISE
            return (img_height - 1 - y, x)
        elif angle == 180:  # ROTATE_180
            return (img_width - 1 - x, img_height - 1 - y)
        elif angle == 270:  # ROTATE_90_COUNTERCLOCKWISE
            return (y, img_width - 1 - x)
        else:
            return (x, y)

    # def add_salt_pepper_noise(self, image, noise_ratio=0.2):
    #     h, w = image.shape[:2]
    #     mask = np.random.choice([0, 1, 2], size=(h, w), p=[noise_ratio / 2, noise_ratio / 2, 1 - noise_ratio])
    #     if len(image.shape) == 3:
    #         salt = np.array([255, 255, 255], dtype=image.dtype)
    #         pepper = np.array([0, 0, 0], dtype=image.dtype)
    #         result = np.where(mask[..., None] == 0, pepper,
    #                           np.where(mask[..., None] == 1, salt, image))
    #     else:
    #         result = np.where(mask == 0, 0, np.where(mask == 1, 255, image))
    #     return result

    # def transform_img(self, img):
    #     transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     return transform(Image.fromarray(img))

    # def transform_img(self, img):
    #     transform = torch.nn.Sequential(
    #         transforms.ConvertImageDtype(torch.float),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     )
    #     return transform(img)

    def crop_non_black(self, img):
        # 找出所有非黑色像素的位置（注意是 BGR）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)  # 返回非零点坐标
        if coords is None:
            return img  # 全黑图，直接返回原图
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y + h, x:x + w]
        return cropped

    def center_crop_img(self, img, digit_points):
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

    def _test_visual_instruction_variable(self, obs):
        batch_size = len(obs)
        vis_instr_imgs = []
        for i, ob in enumerate(obs):
            img_dict = self.env.test_vps[ob['path_id']]
            vis_instr_imgs.append(img_dict)

        batch_images = []
        batch_indexes = []
        batch_floors = []
        for i, img_dict in enumerate(vis_instr_imgs):
            last_two = list(img_dict.items())[-2:]
            img_dict = dict(last_two)
            for flo_id, img in img_dict.items():
                crop_img = self.crop_non_black(img)
                rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                # norm_img = self.transform_img(rgb_img)
                rgb_img = cv2.resize(rgb_img, (224, 224))
                norm_img = torch.from_numpy(rgb_img).permute(2, 0, 1).cuda()
                batch_images.append(norm_img)
                batch_indexes.append(i)

                # save_path = f"{i}_{flo_id}.png"
                # success = cv2.imwrite(save_path, crop_img)
                # print(obs[i]['instructions'])
                # print(f"{i}_{flo_id}.png")
                # print()

                batch_floors.append(flo_id)
        # assert len(batch_images) == len(batch_indexes) == len(batch_floors)
        batch_images = torch.stack(batch_images, 0)
        batch_images = self.transform_img(batch_images)

        # indices = torch.randperm(batch_images.size(0)).to(batch_images.device)
        # shuffled_images = batch_images[indices]
        # batch_images = shuffled_images

        batch_indexes = torch.LongTensor(batch_indexes).cuda()
        batch_floors = torch.LongTensor(batch_floors).cuda()
        return {
            'batch_vis_instrs': batch_images, 'batch_indexes': batch_indexes, 'batch_floors': batch_floors
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )  # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens + 1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0  # Stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                   + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(
            self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0  # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0  # Stop if arrived
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan],
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][
                                                                   1:],
                                        ob['gt_path'],
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                           + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:  # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'path_id': ob['path_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        # language_inputs = self._language_variable(obs)
        # txt_embeds = self.vln_bert('language', language_inputs)
        if self.env.name == 'test':
            vis_instr_inputs = self._test_visual_instruction_variable(obs)
        else:
            vis_instr_inputs = self._visual_instruction_variable(obs)
        txt_embeds, txt_masks = self.vln_bert('vis_instr', vis_instr_inputs)

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            # nav_inputs.update({
            #     'txt_embeds': txt_embeds,
            #     'txt_masks': None,
            # })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }

            if train_ml is not None:
                # Supervised training
                if self.args.dataset == 'r2r':
                    # nav_targets = self._teacher_action(
                    #     obs, nav_vpids, ended,
                    #     visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                    # )
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                elif self.args.dataset == 'r4r':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())

            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs[
                        'gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample':  # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

                    # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj
