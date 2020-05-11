# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os

import cv2
import matplotlib.figure as mplfigure
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch import nn

from .file_io import PathManager


class Visualizer:
    r"""Visualize images(activation map) ranking list of features generated by reid models."""

    def __init__(self, dataset):
        self.dataset = dataset

    def get_model_output(self, all_ap, sim, q_pids, g_pids, q_camids, g_camids):
        self.all_ap = all_ap
        self.sim = sim
        self.q_pids = q_pids
        self.g_pids = g_pids
        self.q_camids = q_camids
        self.g_camids = g_camids

        self.indices = np.argsort(1 - sim, axis=1)
        self.matches = (g_pids[self.indices] == q_pids[:, np.newaxis]).astype(np.int32)

        self.num_query = len(q_pids)

    def get_matched_result(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)
        cmc = self.matches[q_index][keep]
        sort_idx = order[keep]
        return cmc, sort_idx

    def save_rank_result(self, query_indices, output, max_rank=5, actmap=False):
        fig, axes = plt.subplots(1, max_rank + 1, figsize=(3 * max_rank, 6))
        # fig.suptitle('query/AP/camid  sim/true(false)/camid')
        for cnt, q_idx in enumerate(tqdm.tqdm(query_indices)):
            all_imgs = []
            cmc, sort_idx = self.get_matched_result(q_idx)
            query_info = self.dataset[q_idx]
            query_img = query_info['images']
            cam_id = query_info['camid']
            query_name = query_info['img_path'].split('/')[-1]
            all_imgs.append(query_img)
            query_img = np.rollaxis(np.asarray(query_img.numpy(), dtype=np.uint8), 0, 3)
            axes.flat[0].imshow(query_img)
            axes.flat[0].set_title('{}/{:.2f}/cam{}'.format(query_name, self.all_ap[q_idx], cam_id))
            axes.flat[0].axis("off")
            # print('query' + query_info['img_path'].split('/')[-1])
            for i in range(max_rank):
                g_idx = self.num_query + sort_idx[i]
                gallery_info = self.dataset[g_idx]
                gallery_img = gallery_info['images']
                cam_id = gallery_info['camid']
                all_imgs.append(gallery_img)
                gallery_img = np.rollaxis(np.asarray(gallery_img, dtype=np.uint8), 0, 3)
                if cmc[i] == 1:
                    label = 'true'
                    axes.flat[i + 1].add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                                             height=gallery_img.shape[0] - 1, edgecolor=(1, 0, 0),
                                                             fill=False, linewidth=5))
                else:
                    label = 'false'
                    axes.flat[i + 1].add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                                             height=gallery_img.shape[0] - 1,
                                                             edgecolor=(0, 0, 1), fill=False, linewidth=5))
                axes.flat[i + 1].imshow(gallery_img)
                # print('/'.join(gallery_info['img_path'].split('/')[-2:]))
                axes.flat[i + 1].set_title(f'{self.sim[q_idx, sort_idx[i]]:.3f}/{label}/cam{cam_id}')
                axes.flat[i + 1].axis("off")
            # if actmap:
            #     act_outputs = []
            #
            #     def hook_fns_forward(module, input, output):
            #         act_outputs.append(output.cpu())
            #
            #     all_imgs = np.stack(all_imgs, axis=0)  # (b, 3, h, w)
            #     all_imgs = torch.from_numpy(all_imgs).float()
            #     # normalize
            #     all_imgs = all_imgs.sub_(self.mean).div_(self.std)
            #     sz = list(all_imgs.shape[-2:])
            #     handle = m.base.register_forward_hook(hook_fns_forward)
            #     with torch.no_grad():
            #         _ = m(all_imgs.cuda())
            #     handle.remove()
            #     acts = self.get_actmap(act_outputs[0], sz)
            #     for i in range(top + 1):
            #         axes.flat[i].imshow(acts[i], alpha=0.3, cmap='jet')
            plt.tight_layout()
            filepath = os.path.join(output, "{}.jpg".format(cnt))
            fig.savefig(filepath)
            plt.cla()

    def vis_ranking_list(self, output, num_vis=100, rank_sort='ascending', max_rank=5, actmap=False):
        """
        Args:
            output (str): a file or directory to save rankling list result.
            rank_sort (str): save visualization results by which order,
                if rank_sort is ascending, AP from low to high, vice versa.
            num_vis (int):
            max_rank (int):
            actmap (bool):
        """
        assert rank_sort in ['ascending', 'descending'], "{} not match [ascending, descending]".format(rank_sort)
        PathManager.mkdirs(output)

        query_indices = np.argsort(self.all_ap)
        if rank_sort == 'descending':   query_indices = query_indices[::-1]

        query_indices = query_indices[:num_vis]
        self.save_rank_result(query_indices, output, max_rank, actmap)

    def plot_roc_curve(self):
        pos_sim, neg_sim = [], []
        for i, q in enumerate(self.q_pids):
            cmc, sort_idx = self.get_matched_result(i)  # remove same id in same camera
            for j in range(len(cmc)):
                if cmc[j] == 1:
                    pos_sim.append(self.sim[i, sort_idx[j]])
                else:
                    neg_sim.append(self.sim[i, sort_idx[j]])
        fig = plt.figure(figsize=(10, 5))
        plt.hist(pos_sim, bins=80, alpha=0.7, density=True, color='red', label='positive')
        plt.hist(neg_sim, bins=80, alpha=0.5, density=True, color='blue', label='negative')
        plt.xticks(np.arange(-0.3, 0.8, 0.1))
        plt.title('positive and negative pair distribution')
        return pos_sim, neg_sim

    def plot_camera_dist(self):
        same_cam, diff_cam = [], []
        for i, q in enumerate(self.q_pids):
            q_camid = self.q_camids[i]

            order = self.indices[i]
            same = (self.g_pids[order] == q) & (self.g_camids[order] == q_camid)
            diff = (self.g_pids[order] == q) & (self.g_camids[order] != q_camid)
            sameCam_idx = order[same]
            diffCam_idx = order[diff]

            same_cam.extend(self.sim[i, sameCam_idx])
            diff_cam.extend(self.sim[i, diffCam_idx])

        fig = mplfigure(figsize=(10, 5))
        plt.hist(same_cam, bins=80, alpha=0.7, density=True, color='red', label='same camera')
        plt.hist(diff_cam, bins=80, alpha=0.5, density=True, color='blue', label='diff camera')
        plt.xticks(np.arange(0.1, 1.0, 0.1))
        plt.title('positive and negative pair distribution')
        return fig

    def get_actmap(self, features, sz):
        """
        :param features: (1, 2048, 16, 8) activation map
        :return:
        """
        features = (features ** 2).sum(1)  # (1, 16, 8)
        b, h, w = features.size()
        features = features.view(b, h * w)
        features = nn.functional.normalize(features, p=2, dim=1)
        acts = features.view(b, h, w)
        all_acts = []
        for i in range(b):
            act = acts[i].numpy()
            act = cv2.resize(act, (sz[1], sz[0]))
            act = 255 * (act - act.max()) / (act.max() - act.min() + 1e-12)
            act = np.uint8(np.floor(act))
            all_acts.append(act)
        return all_acts
