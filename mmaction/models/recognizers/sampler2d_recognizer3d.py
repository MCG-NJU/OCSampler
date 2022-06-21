import torch
import torch.nn.functional as F

import numpy as np

from ..registry import RECOGNIZERS
from .base import BaseRecognizer

from itertools import permutations
import math


@RECOGNIZERS.register_module()
class Sampler2DRecognizer3D(BaseRecognizer):

    def __init__(self,
                 sampler,
                 backbone,
                 cls_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 bp_mode='gradient_policy',
                 calc_mode='all',
                 num_segments=4,
                 num_test_segments=None,
                 use_sampler=False,
                 resize_px=None,
                 explore_rate=1.,
                 num_clips=1,
                 shuffle=False):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        self.resize_px = resize_px
        self.shuffle = shuffle
        self.num_segments = num_segments
        self.bp_mode = bp_mode
        self.num_clips = num_clips
        self.calc_mode = calc_mode
        assert bp_mode in ['gradient_policy', 'tsn', 'random', 'max']
        assert calc_mode in ['approximate', 'all']
        if self.num_segments <= 8:
            self.permute_index = list(permutations(list(range(self.num_segments)), self.num_segments))
            self.index_length = len(self.permute_index)
        if num_test_segments is None:
            self.num_test_segments = num_segments
        else:
            self.num_test_segments = num_test_segments
        self.explore_rate = explore_rate

    def sample(self, imgs, probs, test_mode=False, bp_mode='gradient_policy'):

        if test_mode:
            num_batches, original_segments = probs.shape
            sample_index = probs.topk(self.num_test_segments, dim=1)[1]
            if not self.shuffle:
                sample_index, _ = sample_index.sort(dim=1, descending=False)
            batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
            sample_probs = probs[batch_inds, sample_index]
            distribution = probs
            policy = None
        else:
            if bp_mode == 'gradient_policy':
                num_batches, original_segments = probs.shape
                sample_index = torch.multinomial(probs, self.num_segments, replacement=False)
                batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
                sample_probs = probs[batch_inds, sample_index]
                distribution = probs
                policy = None
                if not self.shuffle:
                    sample_index = sample_index.sort(dim=1)[0]
            elif bp_mode == 'max':
                sample_probs, sample_index = probs.topk(self.num_segments, dim=1)
                sample_index, _ = sample_index.sort(dim=1, descending=False)
                distribution = probs
                policy = None
            elif bp_mode == 'tsn':
                num_batches, original_segments = probs.shape
                num_len = original_segments // self.num_segments
                base_offset = torch.linspace(0, original_segments - num_len,
                                             steps=self.num_segments, dtype=int).repeat(num_batches, 1)
                rand_shift = np.random.randint(num_len, size=(num_batches, self.num_segments))
                sample_index = base_offset + rand_shift
                distribution = probs
                policy = None
                batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
                sample_probs = probs[batch_inds, sample_index]
            elif bp_mode == 'random':
                num_batches, original_segments = probs.shape
                sample_index = []
                for _ in range(num_batches):
                    sample_index.append(np.random.choice(original_segments, self.num_segments, replace=False))
                sample_index = torch.tensor(np.stack(sample_index))
                sample_index = sample_index.sort(dim=1)[0]
                distribution = probs
                policy = None
                batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
                sample_probs = probs[batch_inds, sample_index]

        # num_batches, num_segments
        num_batches = sample_index.shape[0]
        batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
        selected_imgs = imgs[batch_inds, sample_index]
        return selected_imgs, distribution, policy, sample_index, sample_probs

    def forward_sampler(self, imgs, num_batches, test_mode=False, **kwargs):
        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            view_ptr = 0
            probs = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                prob = self.sampler(batch_imgs)
                probs.append(prob)
                view_ptr += self.max_testing_views
            probs = torch.cat(probs)
        else:
            if self.resize_px is None:
                probs = self.sampler(imgs)
            else:
                probs = self.sampler(F.interpolate(imgs, size=self.resize_px))
        imgs = imgs.reshape((num_batches, -1) + (imgs.shape[-3:]))
        if self.sampler.freeze_all:
            probs = probs.detach()

        selected_imgs, distribution, policy, sample_index, sample_probs = self.sample(imgs, probs, test_mode)
        bs_imgs, bs_distribution, bs_policy, bs_sample_index, bs_sample_probs = self.sample(imgs, probs, test_mode, self.bp_mode)

        return selected_imgs, bs_imgs, distribution, policy, sample_index, sample_probs

    def forward_train(self, imgs, labels, **kwargs):
        num_batches = imgs.shape[0]

        if hasattr(self, 'sampler'):
            imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))

            if self.sampler.freeze_all:
                imgs, bs_imgs, distribution, policy, sample_index, sample_probs = self.forward_sampler(imgs, num_batches, True)
            else:
                imgs, bs_imgs, distribution, policy, sample_index, sample_probs = self.forward_sampler(imgs, num_batches, **kwargs)


        imgs = imgs.transpose(1, 2).contiguous()
        bs_imgs = bs_imgs.transpose(1, 2).contiguous()

        losses = dict()
        x = self.extract_feat(imgs)
        bs_x = self.extract_feat(bs_imgs)

        cls_score = self.cls_head(x)
        bs_cls_score = self.cls_head(bs_x)

        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        if gt_labels.shape == torch.Size([]):
            gt_labels = gt_labels.unsqueeze(0)

        reward_list = []
        origin_reward_list = []
        bs_reward_list = []
        for i in range(gt_labels.shape[0]):
            gt_label = gt_labels[i]

            gt_score = F.softmax(cls_score[i], dim=0)[gt_label].unsqueeze(0)
            max_score, max_index = torch.max(F.softmax(cls_score[i], dim=0), 0)
            if max_index == gt_label:
                reward = gt_score
            else:
                reward = gt_score - max_score

            bs_gt_score = F.softmax(bs_cls_score[i], dim=0)[gt_label].unsqueeze(0)
            bs_max_score, bs_max_index = torch.max(F.softmax(bs_cls_score[i], dim=0), 0)
            if bs_max_index == gt_label:
                bs_reward = bs_gt_score
            else:
                bs_reward = bs_gt_score - bs_max_score
            bs_reward = bs_reward.detach()
            advantage = reward - bs_reward

            origin_reward_list.append(reward)
            bs_reward_list.append(bs_reward)
            reward_list.append(advantage)
        reward = torch.cat(reward_list).clone().detach()
        origin_reward = torch.cat(origin_reward_list).clone().detach()
        bs_reward = torch.cat(bs_reward_list).clone().detach()

        loss_cls['reward'] = reward.mean()
        loss_cls['origin_reward'] = origin_reward.mean()
        loss_cls['bs_reward'] = bs_reward.mean()

        if self.sampler.freeze_all:
            loss_cls['all_probs'] = sample_probs.clone().detach().mean()
            loss_cls['loss_cls'] = loss_cls['loss_cls']
            losses.update(loss_cls)
            return losses

        entropy = torch.sum(-distribution * torch.log(distribution), dim=1)

        sample_probs = sample_probs.clamp(1e-15, 1-1e-15)

        if self.calc_mode == 'approximate':
            eps = 1e-9
            sample_probs, _ = torch.sort(sample_probs, dim=1)
            sample_probs = sample_probs[:, :min(self.num_segments, 8)]

            ones = torch.ones(sample_probs.shape[0], sample_probs.shape[1], device=imgs.device)
            sample_probs_reverse = torch.flip(sample_probs, [-1])

            divisor = ones - F.pad(sample_probs.cumsum(dim=1)[:, :-1], (1, 0))
            divisor_reverse = ones - F.pad(sample_probs_reverse.cumsum(dim=1)[:, :-1], (1, 0))

            sample_probs_1 = torch.cumprod(sample_probs / divisor, dim=1)[:, -1]
            sample_probs_2 = torch.cumprod(sample_probs_reverse / divisor_reverse, dim=1)[:, -1]

            sample_probs = (sample_probs_1 + sample_probs_2) / 2 * math.factorial(min(self.num_segments, 8))
            policy_cross_entropy = torch.log(sample_probs)
        elif self.calc_mode == 'all':
            dividend = torch.cumprod(sample_probs, dim=1)[:, -1]
            permute_index = torch.tensor(self.permute_index, dtype=torch.long, device=imgs.device)
            multi_index = permute_index.repeat(1, num_batches).reshape(self.index_length * num_batches, self.num_segments)
            multi_probs = sample_probs.repeat(self.index_length, 1)
            batch_inds = torch.arange(self.index_length * num_batches, device=imgs.device).unsqueeze(-1).expand_as(multi_index)
            multi_probs = multi_probs[batch_inds, multi_index]
            multi_probs = multi_probs.reshape(-1, num_batches, self.num_segments)
            ones = torch.ones(self.index_length, num_batches, self.num_segments, device=imgs.device)
            divisor = ones - F.pad(multi_probs.cumsum(dim=2)[:, :, :-1], (1, 0))
            divisor = torch.cumprod(divisor, dim=2)[:, :, -1]
            sample_probs = (dividend / divisor).sum(dim=0)

            policy_cross_entropy = torch.log(sample_probs)

        if self.cls_head.final_loss:
            loss_cls_ = -(reward * policy_cross_entropy).mean()
            loss_cls['entropy_fc'] = loss_cls['loss_cls'].clone().detach()
            loss_cls['sampler_loss'] = loss_cls_.clone().detach()
            loss_cls['loss_cls'] = loss_cls_ + loss_cls['loss_cls']
            loss_cls['all_probs'] = sample_probs.clone().detach().mean()
        else:
            loss_cls_ = -(reward * policy_cross_entropy + self.explore_rate * entropy).mean()
            loss_cls['loss_cls'] = loss_cls_
            loss_cls['entropy'] = entropy.clone().detach() * self.explore_rate
            loss_cls['explore_rate'] = torch.tensor([self.explore_rate], device=imgs.device)
            loss_cls['all_probs'] = sample_probs.clone().detach().mean()

        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs, **kwargs):
        num_batches = imgs.shape[0]
        num_clips = imgs.shape[1] // self.sampler.total_segments

        imgs = imgs.reshape((-1,) + (imgs.shape[-3:]))

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            view_ptr = 0
            cls_scores = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                batch_imgs, _, distribution, _, sample_index, _ = self.forward_sampler(batch_imgs, self.max_testing_views//self.sampler.total_segments, test_mode=True)
                batch_imgs = batch_imgs.transpose(1, 2).contiguous()
                x = self.extract_feat(batch_imgs)
                cls_score = self.cls_head(x)
                cls_scores.append(cls_score)
                view_ptr += self.max_testing_views
            cls_score = torch.cat(cls_scores)
        else:
            imgs, _, distribution, _, sample_index, _ = self.forward_sampler(imgs, num_batches * num_clips, test_mode=True)
            imgs = imgs.transpose(1, 2).contiguous()
            x = self.extract_feat(imgs)
            cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score, num_clips)
        return cls_score.cpu().numpy()

    def forward_test(self, imgs, **kwargs):
        return self._do_test(imgs, **kwargs)

    def forward_gradcam(self, imgs):
        pass
