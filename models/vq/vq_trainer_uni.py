import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from models.losses import Geometric_Losses, Inter_Losses
from utils.utils import print_current_loss


def def_value():
    return 0.0


class UnifiedRVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            self.geo_losses = Geometric_Losses(
                args.recons_loss,
                self.opt.joints_num,
                self.opt.dataset_name,
                self.device,
                data_rep=getattr(self.opt, 'data_rep', 'rot6d'),
                fast_mode=getattr(self.opt, 'fast_mode', False),
            )
            self.inter_losses = Inter_Losses(
                args.recons_loss,
                self.opt.joints_num,
                self.opt.dataset_name,
                self.device,
                data_rep=getattr(self.opt, 'data_rep', 'rot6d'),
                fast_mode=getattr(self.opt, 'fast_mode', False),
            )

    def _compute_caption_loss(self, caption_logits, caption_tokens):
        if caption_logits is None:
            return torch.tensor(0.0, device=self.device)
        targets = caption_tokens[:, 1:]
        return F.cross_entropy(
            caption_logits.reshape(-1, caption_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.vq_model.caption_pad_id,
        )

    def forward(self, batch_data):
        paired_motions = None
        if len(batch_data) == 5:
            motions, paired_motions, captions, caption_tokens, caption_lengths = batch_data
            paired_motions = paired_motions.detach().to(self.device).float()
        else:
            motions, captions, caption_tokens, caption_lengths = batch_data

        motions = motions.detach().to(self.device).float()
        caption_tokens = caption_tokens.to(self.device).long()

        # Motion tokenization remains single-person; caption supervision sees the paired interaction branch.
        outputs = self.vq_model(motions, caption_tokens=caption_tokens, caption_x=paired_motions, verbose=False)
        pred_motion = outputs['pred_motion']
        loss_commit = outputs['commit_loss']
        perplexity = outputs['perplexity']
        caption_loss = self._compute_caption_loss(outputs['caption_logits'], caption_tokens)

        loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, _, _ = self.geo_losses.forward(motions, pred_motion)
        loss = (
            loss_rec
            + (self.opt.commit * loss_commit)
            + (self.opt.loss_explicit * loss_explicit)
            + (self.opt.loss_vel * loss_vel)
            + (self.opt.loss_bn * loss_bn)
            + (self.opt.loss_geo * loss_geo)
            + (self.opt.loss_fc * loss_fc)
            + (self.opt.caption_loss_weight * caption_loss)
        )

        return {
            'loss': loss,
            'loss_rec': loss_rec,
            'loss_explicit': loss_explicit,
            'loss_vel': loss_vel,
            'loss_bn': loss_bn,
            'loss_geo': loss_geo,
            'loss_fc': loss_fc,
            'loss_commit': loss_commit,
            'loss_caption': caption_loss,
            'perplexity': perplexity,
        }

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader):
        self.vq_model.to(self.device)

        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'\nTotal Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        self.opt.warm_up_iter = len(train_loader) // 4
        self.opt.log_every = max(1, len(train_loader) // 10)
        self.opt.save_latest = max(1, len(train_loader) // 2)
        print(f'Warm Up Iters: {self.opt.warm_up_iter}, Log Every: {self.opt.log_every} iters, Save every: {self.opt.save_latest} iters')

        self.opt.milestones = [int(total_iters * 0.7), int(total_iters * 0.85)]
        print(f"LR milestones: {self.opt.milestones}\n")

        trainable_params = [param for param in self.vq_model.parameters() if param.requires_grad]
        self.opt_vq_model = optim.AdamW(
            trainable_params,
            lr=self.opt.lr,
            betas=(0.9, 0.99),
            weight_decay=self.opt.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        logs = defaultdict(def_value, OrderedDict())
        min_val_loss = np.inf

        while epoch < self.opt.max_epoch:
            epoch += 1
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss_dict = self.forward(batch_data)
                if not all(torch.isfinite(value).all() for value in loss_dict.values() if torch.is_tensor(value)):
                    print(f"[Warning] Skip non-finite training batch at iter {it}")
                    self.opt_vq_model.zero_grad(set_to_none=True)
                    continue
                self.opt_vq_model.zero_grad()
                loss_dict['loss'].backward()
                clip_grad_norm_(self.vq_model.parameters(), max_norm=1.0)
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()

                for key, value in loss_dict.items():
                    logs[key] += value.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            val_logs = defaultdict(list)
            with torch.no_grad():
                for batch_data in val_loader:
                    loss_dict = self.forward(batch_data)
                    for key, value in loss_dict.items():
                        val_logs[key].append(value.item())

            for key, values in val_logs.items():
                self.logger.add_scalar(f'Val/{key}', sum(values) / len(values), epoch)

            mean_val_loss = sum(val_logs['loss']) / len(val_logs['loss'])
            print(
                'Validation Loss: %.5f Reconstruction: %.5f, Explicit: %.5f, Velocity: %.5f, Bone Length: %.5f, Geodesic: %.5f, Foot Contact: %.5f, Commit: %.5f, Caption: %.5f'
                % (
                    mean_val_loss,
                    sum(val_logs['loss_rec']) / len(val_logs['loss_rec']),
                    sum(val_logs['loss_explicit']) / len(val_logs['loss_explicit']),
                    sum(val_logs['loss_vel']) / len(val_logs['loss_vel']),
                    sum(val_logs['loss_bn']) / len(val_logs['loss_bn']),
                    sum(val_logs['loss_geo']) / len(val_logs['loss_geo']),
                    sum(val_logs['loss_fc']) / len(val_logs['loss_fc']),
                    sum(val_logs['loss_commit']) / len(val_logs['loss_commit']),
                    sum(val_logs['loss_caption']) / len(val_logs['loss_caption']),
                )
            )

            if mean_val_loss < min_val_loss:
                min_val_loss = mean_val_loss
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')
