"""
Loss functions for Prototypical Networks.

PrototypicalLoss  –  Cross-entropy (with label smoothing) + optional supervised
                     contrastive regularisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, contrastive_weight=0.1, temperature=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    def supervised_contrastive_loss(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        batch_size = embeddings.size(0)

        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)
        eye = torch.eye(batch_size, device=embeddings.device)
        mask = mask - eye

        exp_sim = torch.exp(sim_matrix) * (1 - eye)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mask_sum = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1) / mask_sum
        return loss.mean()

    # ------------------------------------------------------------------ #
    def forward(self, logits, labels, embeddings=None, all_labels=None):
        ce_loss = self.ce_loss(logits, labels)
        total_loss = ce_loss
        contrastive_loss = torch.tensor(0.0)

        if embeddings is not None and all_labels is not None and self.contrastive_weight > 0:
            contrastive_loss = self.supervised_contrastive_loss(embeddings, all_labels)
            total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        return total_loss, ce_loss, contrastive_loss
