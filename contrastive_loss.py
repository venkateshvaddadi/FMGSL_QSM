#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:51:39 2026

@author: venkatesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainClusterContrastiveLoss(nn.Module):
    """
    Implements a domain-targeted contrastive loss to explicitly form 
    a compact 3T cluster and a compact 7T cluster in the semantic space,
    maximizing the distance between the two domains.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_3t, z_7t):
        """
        Args:
            z_3t (Tensor): Reconstructed 3T embeddings [N, 768]
            z_7t (Tensor): High-quality 7T reference embeddings [N, 768]
        Returns:
            Tensor: Scalar loss value
        """
        batch_size = z_3t.size(0)
        device = z_3t.device
        
        # 1. Normalize representations to compute cosine similarities
        z_3t_norm = F.normalize(z_3t, p=2, dim=-1)
        z_7t_norm = F.normalize(z_7t, p=2, dim=-1)
        
        # Concatenate into a combined batch matrix of size [2N, 768]
        embeddings = torch.cat([z_3t_norm, z_7t_norm], dim=0)
        
        # 2. Compute full pairwise cosine similarity matrix [2N, 2N]
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 3. Define the Explicit Domain Target Masks
        # Create a mask where entry is 1 if both samples belong to the SAME domain (3T-3T or 7T-7T)
        domain_labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)]).to(device)
        domain_labels = domain_labels.unsqueeze(0) # [1, 2N]
        
        # Match mask: shape [2N, 2N], True where domains match
        same_domain_mask = torch.eq(domain_labels.T, domain_labels)
        
        # Filter out self-similarity elements on the main diagonal
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        positive_mask = same_domain_mask & ~self_mask
        negative_mask = ~same_domain_mask

        # 4. Compute the Supervised Contrastive Loss Logits
        # For numerical stability, extract maximum logit per row
        max_logits, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - max_logits.detach()
        
        # Exponential sums for the denominator (all non-self combinations)
        exp_logits = torch.exp(logits) * (~self_mask)
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
        
        # Log-likelihood calculations over all true positive configurations
        log_prob = logits - torch.log(sum_exp_logits + 1e-8)
        
        # Compute mean log-likelihood only over the active positive pairs per row
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        
        # Return negative log-likelihood loss scalar
        loss = -mean_log_prob_pos.mean()
        return loss
#%%

def verify_domain_clustering():
    print("=== Testing Domain Clustering Mechanics ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = DomainClusterContrastiveLoss(temperature=0.07).to(device)
    
    # Mock embeddings for a batch of 3 subjects (N=3)
    # Let's explicitly build structured vectors to confirm the loss behavior
    # Group 1 (3T): Close together
    z_3t = torch.stack([
        torch.tensor([1.0, 0.1, 0.0]),
        torch.tensor([1.0, 0.0, 0.1]),
        torch.tensor([0.9, 0.1, 0.1])
    ]).to(device)
    
    # Group 2 (7T): Close together among themselves, but far from 3T
    z_7t = torch.stack([
        torch.tensor([0.0, 0.1, 1.0]),
        torch.tensor([0.1, 0.0, 1.0]),
        torch.tensor([0.1, 0.1, 0.9])
    ]).to(device)
    
    loss_ideal = loss_fn(z_3t, z_7t)
    print(f"Loss when clusters are cleanly separated: {loss_ideal.item():.4f}")
    
    # Mix them up (bad clustering scenario)
    z_3t_bad = torch.randn(3, 3).to(device)
    z_7t_bad = torch.randn(3, 3).to(device)
    loss_bad = loss_fn(z_3t_bad, z_7t_bad)
    print(f"Loss when representations are randomly mixed: {loss_bad.item():.4f}")
    
    assert loss_bad.item() > loss_ideal.item(), "Clustering orientation math is misaligned."
    print("✅ Success! The loss functions perfectly to penalize domain mixing and reward clean 3T/7T cluster separation.")

if __name__ == "__main__":
    verify_domain_clustering()