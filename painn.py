"""
Inspired by: https://github.com/Yangxinsix/painn-sli/blob/main/PaiNN/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import global_add_pool


class PaiNN(nn.Module):
    def __init__(
            self,
            num_rounds=3, 
            state_dim=128,
            cutoff=5,
            edge_dim=20
    ):
        super().__init__()

        self.num_rounds = num_rounds
        self.state_dim = state_dim
        self.cutoff = cutoff
        self.edge_dim = edge_dim # n
        self.num_elements = 119

        self.node_embedding = torch.nn.Linear(self.num_elements, state_dim) # ont-hot nuclear charge -> state_dim
        # self.atom_embedding = nn.Embedding(self.num_elements, self.state_dim)

        self.message_layers = nn.ModuleList([
            PaiNNMessage(self.state_dim, self.edge_dim, self.cutoff)
            for _ in range(self.num_rounds)
        ])

        self.update_layers = nn.ModuleList([
            PaiNNUpdate(self.state_dim)
            for _ in range(self.num_rounds)
        ])

        self.graph_readout = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, 1)
        )

    def forward(self, data):
        num_nodes = data.z.size(0)
        edge = data.edge_index
        r_ij, norm_r_ij = self.get_edge_vectors(data)

        # Embed and initialise
        embedding_input = F.one_hot(data.z, self.num_elements).float()
        state = self.node_embedding(embedding_input)
        # state = self.atom_embedding(data.z)
        state_vec = torch.zeros([num_nodes, self.state_dim, 3], device=data.pos.device, dtype=data.pos.dtype)

        # Message passing loop
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            state, state_vec = message_layer(state, state_vec, edge, r_ij, norm_r_ij)
            state, state_vec = update_layer(state, state_vec)

        # Readout
        state = self.graph_readout(state)
        energy = global_add_pool(state, data.batch) # lke index_add

        return energy # we'll replace later with diffusion stuff


    def get_edge_vectors(self, data):
        row, col = data.edge_index
        r_ij = data.pos[col] - data.pos[row]
        norm_r_ij = r_ij.norm(dim=1, keepdim=True)
        return r_ij, norm_r_ij   



class PaiNNMessage(nn.Module):
    def __init__(self, state_dim, edge_dim, cutoff):
        super().__init__()
        self.state_dim = state_dim
        self.edge_dim = edge_dim
        self.cutoff = cutoff

        self.phi = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim * 3) # tripple the outputs fro splitting
        )

        self.W = nn.Linear(edge_dim, state_dim*3)


    def forward(self, state, state_vec, edge, r_ij, norm_r_ij):
        # W pass
        RBF = self.RBF(norm_r_ij)
        W = self.W(RBF)
        W = W * self.cosine_cutoff(norm_r_ij)[:, :, None]

        # phi pass
        phi = self.phi(state)

        # Combination with hadamard
        combination = phi[edge[1, :]] * W.squeeze(1)

        # Splitting into the 3 parts
        gate_state_vec, gate_edge_vec, scalar_message = torch.split(
            combination,
            self.state_dim,
            dim=1
        )

        #  Vector message part
        vector_message = state_vec[edge[1, :]] * gate_state_vec[:, :, None]
        normalised_r_ij = r_ij / norm_r_ij
        edge_vec = normalised_r_ij[:, None, :] * gate_edge_vec[:, :, None]
        vector_message += edge_vec # edge_vec is edge and vector interaction in the diagram

        # Sum of incoming messages
        delta_si = torch.zeros_like(state)
        delta_vi = torch.zeros_like(state_vec)
        delta_si = torch.index_add(delta_si, 0, edge[0, :], scalar_message)
        delta_vi = torch.index_add(delta_vi, 0, edge[0, :], vector_message)

        state = state + delta_si
        state_vec = state_vec + delta_vi

        return state, state_vec


    def RBF(self, norm_r_ij):
        n = torch.arange(self.edge_dim, device=norm_r_ij.device) + 1
        return torch.sin(norm_r_ij.unsqueeze(-1) * n * torch.pi / self.cutoff) / norm_r_ij.unsqueeze(-1)
    

    def cosine_cutoff(self, norm_r_ij):
        return torch.where(
            norm_r_ij < self.cutoff,
            0.5 * (torch.cos(torch.pi * norm_r_ij / self.cutoff) + 1),
            torch.tensor(0.0, device=norm_r_ij.device, dtype=norm_r_ij.dtype),            
        )


class PaiNNUpdate(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim

        # TODO: Compare linear and parameter implementations
        self.U = nn.Parameter(torch.randn(self.state_dim, self.state_dim))
        self.V = nn.Parameter(torch.randn(self.state_dim, self.state_dim))

        self.a = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim*3) # tripple it here
        )
        

    def forward(self, state, state_vec):
        
        # U-dot and V-dot
        # TODO: Sanity check this
        udot = self.U[None, :, :] @ state_vec
        vdot = self.V[None, :, :] @ state_vec

        # Norm passing to sj stack
        vdot_norm = torch.norm(vdot, dim=2)
        stack = torch.cat([state, vdot_norm], dim=1)

        # sj pass
        split = self.a(stack)

        # Splitting into three groups
        a_vv, a_sv, a_ss = torch.split(
            split,
            self.state_dim,
            dim=1
        )

        # Delta vi line
        delta_vi = udot * a_vv[:, :, None]

        # Delta si line
        dot = torch.sum(udot*vdot, dim=2)
        dot = dot * a_sv
        delta_si = dot + a_ss

        state = state + delta_si
        state_vec = state_vec + delta_vi

        return state, state_vec
