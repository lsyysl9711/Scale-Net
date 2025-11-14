import torch
import torch.nn as nn
import torch.nn.functional as F
from .MOELayer import MoE

# The public name is kept as MOEModel, but it now uses the U-Net architecture internally.
__all__ = ['MOEModel']

# ==============================================================================
# NEW U-NET HELPER MODULES (Internal to this file)
# ==============================================================================

class gPool(nn.Module):
    def __init__(self, in_channels, target_num_nodes):
        super().__init__()
        self.target_num_nodes = target_num_nodes
        self.proj = nn.Linear(in_channels, 1)

    def forward(self, x):
        scores = self.proj(x).squeeze(-1)
        k = min(self.target_num_nodes, x.size(1))
        _, top_k_indices = torch.topk(scores, k, dim=1)
        top_k_indices, _ = torch.sort(top_k_indices, dim=1)
        pooled_x = x.gather(1, top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        return pooled_x, top_k_indices


class gUnpool(nn.Module):
    def forward(self, coarse_features, original_size, indices):
        batch_size, _, embed_dim = coarse_features.shape
        output = torch.zeros(batch_size, original_size, embed_dim, device=coarse_features.device)
        output.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, embed_dim), coarse_features)
        return output

# ==============================================================================
# CORE MODULES MODIFIED TO IMPLEMENT THE U-NET ARCHITECTURE
# ==============================================================================

class MOEModel(nn.Module):
    """
    Main model class. Its name is kept as MOEModel for consistency.
    It now uses a U-Net Encoder and Global Refinement layers internally.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']
        self.aux_loss = 0

        # The name is MTL_Encoder, but its implementation is now a U-Net.
        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        
        # Add the global refinement layers, as designed in MTLModel_UNet.py
        refinement_layer_num = model_params.get('refinement_layer_num', 1)
        self.global_refinement_layers = nn.ModuleList(
            [EncoderLayer(depth=99, **model_params) for _ in range(refinement_layer_num)] # Use a unique depth
        )

        self.encoded_nodes = None
        self.multi_level_context = None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        
        node_xy_demand_tw = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
        
        # 1. The encoder (now a U-Net) returns 3 outputs
        unet_output, context, encoder_moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
        self.aux_loss = encoder_moe_loss
        self.multi_level_context = context

        # 2. Perform global refinement on the U-Net's output
        refined_nodes = unet_output
        for layer in self.global_refinement_layers:
            refined_nodes, refinement_loss = layer(refined_nodes)
            self.aux_loss += refinement_loss
        
        # 3. Set the final, refined nodes for the decoder
        self.encoded_nodes = refined_nodes
        self.decoder.set_kv(self.encoded_nodes)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count < 2:
            if state.selected_count == 0:
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            else: # state.selected_count == 1
                selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None], state.open[:, :, None]), dim=2)
            
            # Pass the multi-level context to the decoder
            probs, decoder_moe_loss = self.decoder(encoded_last_node, attr, self.multi_level_context, ninf_mask=state.ninf_mask)
            self.aux_loss += decoder_moe_loss

            if selected is None:
                if self.training or self.eval_type == 'softmax':
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(1).reshape(batch_size, pomo_size)
                else:
                    selected = probs.argmax(dim=2)
            prob = probs.gather(2, selected.unsqueeze(2)).squeeze(2)

        return selected, prob


class MTL_Encoder(nn.Module):
    """
    This class is now a U-Net Encoder internally, but keeps the name 'MTL_Encoder' for API consistency.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.node_schedule = self.model_params.get('node_schedule', [75, 50, 25])
        self.unet_layers_per_level = self.model_params.get('unet_layers_per_level', 1)

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i, num_pooled_nodes in enumerate(self.node_schedule):
            self.down_convs.append(
                nn.ModuleList([EncoderLayer(depth=i, **model_params) for _ in range(self.unet_layers_per_level)])
            )
            self.pools.append(gPool(embedding_dim, num_pooled_nodes))

        self.bottleneck_convs = nn.ModuleList(
            [EncoderLayer(depth=len(self.node_schedule), **model_params) for _ in range(self.unet_layers_per_level)]
        )

        self.up_convs = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.up_fusion_layers = nn.ModuleList()
        for i, _ in enumerate(self.node_schedule):
            self.unpools.append(gUnpool())
            self.up_fusion_layers.append(nn.Linear(embedding_dim * 2, embedding_dim))
            self.up_convs.append(
                nn.ModuleList([EncoderLayer(depth=len(self.node_schedule)+i+1, **model_params) for _ in range(self.unet_layers_per_level)])
            )

    def forward(self, depot_xy, node_xy_demand_tw):
        x = torch.cat((self.embedding_depot(depot_xy), self.embedding_node(node_xy_demand_tw)), dim=1)
        
        total_moe_loss = 0
        level_context_vectors = []
        skip_connections = []
        pooling_indices = []

        # Downsampling Path
        for i in range(len(self.node_schedule)):
            for layer in self.down_convs[i]:
                x, loss = layer(x)
                total_moe_loss += loss
            skip_connections.append(x)
            level_context_vectors.append(x.mean(dim=1))
            x, indices = self.pools[i](x)
            pooling_indices.append(indices)
        
        # Bottleneck
        for layer in self.bottleneck_convs:
            x, loss = layer(x)
            total_moe_loss += loss
        level_context_vectors.append(x.mean(dim=1))

        # Upsampling Path
        skip_connections.reverse()
        pooling_indices.reverse()
        
        for i in range(len(self.up_convs)):
            x = self.unpools[i](x, skip_connections[i].size(1), pooling_indices[i])
            x = torch.cat([x, skip_connections[i]], dim=2)
            x = F.relu(self.up_fusion_layers[i](x))
            for layer in self.up_convs[i]:
                x, loss = layer(x)
                total_moe_loss += loss

        multi_level_context = torch.cat(level_context_vectors, dim=1)
        
        # Returns 3 values now
        return x, multi_level_context, total_moe_loss


class MTL_Decoder(nn.Module):
    """
    Keeps the name 'MTL_Decoder' but is modified to accept multi-level context.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Calculate context_dim based on the U-Net schedule
        node_schedule = self.model_params.get('node_schedule', [75, 50, 25])
        num_context_levels = len(node_schedule) + 1
        context_dim = embedding_dim * num_context_levels

        # Wq_last now includes the context dimension
        self.Wq_last = nn.Linear(embedding_dim + 4 + context_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        if self.model_params['num_experts'] > 1 and 'Dec' in self.model_params['expert_loc']:
            self.multi_head_combine = MoE(input_size= head_num * qkv_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                          k=self.model_params['topk'], noisy_gating=True, routing_level=self.model_params['routing_level'],
                                          moe_model="Linear")
        else:
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k, self.v, self.single_head_key = None, None, None
        #self.moa = MixtureOfAttention(False, 1, **model_params)
        #self.moa_score = MixtureOfAttention(True, **model_params)

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    # Forward signature is now changed to accept the context
    def forward(self, encoded_last_node, attr, multi_level_context, ninf_mask):
        moe_loss = 0
        batch_size, pomo_size, _ = encoded_last_node.shape
        head_num = self.model_params['head_num']
        
        context_expanded = multi_level_context.unsqueeze(1).expand(-1, pomo_size, -1)
        input_cat = torch.cat((encoded_last_node, attr, context_expanded), dim=2)

        o = self.Wq_last(input_cat)
        q_last = reshape_by_heads(o, head_num=head_num)
        q = q_last

        #q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        #out_concat, loss = self.moa(q, self.k, self.v, o, ninf_mask)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        
        if isinstance(self.multi_head_combine, MoE):
            mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
        else:
            mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key)
        #score, scoring_moa_loss = self.scoring_moa(q_score, self.k_score, None, gating_input=mh_atten_out, rank3_ninf_mask=ninf_mask)
        score_scaled = score / self.model_params['sqrt_embedding_dim']
        score_clipped = self.model_params['logit_clipping'] * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        
        return probs, moe_loss
    #+ loss


# ==============================================================================
# UNMODIFIED HELPER AND SUB-CLASSES
# ==============================================================================

def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size, pomo_size = node_index_to_pick.shape
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick.unsqueeze(2).expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)

class EncoderLayer(nn.Module):
    def __init__(self, depth=0, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        
        if self.model_params['num_experts'] > 1 and "Enc{}".format(depth) in self.model_params['expert_loc']:
            self.feedForward = MoE(input_size=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                   hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], noisy_gating=True,
                                   routing_level=self.model_params['routing_level'], moe_model="MLP")
        else:
            self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
        #self.moa = MixtureOfAttention(False, 0, **model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        #out_concat, loss = self.moa(q,k,v,input1)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2, moe_loss = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)
        return out3, moe_loss

def reshape_by_heads(qkv, head_num):
    batch_s, n, _ = qkv.size()
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)

def multi_head_attention(q, k, v, rank3_ninf_mask=None):
    batch_s, head_num, n, key_dim = q.size()
    score = torch.matmul(q, k.transpose(2, 3)) / (key_dim ** 0.5)
    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, k.size(2))
    weights = nn.Softmax(dim=3)(score)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(batch_s, n, -1)

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        #self.norm = nn.InstanceNorm1d(model_params['embedding_dim'], affine=True, track_running_stats=False)
        self.norm = nn.LayerNorm(model_params['embedding_dim'])

    def forward(self, input1, input2):
        #if isinstance(self.norm, nn.InstanceNorm1d):
        added = input1 + input2 
        #     transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
        #     normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
        #     back_trans = normalized.transpose(1, 2)
        
        return self.norm(added)

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.W1 = nn.Linear(model_params['embedding_dim'], model_params['ff_hidden_dim'])
        self.W2 = nn.Linear(model_params['ff_hidden_dim'], model_params['embedding_dim'])
        self.relu = nn.ReLU()

    def forward(self, input1):
        return self.W2(self.relu(self.W1(input1))), 0

class MixtureOfAttention(nn.Module):
    """ Implements a Mixture of Attention Heads using 'Select First, Compute Later' logic for OOM safety."""

    def __init__(self, is_scoring_head, flag, **model_params):
        super().__init__()
        self.is_scoring_head = is_scoring_head
        self.qkv_dim = model_params['qkv_dim']
        #if flag == 0:
        #   gating_input_dim = model_params.get('gating_input_dim', 128)
        #else:
        gating_input_dim = model_params.get('gating_input_dim', 128)
        #self.transform = nn.Linear(64, 128)
        if self.is_scoring_head:
            self.total_head_num = model_params.get('scoring_head_num', 4)
            self.top_k = model_params.get('scoring_moa_top_k', 2)
            self.score_aggregator = nn.Linear(self.top_k, 1, bias=False)
        else:
            self.total_head_num = model_params['head_num']
            self.top_k = model_params.get('context_moa_top_k', 2)
        self.gate = nn.Linear(gating_input_dim, self.total_head_num)

    def forward(self, q, k, v, gating_input, rank3_ninf_mask=None):
        gating_input_mean = gating_input.mean(dim=1)
        #if gating_input_mean.shape[-1] == 64:
        #    gating_input_mean = self.transform(gating_input_mean)
            
        gate_logits = self.gate(gating_input_mean)
        gate_softmax = F.softmax(gate_logits, dim=1)
        _, top_k_indices = torch.topk(gate_softmax, self.top_k, dim=1, sorted=False)
        gating_mask = torch.zeros_like(gate_softmax).scatter_(1, top_k_indices, 1)
        load_balancing_loss = self.calculate_load_balancing_loss(gate_softmax, gating_mask)
        q_selected, k_selected, v_selected = self.gather_qkv(q, k, v, top_k_indices)
        attention_output = self._perform_attention(q_selected, k_selected, v_selected, rank3_ninf_mask)
        if self.is_scoring_head:
            scores_permuted = attention_output.permute(0, 2, 3, 1)
            final_scores = self.score_aggregator(scores_permuted).squeeze(-1)
            return (final_scores, load_balancing_loss)
        return (attention_output, load_balancing_loss)

    def calculate_load_balancing_loss(self, gate_softmax, gating_mask):
        tokens_per_expert = gating_mask.sum(dim=0)
        router_prob_per_expert = torch.mean(gate_softmax, dim=0)
        density_per_expert = tokens_per_expert / (tokens_per_expert.sum() + 1e-06)
        return self.total_head_num * torch.sum(router_prob_per_expert * density_per_expert)

    def gather_qkv(self, q, k, v, top_k_indices):
        batch_size = q.size(0)
        indices_for_q = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.top_k, q.size(2), self.qkv_dim)
        q_selected = torch.gather(q, 1, indices_for_q)
        indices_for_kv = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.top_k, k.size(2), self.qkv_dim)
        k_selected = torch.gather(k, 1, indices_for_kv)
        v_selected = torch.gather(v, 1, indices_for_kv) if v is not None else None
        return (q_selected, k_selected, v_selected)

    def _perform_attention(self, q, k, v, rank3_ninf_mask=None):
        batch_s, head_num, n, key_dim = q.size()
        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=q.device))
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(-1, head_num, -1, -1)
        if v is None:
            return score_scaled
        weights = nn.Softmax(dim=3)(score_scaled)
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(batch_s, n, head_num * key_dim)