import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

# The top-level model you will use
__all__ = ['MTLModel_ET']


# ==============================================================================
# U-NET ENCODER AND HELPER MODULES (WITH MODIFICATIONS)
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


class UNet_Encoder(nn.Module):
    """
    MODIFIED U-Net Encoder.
    It now returns the final node embeddings AND a concatenated vector of
    context summaries from every level of the U-Net.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.node_schedule = [350, 125, 50]
        #self.node_schedule = [5, 50, 25]
        #[500, 250, 125]
        #[1000, 500, 250]
        #[200, 100, 50] 0.4393 1000 done
        #[800, 400, 200] 900 node
        #[200, 150, 50] 800 node done
        #[150, 100, 25] 700 node done
        #[350, 150, 50] 600 node done
        #[350, 125, 50] 500 node done
        #[300, 150, 75] 20.67
        #[250, 125, 50] 20.92
        #[200, 100, 50] 21.03
        #[300, 100, 50] 20.64
        #[300, 150, 100]20.87
        self.unet_layers_per_level = self.model_params.get('unet_layers_per_level', 1)
        
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for num_pooled_nodes in self.node_schedule:
            self.down_convs.append(
                nn.Sequential(*[EncoderLayer(**model_params) for _ in range(self.unet_layers_per_level)])
            )
            self.pools.append(gPool(embedding_dim, num_pooled_nodes))

        self.bottleneck_convs = nn.Sequential(
            *[EncoderLayer(**model_params) for _ in range(self.unet_layers_per_level)]
        )

        self.up_convs = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.up_fusion_layers = nn.ModuleList()
        for _ in self.node_schedule:
            self.unpools.append(gUnpool())
            self.up_fusion_layers.append(nn.Linear(embedding_dim * 2, embedding_dim))
            self.up_convs.append(
                nn.Sequential(*[EncoderLayer(**model_params) for _ in range(self.unet_layers_per_level)])
            )

    def forward(self, depot_xy, node_xy_demand_tw):
        x = torch.cat((self.embedding_depot(depot_xy), self.embedding_node(node_xy_demand_tw)), dim=1)
        
        # <<< NEW >>> List to store context from each level
        level_context_vectors = []

        # --- Downsampling Path ---
        skip_connections = []
        pooling_indices = []
        for i in range(len(self.node_schedule)):
            x = self.down_convs[i](x)
            skip_connections.append(x)
            
            # <<< NEW >>> Capture context summary pre-pooling
            level_context_vectors.append(x.mean(dim=1))
            
            pooled_x, indices = self.pools[i](x)
            pooling_indices.append(indices)
            x = pooled_x
            
        # --- Bottleneck ---
        x = self.bottleneck_convs(x)
        
        # <<< NEW >>> Capture context summary from the bottleneck
        level_context_vectors.append(x.mean(dim=1))

        # --- Upsampling Path ---
        skip_connections.reverse()
        pooling_indices.reverse()
        
        for i in range(len(self.up_convs)):
            x = self.unpools[i](x, skip_connections[i].size(1), pooling_indices[i])
            x = torch.cat([x, skip_connections[i]], dim=2)
            x = F.relu(self.up_fusion_layers[i](x))
            x = self.up_convs[i](x)
        
        # <<< NEW >>> Concatenate all context vectors into one
        multi_level_context = torch.cat(level_context_vectors, dim=1)
            
        return x, multi_level_context

class MTLModel_ET(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        # ... (init as before) ...
        self.encoder = UNet_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)

        # <<< NEW >>> Add global refinement layers
        # The number of layers can be a new hyperparameter, e.g., 'refinement_layer_num'
        refinement_layer_num = model_params.get('refinement_layer_num', 2)
        self.global_refinement_layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(refinement_layer_num)]
        )

        self.encoded_nodes = None
        self.multi_level_context = None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        # ... (get node_xy_demand_tw as before) ...
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        
        node_xy_demand_tw = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
        
        # 1. Get multi-scale features and context from the U-Net
        unet_output_nodes, self.multi_level_context = self.encoder(depot_xy, node_xy_demand_tw)
        
        # 2. <<< NEW >>> Perform final global refinement
        refined_nodes = unet_output_nodes
        for layer in self.global_refinement_layers:
            refined_nodes = layer(refined_nodes)
            
        # 3. Use these refined nodes for the decoder
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
            
            # <<< MODIFIED >>> Pass the new context vector to the decoder
            probs = self.decoder(encoded_last_node, attr, self.multi_level_context, ninf_mask=state.ninf_mask)

            if selected is None:
                if self.training or self.eval_type == 'softmax':
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                else:
                    selected = probs.argmax(dim=2)
            
            prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob


class MTL_Decoder(nn.Module):
    """
    MODIFIED Decoder. It now accepts the multi-level context to generate its query.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # <<< NEW >>> Calculate the size of the concatenated context vector
        # Number of downsampling levels + 1 for the initial level + 1 for the bottleneck
        num_context_levels = 3 + 1 
        context_dim = embedding_dim * num_context_levels

        # <<< MODIFIED >>> The query projection layer now accepts the extra context
        self.Wq_last = nn.Linear(embedding_dim + 4 + context_dim, head_num * qkv_dim, bias=False)
        
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k, self.v, self.single_head_key = None, None, None

        #self.proj1 = nn.Linear(16, 4)
        #self.proj2 = nn.Linear(16, 4)
        #self.proj3 = nn.Linear(16, 4)
        #self.proj4 = nn.Linear(64, 128)


    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, attr, multi_level_context, ninf_mask):
        batch_size, pomo_size, embedding_dim = encoded_last_node.shape
        head_num = self.model_params['head_num']
        
        # <<< NEW >>> Expand the context vector to match the pomo dimension
        context_expanded = multi_level_context.unsqueeze(1).expand(-1, pomo_size, -1)
        #print(context_expanded.shape)
        # <<< MODIFIED >>> Concatenate the context vector into the query input
        input_cat = torch.cat((encoded_last_node, attr, context_expanded), dim=2)
        
        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        
        #proj_1 = nn.Parameter(torch.randn(embedding_dim // 8, embedding_dim //16) / (embedding_dim//16 ** 0.5), requires_grad=False)
        #proj_2 = nn.Parameter(torch.randn(embedding_dim // 8, embedding_dim //16) / (embedding_dim//16 ** 0.5), requires_grad=False)
        #proj_3 = nn.Parameter(torch.randn(embedding_dim // 8, embedding_dim //16) / (embedding_dim//16 ** 0.5), requires_grad=False)
        #proj_4 = nn.Parameter(torch.randn(embedding_dim // 2, embedding_dim) / (embedding_dim ** 0.5), requires_grad=False)
        #print(q.shape)
        #k1, k2, k3 = q@proj_1, self.k@proj_2, self.v@proj_3
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        #print(out_concat.shape)
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, self.single_head_key)
        
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        
        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        return probs


# ==============================================================================
# UNMODIFIED HELPER CLASSES AND FUNCTIONS
# ==============================================================================

def _get_encoding(encoded_nodes, node_index_to_pick):
    # ... (unchanged)
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)

class EncoderLayer(nn.Module):
    # ... (unchanged)
    def __init__(self, **model_params):
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
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)
        return out3


def reshape_by_heads(qkv, head_num):
    # ... (unchanged)
    batch_s, n, _ = qkv.size()
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)


def multi_head_attention(q, k, v, rank3_ninf_mask=None):
    # ... (unchanged)
    batch_s, head_num, n, key_dim = q.size()
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3)) / (key_dim ** 0.5)
    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = nn.Softmax(dim=3)(score)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(batch_s, n, -1)


class Add_And_Normalization_Module(nn.Module):
    # ... (unchanged)
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True)
        else: # LayerNorm
            self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input1, input2):
        added = input1 + input2
        if isinstance(self.norm, nn.BatchNorm1d):
            return self.norm(added.reshape(-1, added.size(-1))).reshape(added.size())
        else:
            return self.norm(added)


class FeedForward(nn.Module):
    # ... (unchanged)
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, input1):
        return self.W2(self.relu(self.W1(input1)))