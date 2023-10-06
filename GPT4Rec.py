import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from moduleslinear.feature_map import elu_feature_map
from fast_transformers.masking import TriangularCausalMask
import numpy as np
class GPTReclinearModel(nn.Module):
    def __init__(self, args):
        super(GPTReclinearModel, self).__init__()
        self.args = args
        self.decoder = TransformerDecoder(args.user_size,args.item_size,args.max_seq_length,args.hidden_size,
                                        args.num_hidden_layers,args.num_attention_heads,args.dim_feed_forward,
                                        args.hidden_dropout_prob,args.attention_probs_dropout_prob,
                                        args.residual_prob,args.pad_id)

        self.gpt_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer = nn.Linear(args.hidden_size, args.item_size)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.normal_(self.gpt_norm.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)

    def forward(self, user_id, inputs, segments):
        outputs = self.decoder(user_id, inputs, segments)
        predict_logits = self.output_layer(outputs)

        if self.args.pretrainstage == "No":
            recommend_output = outputs
            recommend_output = recommend_output[:, -1, :]
            test_item_emb = self.decoder.item_embeddings.weight
            rating_pred = torch.matmul(recommend_output, test_item_emb.transpose(0, 1))
            rating_pred = rating_pred.cpu().data.numpy().copy()
            batch_user_index = user_id.cpu().numpy()
            rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
            ind = np.argpartition(rating_pred, -1)[:, -1:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            index = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
            index = torch.tensor(index,device=outputs.device)
            return outputs, predict_logits, index
        else:
            return outputs, predict_logits,

    def pretrain(self, user_id, input_ids, pos_ids, neg_ids):
        # [B L H]
        # GPT sequence recommendation with softmax
        """
        segments = torch.zeros(input_ids.size(), device=input_ids.device, dtype=torch.long)
        _, predict_logits = self.forward(user_id, input_ids, segments)
        loss = F.cross_entropy(predict_logits.view(-1, predict_logits.size(-1)), pos_ids.view(-1), ignore_index=0)
        """
        segments = torch.zeros(input_ids.size(), device=input_ids.device, dtype=torch.long)
        seq_out, _ = self.forward(user_id, input_ids, segments)
        pos_emb = self.decoder.item_embeddings(pos_ids)
        neg_emb = self.decoder.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget) / torch.sum(istarget)
        return loss

    @torch.no_grad()
    def GPT_rec_next_item_train_all(self, user_id, input_id, target_pos, target_neg, top_k=1, temperature=1.0):
        rec_new_item = self.args.next_window_train
        origin_target_pos = target_pos
        origin_target_neg = target_neg
        segment = torch.zeros(input_id.size(), device=input_id.device, dtype=torch.long)
        origin_target_segment = segment
        n = input_id.size()[1]
        for i in range(n - 1):
            if i == 0:
                zero = torch.zeros(input_id.size()[0], n - i - 1, device=input_id.device, dtype=torch.long)

                input_id_i = input_id[:, i]
                input_id_i = torch.unsqueeze(input_id_i, 1)
                input_id = torch.cat((zero, input_id_i), dim=1)

                target_pos = origin_target_pos[:, i]
                target_pos = torch.unsqueeze(target_pos, 1)
                target_pos = torch.cat((zero, target_pos), dim=1)

                target_neg = origin_target_neg[:, i]
                target_neg = torch.unsqueeze(target_neg, 1)
                target_neg = torch.cat((zero, target_neg), dim=1)

                segment = origin_target_segment[:, i]
                segment = torch.unsqueeze(segment, 1)
                segment = torch.cat((zero, segment), dim=1)

            else:
                input_id = input_id
                target_pos = target_pos
                target_neg = target_neg
                segment =segment

            for j in range(rec_new_item):
                ones_seg = torch.ones(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                zero_pos = torch.zeros(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)

                if self.args.Finetune_generate_idx_next_softmax_logits == "Yes":
                    _, predict_logits, _ = self.forward(user_id, input_id, segment)
                    predict_logits = predict_logits[:, -1, :] / temperature
                    probs = F.softmax(predict_logits, dim=-1)
                    _, idx_next = torch.topk(probs, k=top_k, dim=-1)
                else:
                    _, _, idx_next = self.forward(user_id, input_id, segment)

                new_id = torch.cat((input_id[:, 1:], idx_next), dim=1)
                input_id = new_id
                target_pos = torch.cat((target_pos[:, 1:], zero_pos), dim=1)
                target_neg = torch.cat((target_neg[:, 1:], zero_pos), dim=1)
                segment = torch.cat((segment[:, 1:], ones_seg), dim=1)

            input_id = torch.cat((input_id[:, 1:], torch.unsqueeze(origin_target_pos[:, i], 1)), dim=1)
            target_pos = torch.cat((target_pos[:, 1:], torch.unsqueeze(origin_target_pos[:, i + 1], 1)), dim=1)
            target_neg = torch.cat((target_neg[:, 1:], torch.unsqueeze(origin_target_neg[:, i + 1], 1)), dim=1)
            segment = torch.cat((segment[:, 1:], torch.unsqueeze(origin_target_segment[:, i+1], 1)), dim=1)
        return input_id, target_pos, target_neg, segment

    @torch.no_grad()
    def GPT_rec_next_item_train_n_1(self, user_id, input_id, target_pos, target_neg, top_k=1, temperature=1.0):
        rec_new_item = self.args.next_window_train
        segment = torch.zeros(input_id.size(), device=input_id.device, dtype=torch.long)
        origin_input_id = input_id
        origin_target_pos = target_pos
        origin_target_neg = target_neg
        origin_target_segment = segment
        n = input_id.size()[1]
        GPT_generate = n-rec_new_item
        for i in range(n):
            if i >= GPT_generate:
                input_id = input_id
                target_pos = target_pos
                target_neg = target_neg
                segment = segment
                ones_seg = torch.ones(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                zero_pos = torch.zeros(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                if self.args.Finetune_generate_idx_next_softmax_logits == "Yes":
                    _, predict_logits, _ = self.forward(user_id, input_id, segment)
                    predict_logits = predict_logits[:, -1, :] / temperature
                    probs = F.softmax(predict_logits, dim=-1)
                    _, idx_next = torch.topk(probs, k=top_k, dim=-1)
                else:
                    _, _, idx_next = self.forward(user_id, input_id, segment)
                new_id = torch.cat((input_id[:, 1:], idx_next), dim=1)
                input_id = new_id
                segment = torch.cat((segment[:, 1:], ones_seg), dim=1)
                target_pos = torch.cat((target_pos[:, 1:], torch.unsqueeze(origin_target_pos[:, i], 1)), dim=1)
                target_neg = torch.cat((target_neg[:, 1:], torch.unsqueeze(origin_target_neg[:, i], 1)), dim=1)
            else:
                if i == GPT_generate-1:
                    input_id = torch.cat((input_id[:, 1:], torch.unsqueeze(origin_input_id[:, i], 1)), dim=1)
                    target_pos = torch.cat((target_pos[:, 1:], torch.unsqueeze(origin_target_pos[:, i], 1)), dim=1)
                    target_neg = torch.cat((target_neg[:, 1:], torch.unsqueeze(origin_target_neg[:, i], 1)), dim=1)
                    segment = torch.cat((segment[:, 1:], torch.unsqueeze(origin_target_segment[:, i], 1)), dim=1)
                else:
                    zero_pos = torch.zeros(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                    input_id = torch.cat((input_id[:, 1:], torch.unsqueeze(origin_input_id[:, i], 1)), dim=1)
                    target_pos = torch.cat((target_pos[:, 1:], torch.unsqueeze(origin_target_pos[:, i], 1)), dim=1)
                    target_neg = torch.cat((target_neg[:, 1:],  torch.unsqueeze(origin_target_neg[:, i], 1)), dim=1)
                    segment = torch.cat((segment[:, 1:], torch.unsqueeze(origin_target_segment[:, i], 1)), dim=1)
        return input_id, target_pos, target_neg, segment

    @torch.no_grad()
    def GPT_rec_next_item_test_all(self, user_id, input_id, target_pos, target_neg, top_k=1, temperature=1.0):
        rec_new_item = self.args.next_window_test
        segment = torch.zeros(input_id.size(), device=input_id.device, dtype=torch.long)
        origin_input_id = input_id
        origin_target_pos = target_pos
        origin_target_neg = target_neg
        origin_target_segment = segment
        n = input_id.size()[1]

        for i in range(n):
            if i == 0:
                zero = torch.zeros(input_id.size()[0], n - i - 1, device=input_id.device, dtype=torch.long)
                input_id_i = input_id[:, i]
                input_id_i = torch.unsqueeze(input_id_i, 1)
                input_id = torch.cat((zero, input_id_i), dim=1)

                target_pos = origin_target_pos[:, i]
                target_pos = torch.unsqueeze(target_pos, 1)
                target_pos = torch.cat((zero, target_pos), dim=1)

                target_neg = origin_target_neg[:, i]
                target_neg = torch.unsqueeze(target_neg, 1)
                target_neg = torch.cat((zero, target_neg), dim=1)

                segment = origin_target_segment[:, i]
                segment = torch.unsqueeze(segment, 1)
                segment = torch.cat((zero, segment), dim=1)
            else:
                input_id = input_id
                target_pos = target_pos
                target_neg = target_neg
                segment = segment
            for j in range(rec_new_item):
                ones_seg = torch.ones(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                zero_pos = torch.zeros(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                if self.args.Finetune_generate_idx_next_softmax_logits == "Yes":
                    _, predict_logits, _ = self.forward(user_id, input_id, segment)
                    predict_logits = predict_logits[:, -1, :] / temperature
                    probs = F.softmax(predict_logits, dim=-1)
                    _, idx_next = torch.topk(probs, k=top_k, dim=-1)
                else:
                    _, _, idx_next = self.forward(user_id, input_id, segment)
                new_id = torch.cat((input_id[:, 1:], idx_next), dim=1)
                input_id = new_id
                target_pos = torch.cat((target_pos[:, 1:], zero_pos), dim=1)
                target_neg = torch.cat((target_neg[:, 1:], zero_pos), dim=1)
                segment = torch.cat((segment[:, 1:], ones_seg), dim=1)

            input_id = torch.cat((input_id[:, 1:], torch.unsqueeze(origin_input_id[:, i], 1)), dim=1)
            target_pos = torch.cat((target_pos[:, 1:], torch.unsqueeze(origin_target_pos[:, i], 1)), dim=1)
            target_neg = torch.cat((target_neg[:, 1:], torch.unsqueeze(origin_target_neg[:, i], 1)), dim=1)
            segment = torch.cat((segment[:, 1:], torch.unsqueeze(origin_target_segment[:, i], 1)), dim=1)
        return input_id, target_pos, target_neg, segment
    @torch.no_grad()
    def GPT_rec_next_item_test_n_1(self, user_id, input_id, target_pos, target_neg, top_k=1, temperature=1.0):
        rec_new_item = self.args.next_window_test
        segment = torch.zeros(input_id.size(), device=input_id.device, dtype=torch.long)
        origin_input_id = input_id
        origin_target_pos = target_pos
        origin_target_neg = target_neg
        origin_target_segment = segment
        n = input_id.size()[1]
        for i in range(n):
            if i == n - 1:
                input_id = input_id
                target_pos = target_pos
                target_neg = target_neg
                segment = segment
                for j in range(rec_new_item):
                    ones_seg = torch.ones(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                    zero_pos = torch.zeros(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
                    if self.args.Finetune_generate_idx_next_softmax_logits == "Yes":
                        _, predict_logits, _ = self.forward(user_id, input_id, segment)
                        predict_logits = predict_logits[:, -1, :] / temperature
                        probs = F.softmax(predict_logits, dim=-1)
                        _, idx_next = torch.topk(probs, k=top_k, dim=-1)
                    else:
                        _, _, idx_next = self.forward(user_id, input_id, segment)
                    new_id = torch.cat((input_id[:, 1:], idx_next), dim=1)
                    input_id = new_id
                    target_pos = torch.cat((target_pos[:, 1:], zero_pos), dim=1)
                    target_neg = torch.cat((target_neg[:, 1:], zero_pos), dim=1)
                    segment = torch.cat((segment[:, 1:], ones_seg), dim=1)

                input_id = torch.cat((input_id[:, 1:], torch.unsqueeze(origin_input_id[:, i], 1)), dim=1)
                target_pos = torch.cat((target_pos[:, 1:], torch.unsqueeze(origin_target_pos[:, i], 1)), dim=1)
                target_neg = torch.cat((target_neg[:, 1:], torch.unsqueeze(origin_target_neg[:, i], 1)), dim=1)
                segment = torch.cat((segment[:, 1:], torch.unsqueeze(origin_target_segment[:, i], 1)), dim=1)
            else:
                input_id = torch.cat((input_id[:, 1:], torch.unsqueeze(origin_input_id[:, i], 1)), dim=1)
                target_pos = torch.cat((target_pos[:, 1:], torch.unsqueeze(origin_target_pos[:, i], 1)), dim=1)
                target_neg = torch.cat((target_neg[:, 1:], torch.unsqueeze(origin_target_neg[:, i], 1)), dim=1)
                segment = torch.cat((segment[:, 1:], torch.unsqueeze(origin_target_segment[:, i], 1)), dim=1)
        return input_id, target_pos, target_neg, segment
    @torch.no_grad()
    def infer_GPT_recall_20(self, user_id, input_id, top_k=1, temperature=1.0):
        infer_GPT_generate = self.args.infer_GPT_recall_number_20
        segment = torch.zeros(input_id.size(), device=input_id.device, dtype=torch.long)
        for j in range(infer_GPT_generate): # infer_GPT_generate
            ones_seg = torch.ones(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
            segment = torch.cat((segment[:, 1:], ones_seg), dim=1)
            if self.args.Finetune_generate_idx_next_softmax_logits == "Yes":
                _, predict_logits, _ = self.forward(user_id, input_id, segment)
                predict_logits = predict_logits[:, -1, :] / temperature
                probs = F.softmax(predict_logits, dim=-1)
                _, idx_next = torch.topk(probs, k=top_k, dim=-1)
            else:
                _, _, idx_next = self.forward(user_id, input_id, segment)
            new_id = torch.cat((input_id[:, 1:], idx_next), dim=1)
            input_id = new_id

        recall_list = input_id[:, -infer_GPT_generate:]

        return recall_list.tolist()

    # Prompts Fine-tune
    def finetune_train(self, user_id,input_ids, target_pos, target_neg, GPT):
        if GPT is True:
            if self.args.Finetune_train_Prompts_n_1 == "Yes":
                generate_list, target_pos_i, target_neg_i, segment = self.GPT_rec_next_item_train_n_1(user_id,input_ids,target_pos,target_neg)
                sequence_output, predict_logits, _ = self.forward(user_id, generate_list, segment)
            else:
                generate_list, target_pos_i, target_neg_i, segment = self.GPT_rec_next_item_train_all(user_id, input_ids, target_pos,target_neg)
                sequence_output, predict_logits, _ = self.forward(user_id, generate_list, segment)
        else:
            segment = torch.zeros(input_ids.size(), device=input_ids.device, dtype=torch.long)
            sequence_output, predict_logits, _ = self.forward(user_id,input_ids, segment)
            target_pos_i = target_pos
            target_neg_i = target_neg
        return sequence_output,predict_logits,target_pos_i,target_neg_i

    def finetune_test(self, user_id, input_ids, target_pos, target_neg, GPT):
        if GPT is True:
            # if self.args.Finetune_test_history_Prompts_n_1 == "Yes":
            if self.args.Finetune_infer_Norecall_history_Prompts_n_1_or_all == "Yes":
                generate_list, target_pos_i, target_neg_i,segment = self.GPT_rec_next_item_test_n_1(user_id, input_ids, target_pos, target_neg)
            else:
                generate_list, target_pos_i, target_neg_i,segment = self.GPT_rec_next_item_test_all(user_id, input_ids, target_pos,target_neg)
            sequence_output, predict_logits,_ = self.forward(user_id, generate_list,segment)
        else:
            segment = torch.zeros(input_ids.size(), device=input_ids.device, dtype=torch.long)
            sequence_output, predict_logits,_ = self.forward(user_id, input_ids,segment)
            target_pos_i = target_pos
            target_neg_i = target_neg
        return sequence_output, predict_logits, target_pos_i, target_neg_i

    def finetune_infer_n_1(self, user_id, input_ids, target_pos, target_neg):
        generate_list, target_pos_i, target_neg_i,segment = self.GPT_rec_next_item_test_n_1(user_id, input_ids, target_pos, target_neg)
        return generate_list,target_pos_i, target_neg_i, segment

    def finetune_infer_all(self, user_id, input_ids, target_pos, target_neg):
        generate_list, target_pos_i, target_neg_i,segment = self.GPT_rec_next_item_test_all(user_id, input_ids, target_pos, target_neg)
        return generate_list,target_pos_i, target_neg_i, segment


class linear_dot_product_attnention(nn.Module):
    def __init__(self, d_k, attn_pdrop, eps=1e-6):
        super(linear_dot_product_attnention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(attn_pdrop)
        self.eps =eps

    def linear_attention(self,q, k, v, eps, attn_mask):

        Q = q.contiguous().permute(0, 2, 1, 3)
        K = k.contiguous().permute(0, 2, 1, 3)
        V = v.contiguous().permute(0, 2, 1, 3)
        KV = torch.einsum('...sd,...se->...de', K, V)
        Z = 1.0 / torch.einsum('...sd,...d->...s', Q, K.sum(dim=-2) + eps)
        V_new = torch.einsum('...de,...sd,...s->...se', KV, Q, Z)
        V_new= V_new.contiguous().permute(0, 2, 1, 3)

        return V_new

    def permute_for_matrix(self,matrix: torch.Tensor):
        assert matrix.dim() == 4
        return matrix.contiguous().permute(0, 2, 1, 3)

    def causal_linear(self,Q, K, V):
        Q = self.permute_for_matrix(Q)
        K = self.permute_for_matrix(K)
        V = self.permute_for_matrix(V)
        V_new = causal_dot_product(Q, K, V)
        V_new= self.permute_for_matrix(V_new)
        return V_new

    def forward(self, q, k, v, attn_mask):
        causal_linear_view = self.causal_linear(q, k, v)
        return causal_linear_view
        #return output, attn_weights

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        outputs = self.gelu(self.linear1(inputs))
        # |outputs| : (batch_size, seq_len, d_ff)
        outputs = self.linear2(outputs)
        # |outputs| : (batch_size, seq_len, d_model)

        return outputs

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_pdrop, resid_pdrop):
        super(DecoderLayer, self).__init__()
        # self.mha = LinearMultiHeadAttention(d_model, n_heads, attn_pdrop)
        self.mha = CausalSelfAttention(d_model, n_heads, attn_pdrop)
        self.dropout1 = nn.Dropout(resid_pdrop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-5)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(resid_pdrop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, inputs, attn_mask):
        attn_outputs = self.mha(inputs, inputs, inputs)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)

        return ffn_outputs

class TransformerDecoder(nn.Module):
    def __init__(self,user_size,item_size,max_seq_length,hidden_size,num_hidden_layers,num_attention_heads,dim_feed_forward,hidden_dropout_prob,attention_probs_dropout_prob,
                                        residual_prob,pad_id):
        super(TransformerDecoder,self).__init__()
        self.pad_id = pad_id
        self.user_embeddings = nn.Embedding(user_size, hidden_size)
        self.item_embeddings = nn.Embedding(item_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.segment_embeddings = nn.Embedding(2, hidden_size)

        self.max_seq_length=max_seq_length
        # layers
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size, num_attention_heads, dim_feed_forward, attention_probs_dropout_prob, residual_prob) for _ in range(num_hidden_layers)])

        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        nn.init.normal_(self.segment_embeddings.weight, std=0.02)

    def forward(self,user_id, inputs, segment):
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1)
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        user_id = user_id.unsqueeze(1)+1
        user_id = user_id.repeat_interleave(self.max_seq_length, dim=1)

        outputs = self.item_embeddings(inputs) + self.user_embeddings(user_id) \
                  + self.position_embeddings(positions) + self.segment_embeddings(segment)

        attn_pad_mask_linear = self.get_attention_padding_mask_linear(inputs, self.pad_id)

        for layer in self.layers:
            outputs = layer(outputs, attn_pad_mask_linear)
        return outputs

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)
        return attn_pad_mask

    def get_attention_padding_mask_linear(self, inputs, pad_id):
        attn_pad_mask0 = inputs.eq(pad_id)
        return attn_pad_mask0


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, d_model, n_heads, attn_pdrop):
        super().__init__()
        self.n_embd = d_model
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attn_pdrop)
        self.eps = 1e-6

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def causal_dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhldm", k, v)
        kv = torch.cumsum(kv, dim=2)
        qkv = torch.einsum("nhld,nhldm->nhlm", q, kv)
        return qkv

    def forward(self,  Q, K, V,):
        queries, values, keys =  Q, K, V,
        ## 1. Linear projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # 3. Causal Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhld->nhl", queries + self.eps, keys.cumsum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhld->nhl", keys + self.eps, queries.cumsum(dim=2) + self.eps))
        # approximate normal conservation col and row by multiplying corresponding element number
        normal = (((torch.arange(queries.shape[2])).float() + 1.0)).to(queries.device)[None, None, :]
        sink_incoming = sink_incoming * normal
        source_outgoing = source_outgoing * normal
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhld->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).cumsum(dim=2) + self.eps) / normal
        conserved_source = torch.einsum("nhld,nhld->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).cumsum(
                                            dim=2) + self.eps) / normal
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink)
        conserved_source = torch.exp(conserved_source)
        source_competition = (conserved_source / conserved_source.cumsum(dim=-1)) * normal
        # (4) Causal dot product
        x = (self.causal_dot_product(queries * (sink_incoming[:, :, :, None] / normal[:, :, :, None]),
                                     # for value normalization
                                     keys,
                                     values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        #x = self.dropout(x)
        return x
