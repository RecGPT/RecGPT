import time
import numpy as np
import tqdm
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from utils import recall_at_k, ndcg_k, get_metric

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.criterion = nn.BCELoss()
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score_20_0(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "H5": eval('{:.4f}'.format(recall[0])), "H10": eval('{:.4f}'.format(recall[1])),"H20": eval('{:.4f}'.format(recall[2])),
            "N5": eval('{:.4f}'.format(ndcg[0])), "N10": eval('{:.4f}'.format(ndcg[1])), "N20": eval('{:.4f}'.format(ndcg[2]))
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return [recall[0],recall[1],recall[2], ndcg[0],ndcg[1],ndcg[2]], str(post_fix)
    def get_full_sort_score_20(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "H20": eval('{:.4f}'.format(recall[0])),"N20": eval('{:.4f}'.format(ndcg[0]))
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return [recall[0], ndcg[0]], str(post_fix)

    def get_full_sort_score_10(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT10": eval('{:.4f}'.format(recall[0])),
            "NDCG@10": eval('{:.4f}'.format(ndcg[0]))
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return [recall[0],ndcg[0]], str(post_fix)

    def get_full_sort_score_5(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT5": eval('{:.4f}'.format(recall[0])),
            "NDCG@5": eval('{:.4f}'.format(ndcg[0]))
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return [recall[0],ndcg[0]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):

        # [batch seq_len hidden_size]
        pos_emb = self.model.decoder.item_embeddings(pos_ids)
        neg_emb = self.model.decoder.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = seq_out.view(-1, self.args.hidden_size) #[batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.decoder.item_embeddings.weight
        # [batch hidden_size]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def infer_GPT_recall_2(self, user_ids, input_id, segment, top_k=1, temperature=1.0):
        infer_GPT_recall_total = self.args.infer_GPT_recall_number_total
        infer_GPT_generate = self.args.infer_GPT_recall_number
        recall_list =[]
        for j in range(2):  # infer_GPT_generate
            if j == 0:
                ind_num = infer_GPT_generate
                ones_seg = torch.zeros(input_id.size()[0], 1, device=input_id.device,
                                      dtype=torch.long)  # one_seg=1
            else:
                ind_num = infer_GPT_recall_total-infer_GPT_generate
                ones_seg = torch.ones(input_id.size()[0], 1, device=input_id.device, dtype=torch.long)
            segment = torch.cat((segment, ones_seg), dim=1)[:,-self.args.max_seq_length:]
            if self.args.Finetune_generate_idx_next_softmax_logits == "Yes":
                recommend_output, predict_logits, _ = self.model.forward(user_ids, input_id, segment)
                predict_logits = predict_logits[:, -1, :] / temperature
                probs = F.softmax(predict_logits, dim=-1)
                _, idx_next = torch.topk(probs, k=top_k, dim=-1)
            else:
                _, _, idx_next = self.model.forward(user_ids, input_id, segment)

            new_id = torch.cat((input_id[:, 1:], idx_next), dim=1)
            input_id = new_id

            if self.args.Finetune_infer_GPT_recall_2_next_idx_is_output == "No":
                recommend_output = self.model.decoder.item_embeddings(idx_next)
                recommend_output = torch.squeeze(recommend_output)
            else:
                recommend_output = recommend_output[:, -1, :]
            if j == 0:
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                # print("user_ids",user_ids)
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -ind_num)[:, -ind_num:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                recall_list.append(batch_pred_list)
            else:
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -100)[:, -100:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                recall_0 = recall_list[0].tolist()
                batch_pred_list_i = []
                for k in range(len(recall_0)):
                    T = []
                    [T.append(i) for i in batch_pred_list[k] if not i in recall_0[k]]
                    batch_pred_list_i.append(T[:20])
                batch_pred_list_j = torch.tensor(batch_pred_list_i)[:,:ind_num].tolist()
                recall_list.append(batch_pred_list_j)
        recall_0 = recall_list[0].tolist()
        recall_1 = recall_list[1]
        recall_result = np.concatenate((recall_0,recall_1),axis=1)

        return recall_result

class PretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = f'GPT-{self.args.data_name}-'f'{self.args.model_name}'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        gpt_loss_avg = 0.0
        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            # sasrec
            if self.args.data_process == 'SASRec':
                user_id, input_ids, pos_items, neg_items, _ = batch
            else:
                user_id, input_ids, pos_items, neg_items = batch

            gpt_loss = self.model.pretrain(user_id,input_ids, pos_items, neg_items)
            self.optim.zero_grad()
            gpt_loss.backward()
            self.optim.step()

            gpt_loss_avg += gpt_loss.item()
        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {"epoch": epoch,"gpt_loss_avg": '{:.4f}'.format(gpt_loss_avg / num)
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            self.iter_num = 0
            self.iter_time = time.time()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_id, input_ids, target_pos, target_neg, _ = batch
                # fine tune GPT inter
                if self.args.Finetune_train_Prompts == "Yes":
                    sequence_output, predict_logits, target_pos, target_neg = self.model.finetune_train(user_id, input_ids, target_pos,
                                                                                  target_neg, GPT=True)
                else:
                    sequence_output, predict_logits, target_pos, target_neg = self.model.finetune_train(user_id, input_ids, target_pos,
                                                                                  target_neg, GPT=False)
                    
                if self.args.Finetune_logit_loss == "No":
                    self.loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                else:
                    self.loss = F.cross_entropy(predict_logits.view(-1, predict_logits.size(-1)), target_pos.view(-1),ignore_index=0)

                self.optim.zero_grad()
                self.loss.backward()
                self.optim.step()

                rec_avg_loss += self.loss.item()
                rec_cur_loss = self.loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        else:
            self.model.eval()
            pred_list = None
            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    if self.args.Finetune_infer_GPT_recall_2 == "No": # Finetune_infer_GPT_no_recall_2
                        if self.args.Finetune_infer_Norecall_history_prompts == "Yes":
                            recommend_output, _, _, _ = self.model.finetune_test(user_ids,input_ids,target_pos, target_neg,GPT=True)
                        else:
                            recommend_output, _, _, _ = self.model.finetune_test(user_ids, input_ids, target_pos, target_neg,GPT=False)
                        recommend_output = recommend_output[:, -1, :]
                        rating_pred = self.predict_full(recommend_output)
                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                        ind = np.argpartition(rating_pred, -20)[:, -20:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                    else:
                        if self.args.Finetune_infer_GPT_recall_20 == "Yes":
                            batch_pred_list = self.model.infer_GPT_recall_20(user_ids, input_ids)
                        else:
                            if self.args.Finetune_infer_GPT_recall_2_history_prompts_n_1 == "Yes":
                                input_ids, _, _, segment = self.model.finetune_infer_n_1(user_ids, input_ids, target_pos,target_neg)
                                batch_pred_list = self.infer_GPT_recall_2(user_ids, input_ids, segment)
                            elif self.args.Finetune_infer_GPT_recall_2_history_prompts_all == "Yes":
                                input_ids, _, _, segment = self.model.finetune_infer_all(user_ids, input_ids, target_pos,target_neg)
                                batch_pred_list = self.infer_GPT_recall_2(user_ids, input_ids, segment)
                            else:
                                segment = torch.zeros(input_ids.size(), device=input_ids.device, dtype=torch.long)
                                batch_pred_list = self.infer_GPT_recall_2(user_ids, input_ids, segment)

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

                if self.args.infer_GPT_recall_number_total == 20:
                    return self.get_full_sort_score_20(epoch, answer_list, pred_list)
                elif self.args.infer_GPT_recall_number_total == 10:
                    return self.get_full_sort_score_10(epoch, answer_list, pred_list)
                else:
                    return self.get_full_sort_score_5(epoch, answer_list, pred_list)
            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
