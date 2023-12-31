import os
import numpy as np
import random
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from trainers import FinetuneTrainer
from GPT4Rec import GPTReclinearModel
from utils import EarlyStopping, get_user_seqs, check_path, set_seed,get_local_time

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument("--data_process", default='SASRec', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=5, type=int, help="pretrain epochs 5, 10, 15, 20...")

    # model args
    parser.add_argument("--model_name", default='Finetune_full', type=str)
    parser.add_argument("--Finetune_train_Prompts", default='Yes', type=str, help='whether add Prompts')
    parser.add_argument("--Finetune_train_Prompts_n_1", default='Yes', type=str, help='whether add Prompts')
    parser.add_argument("--Finetune_infer_Norecall_history_prompts", default='No', type=str,help='whether add Prompts in validate and test datasets')
    parser.add_argument("--Finetune_infer_Norecall_history_Prompts_n_1_or_all", default='No', type=str,help='whether add Prompts in validate and test datasets')
    parser.add_argument("--Finetune_logit_loss", default='Yes', type=str, help="generate number of next-window")
    parser.add_argument("--Finetune_generate_idx_next_softmax_logits", default='Yes', type=str, help="generate number of next-window")
    parser.add_argument("--Finetune_infer_GPT_recall_2", default='Yes', type=str, help='whether add Prompts')
    parser.add_argument("--Finetune_infer_GPT_recall_2_next_idx_is_output", default='Yes', type=str, help='whether add Prompts')
    parser.add_argument("--Finetune_infer_GPT_recall_20", default='No', type=str, help='whether add Prompts')
    parser.add_argument("--Finetune_infer_GPT_recall_2_history_prompts_n_1", default='No', type=str, help='whether add Prompts')
    parser.add_argument("--Finetune_infer_GPT_recall_2_history_prompts_all", default='No', type=str,
                        help='whether add Prompts')
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--dim_feed_forward", type=int, default=256,
                        help="dimensional of position-wise feed-forward networks")
    parser.add_argument("--prompts_emb_output", default='No', type=str, help='whether add Prompts')
    parser.add_argument("--pretrainstage", default='No', type=str, help='whether add Prompts')
    # parser.add_argument("--Finetune_train_Prompts_n_1", default='No', type=str, help='whether add Prompts')
    # parser.add_argument("--Finetune_train_Prompts_n_1", default='No', type=str, help='whether add Prompts')

    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="embedding dropout p")
    parser.add_argument('--residual_prob', type=float, default=0.1, help="residual dropout p")
    parser.add_argument('--pad_id', type=int, default=0, help="padding id for sequence")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--max_iters", type=int, default=10, help="number of inter fine tune")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--next_window_train", type=int, default=2, help="generate number of next-window")
    parser.add_argument("--next_window_test", type=int, default=2, help="generate number of next-window")
    parser.add_argument("--infer_GPT_recall_number", type=int, default=20, help="generate number of next-window")
    parser.add_argument("--infer_GPT_recall_number_total", type=int, default=20, help="generate number of next-window")
    parser.add_argument("--infer_GPT_recall_number_20", type=int, default=20, help="generate number of next-window")

    parser.add_argument("--rec_top_k", type=int, default=1, help="generate next-n item via GPT")
    parser.add_argument("--generate_p_train", type=float, default=1, help="generate candiadte item probability")
    parser.add_argument("--generate_p_test", type=float, default=1, help="generate candiadte item probability")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'

    users,user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    args.user_size = len(users) + 2

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.nowtime = get_local_time()

    # save model args
    if args.epochs == 0:
        args_str = f'{args.data_name}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_size}-{args.dim_feed_forward}-epochs-630-{args.ckp}'
    else:
        # save model args
        args_str = f'{args.model_name}-{args.data_name}-{args.ckp}-{args.lr}-{args.Finetune_train_Prompts}-{args.nowtime}-{args.do_eval}-{args.next_window_train}'

    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = GPTReclinearModel(args=args)
    trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)
    else:
        pretrained_path = os.path.join(args.output_dir,
                                            f'{args.data_name}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_size}-{args.dim_feed_forward}-epochs-630-{args.ckp}.pt')
        try:
            trainer.load(pretrained_path)
            print(f'Load Checkpoint From {pretrained_path}!')

        except FileNotFoundError:
            print(f'{pretrained_path} Not Found The Model !')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

import time
import torch
start = time.time()
main()
end = time.time()
print('finetune_time_cost：{}h'.format((end-start)/3600))