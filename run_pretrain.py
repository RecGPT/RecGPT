import numpy as np
import random
import torch
from torch.utils.data import DataLoader, RandomSampler

import os
import argparse

from datasets import PretrainDataset,SASRecDataset
from trainers import PretrainTrainer

from GPT4Rec import GPTReclinearModel

from utils import get_user_seqs_long, check_path, set_seed, get_local_time, get_user_seqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)

    # model args
    parser.add_argument("--model_name", default='Pretrain', type=str)
    parser.add_argument("--data_process", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model") # 768
    parser.add_argument("--dim_feed_forward", type=int, default=256, help="dimensional of position-wise feed-forward networks") #3072

    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="embedding dropout p")
    parser.add_argument('--residual_prob', type=float, default=0.1, help="residual dropout p")
    parser.add_argument('--pad_id', type=int, default=0, help="padding id for sequence")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--GPT_rec_item', default=1, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=100, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--pretrainstage", default='Yes', type=str, help='whether add Prompts')

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'

    # concat all user_seq get a long sequence, from which sample neg segment for SP
    # sasrec
    if args.data_process == 'SASRec':
        users, user_seq, max_item, _, _ = get_user_seqs(args.data_file)
    else:
        users, user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)

    args.user_size = len(users) + 2
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    # save model args
    # args_str = f'{args.model_name}-{args.data_name}-{nowtime}'
    args_str = f'{args.model_name}-{args.data_name}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_size}-{args.dim_feed_forward}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    model = GPTReclinearModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, args)

    for epoch in range(args.pre_epochs):
        if args.data_process == 'SASRec':
            pretrain_dataset = SASRecDataset(args, user_seq, data_type='train')
        else:
            pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)
        trainer.pretrain(epoch, pretrain_dataloader)

        if (epoch+1) % 5 == 0:
            ckp = f'{args.data_name}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_size}-{args.dim_feed_forward}-epochs-630-{epoch+1}.pt'
            checkpoint_path = os.path.join(args.output_dir, ckp)
            trainer.save(checkpoint_path)

import time
import torch
torch.cuda.synchronize()
start = time.time()
main()
torch.cuda.synchronize()
end = time.time()
print('time_cost：{}hours'.format((end-start)/3600))
