import random

import torch
from torch.utils.data import Dataset
from utils import neg_sample

class PretrainDataset(Dataset):

    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()
        self.users_id = []

    def split_sequence(self):
        j = 0
        self.users_id = []
        for seq in self.user_seq:
            j += 1
            input_ids = seq[-(self.max_len+2):-2] # keeping same as train set
            for i in range(1,len(input_ids)):
                self.part_sequence.append(input_ids[:i+1]+[j])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):
        sequence = self.part_sequence[index] # pos_items
        user_id = sequence[-1]-1
        sequence = sequence[:-1]
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        # item_set = set(sequence)
        # for item in sequence[:-1]:
        #     # prob = random.random()
        #     # if prob < self.args.mask_p:
        #     #     masked_item_sequence.append(self.args.mask_id)
        #     #     neg_items.append(neg_sample(item_set, self.args.item_size))
        #     # else:
        #     masked_item_sequence.append(item)
        #     neg_items.append(item)
        # # add mask at the last position
        # masked_item_sequence.append(self.args.mask_id)
        input_id = sequence[:-1]
        # input_id.append(self.args.mask_id)
        #masked_item_sequence = input_id
        pos_items = sequence[1:]
        item_set = set(sequence)
        for _ in input_id:
            neg_items.append(neg_sample(item_set, self.args.item_size))
        # Segment Prediction
        # if len(sequence) < 2:
        #     masked_segment_sequence = sequence
        #     pos_segment = sequence
        #     neg_segment = sequence
        # else:
        #     sample_length = random.randint(1, len(sequence) // 2)
        #     start_id = random.randint(0, len(sequence) - sample_length)
        #     neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
        #     pos_segment = sequence[start_id: start_id + sample_length]
        #     neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]


        # assert len(masked_segment_sequence) == len(sequence)
        # assert len(pos_segment) == len(sequence)
        # assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(input_id)
        input_ids = [0] * pad_len + input_id
        # pos_items = [0] * pad_len + sequence
        pos_items = [0] * pad_len + pos_items
        neg_items = [0] * pad_len + neg_items

        input_ids = input_ids[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len

        cur_tensors = (torch.tensor(user_id, dtype=torch.long),
                       torch.tensor(input_ids, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
        )
        return cur_tensors

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        # user_id = [index + 1]
        user_id = index

        # print("user_id_train", user_id) # 692
        # exit()
        items = self.user_seq[index]
        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # 0, 1, 2, 3
        # label 4

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            # input_ids1 = [items[0]]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

            # Kuairand_dataset_no_x1_x2
            # itemss = items[:-6] + items[-4:]
            # input_ids = itemss[:-2]
            # target_pos = itemss[1:-1]
            # answer = [itemss[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

            # Kuairand_dataset_no_x1_x2
            # itemsss = items[:-6] + items[-4:]
            # input_ids = itemsss[:-1]
            # target_pos = itemsss[1:]
            # answer = [itemsss[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        #input_ids = [0] * pad_len + input_ids

        if self.data_type == "train":
            pad_len = self.max_len - len(input_ids)
            # pad_len1 = self.max_len - len(input_ids1)
            # input_ids = [0] * pad_len1 + input_ids1
            input_ids = [0] * pad_len + input_ids
            # target_pos = target_pos + [0] * pad_len
            target_pos = [0] * pad_len+ target_pos
            target_neg = [0] * pad_len + target_neg
        else:
            pad_len = self.max_len - len(input_ids)
            input_ids = [0] * pad_len + input_ids
            # target_pos = target_pos + [0] * pad_len
            target_pos = [0] * pad_len + target_pos
            target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )
        return cur_tensors

    def __len__(self):
        return len(self.user_seq)