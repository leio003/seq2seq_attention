from datetime import datetime
import argparse
import torch
from tqdm import tqdm
import random
import numpy as np
import logging
from gensim.models import KeyedVectors
import re
from collections import defaultdict
import os
from os.path import join, exists
from dataset import MyDataset
from seq2seqmodel import Encoder, AttnDecoder, seq2seq
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import json

pad_id = 1
CLS = 2

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', default=False, action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--train_raw_path', default='data/dialogues_train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--test_raw_path', default='data/dialogues_test.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--test_tokenized_path', default='data/test_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='log/training_{}.log'.format(datetime.now().strftime('%Y-%m-%d')), type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练batch size')
    parser.add_argument('--hidden_size', default=300, type=int, required=False, help='隐藏层大小')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=25, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=16, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model_{}/'.format(datetime.now().strftime('%Y-%m-%d')), type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=42, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--embedding_path', type=str, default='embedding/glove.txt', help="加载glove向量")
    parser.add_argument('--num_words', type=int, default=50000, help="embedding的词个数")
    parser.add_argument('--dropout', type=int, default=0.2, help="dropout的大小")
    parser.add_argument('--word2idx', type=str, default='embedding/word2idx.json', help="word2idx文件保存位置")
    return parser.parse_args()

def create_model(args, emb=None):
    encoder = Encoder(vocab_size=args.num_words+4,
                      hidden_size=args.hidden_size,
                      dropout=args.dropout,
                      emb=emb)
    decoder = AttnDecoder(vocab_size=args.num_words+4,
                      hidden_size=args.hidden_size,
                      dropout=args.dropout,
                      emb=emb)
    model = seq2seq(encoder, decoder)

    return model


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []

    btc_size = len(batch)
    # print(btc_size)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    # print(max_input_len)
    for btc_idx in range(btc_size):
        ids = batch[btc_idx]
        input_len = len(ids)
        input_ids.append(ids)
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))

    return torch.tensor(input_ids, dtype=torch.long)

args = setup_train_args()
model = create_model(args)

with open(args.train_tokenized_path, "r", encoding="utf8") as f:
    data = f.read()
train_list = data.split("\n")
train_dataset = MyDataset(train_list)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_fn)

model.train()
for epoch in tqdm(range(args.epochs)):
    epoch_start_time = datetime.now()
    for batch_idx, input_ids in enumerate(tqdm(train_dataloader)):
        input_ = input_ids[:, :-1]
        label_ids = input_ids[:, 1:]
        decoder_input = torch.LongTensor([[CLS for _ in range(args.batch_size)]]).view(args.batch_size, 1)
        output, hidden, atten_weights = model.forward(input_, decoder_input)
