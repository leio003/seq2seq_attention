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
from dataset import dailyDataset
from seq2seqmodel import Encoder, AttnDecoder, seq2seq
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import json
import torch.nn.functional as F


SENT_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
logger = None
pad_id = 1
CLS = 2


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', default=False, action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--train_path', default='data/dialogues_train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--test_path', default='data/dialogues_test.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--log_path', default='log/training_{}.log'.format(datetime.now().strftime('%Y-%m-%d')), type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练batch size')
    parser.add_argument('--hidden_size', default=300, type=int, required=False, help='隐藏层大小')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--log_step', default=25, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=16, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=42, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--embedding_path', type=str, default='embedding/glove.txt', help="加载glove向量")
    parser.add_argument('--num_words', type=int, default=50000, help="embedding的词个数")
    parser.add_argument('--dropout', type=int, default=0.2, help="dropout的大小")
    parser.add_argument('--word2idx', type=str, default='embedding/word2idx.json', help="word2idx文件保存位置")
    return parser.parse_args()

def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

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

def load_word_embeddings(args):
    """
    Loads pre-trained embeddings from the specified path.

    @return (embeddings as an numpy array, word to index dictionary)
    """
    logger.info("加载glove向量，增加特殊字符")
    word2idx = defaultdict(lambda: 1)  # Maps a word to the index in the embeddings matrix
    emb_model = KeyedVectors.load_word2vec_format(args.embedding_path)
    num_words = args.num_words
    embedding_dim = 300
    # 初始化embedding_matrix，
    embedding_matrix = np.zeros((num_words + 4, embedding_dim))
    for idx, special_token in enumerate(SENT_TOKENS):
        word2idx[special_token] = idx
        embedding_matrix[idx, :] = np.random.randn(300)
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 50004 * 300
    for i in range(num_words):
        embedding_matrix[i + 4, :] = emb_model[emb_model.index2word[i]]
        word2idx[emb_model.index2word[i]] = i+4
    embedding_matrix = embedding_matrix.astype('float32')
    emb = torch.from_numpy(embedding_matrix)

    logger.info("加载完成")
    json_str = json.dumps(word2idx, indent=4)
    with open(args.word2idx, 'w') as json_file:
        json_file.write(json_str)
    logger.info("word2idx文件保存完成")

    return emb, word2idx


def collate_fn_gen(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    label_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    max_label_len = 0
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]['input_ids']):
            max_input_len = len(batch[btc_idx]['input_ids'])

        if max_label_len < len(batch[btc_idx]['label']):
            max_label_len = len(batch[btc_idx]['label'])

    # 使用pad_id对小于max_input_len的input_id进行补全
    # print(max_input_len)
    for btc_idx in range(btc_size):
        ids, labels = batch[btc_idx]['input_ids'], batch[btc_idx]['label']
        input_len = len(ids)
        input_ids.append(ids)
        pad = [pad_id] * (max_input_len - input_len)
        pad.extend(input_ids[btc_idx])
        input_ids[btc_idx] = pad

        label_len = len(labels)
        label_ids.append(labels)
        pad = [pad_id] * (max_label_len - label_len)
        pad.extend(label_ids[btc_idx])
        label_ids[btc_idx] = pad

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def train(model, word2idx, args):
    gen_train_dataset = dailyDataset(args.train_path, word2idx)
    gen_train_dataloader = DataLoader(gen_train_dataset, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_fn_gen)
    total_steps = int(gen_train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    logger.info('starting training')
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    # 开始训练
    for epoch in tqdm(range(args.epochs)):
        epoch_start_time = datetime.now()
        for batch_idx, (input_, label_ids) in enumerate(tqdm(gen_train_dataloader)):
            input_ = input_.to(args.device)
            label_ids = label_ids.to(args.device)

            decoder_input = torch.ones((args.batch_size, 1), dtype=torch.long).to(args.device) * CLS
            # teacher-force
            output = []
            for i in range(label_ids.shape[1]):
                outputs, hidden, atten_weights = model(input_, decoder_input)
                decoder_input = label_ids[:, i].view(args.batch_size, 1)
                output.append(outputs)
            output = torch.cat(output, dim=1)
            output = output[:, :-1, :].contiguous().view(-1, outputs.size(2))
            label = label_ids[:, 1:].contiguous().view((-1))
            loss = loss_fn(output, label)
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                running_loss += loss.item()
                # 更新参数
                optimizer.step()
                # 清空梯度信息
                optimizer.zero_grad()
                overall_step += 1
                # 更新日志与tnesorboardX信息
                if (overall_step + 1) % args.log_step == 0:
                    logger.info(
                        "batch {} of epoch {}, loss {}".format(batch_idx + 1, epoch + 1, loss))
                    tb_writer.add_scalar('loss', loss.item(), overall_step)
        if ((epoch + 1) % 5 == 0):
            logger.info('saving model for epoch {}'.format(epoch + 1))
 # 当前训练对话模型
            model_path = join(args.dialogue_model_output_path,
                            'model_epoch{}.pt'.format(epoch + 1))
            torch.save(model, model_path)

        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    logger.info('saving model finally')
# 当前训练对话模型
    model_path = join(args.dialogue_model_output_path, 'model_finally.pt')
    torch.save(model, model_path)

    logger.info('training finished')

def evaluate(model, word2idx, args):
    test_dataset = dailyDataset(args.test_path, word2idx, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn_gen)
    model.eval()
    logger.info('starting evaluating')
    loss_ls = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    with torch.no_grad():
        for batch_idx, (input_, label_ids) in enumerate(tqdm(test_dataloader)):
            input_ = input_.to(args.device)
            label_ids = label_ids.to(args.device)
            decoder_input = torch.ones((args.batch_size, 1), dtype=torch.long).to(args.device) * CLS
            output = []
            for i in range(label_ids.shape[1]):
                outputs, hidden, atten_weights = model(input_, decoder_input)
                next_token_logits = outputs.squeeze(1)
                decoder_input = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                output.append(outputs)
            output = torch.cat(output, dim=1)
            output = output[:, :-1, :].contiguous().view(-1, outputs.size(2))
            label = label_ids[:, 1:].contiguous().view((-1))
            loss = loss_fn(output, label)
            loss_ls.append(loss.item())
        logger.info("finishing evaluating")
    return np.mean(loss_ls)

def main():
    args = setup_train_args()
    global logger
    logger = create_logger(args)
    logger.info('args config:\n{}'.format(args.__dict__))
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(args.device))
    # 设置随机数种子
    if args.seed:
        set_random_seed(args)


    # 加载预训练模型
    # emb, word2idx = load_word_embeddings(args)
    word2idx = defaultdict(lambda: 1)
    with open(args.word2idx, 'r') as load_f:
        wo = json.load(load_f)
    for k, v in wo.items():
        word2idx[k] = v
    model = create_model(args)
    generator = model.to(args.device)
    # 创建模型存放文件夹
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)
    # 开始训练
    train(generator, word2idx, args)
    # 测试模型
    eval_loss = evaluate(model, word2idx, args)
    logger.info('eval loss:{}'.format(eval_loss))


if __name__ == "__main__":
    main()