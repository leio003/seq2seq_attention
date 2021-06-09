import torch
import os
import json
import argparse
from datetime import datetime
import logging
import torch.nn.functional as F
from seq2seqmodel import Encoder, Decoder, seq2seq
from collections import defaultdict



SENT_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
logger = None
pad_id = 1

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--log_path', default='data/interacting.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--dialogue_model_path', default='dialogue_model_2021-06-09/model_finally.pt', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=42, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--word2idx', type=str, default='embedding/word2idx.json', help="word2idx文件保存位置")
    return parser.parse_args()


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


def main():
    args = set_interact_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(args.device))

    word2idx = defaultdict(lambda: 1)
    with open(args.word2idx, 'r') as load_f:
        wo = json.load(load_f)
    for k,v in wo.items():
        word2idx[k] = v

    idx2word = dict([(v,k) for k, v in word2idx.items()])

    model = torch.load(args.dialogue_model_path)
    model.to(args.device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)

        # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')
    with open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8') as samples_file:
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
        while True:
            cnt = 0
            # try:
            cnt += 1
            text = input("user:")
            if args.save_samples_path:
                samples_file.write("user:{}\n".format(text))
            history.append([word2idx[word] for word in text.split()])
            input_ids = [word2idx['[CLS]']]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(word2idx['[SEP]'])
            curr_input_tensor = torch.tensor(input_ids).long().to(args.device).unsqueeze(0)
            output_id = [word2idx['[CLS]']]
            curr_output_tensor = torch.tensor(output_id).long().to(args.device).unsqueeze(0)
            generated = []
            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = model.forward(x=curr_input_tensor, y=curr_output_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[word2idx['[UNK]']] = -float('Inf')
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                if next_token == word2idx['[SEP]']:  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token.item())
                next_token = next_token.unsqueeze(0)
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=1)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            history.append(generated)
            text = [idx2word[idx] for idx in generated]
            print("chatbot:" + " ".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format(" ".join(text)))



if __name__ == '__main__':
    main()
