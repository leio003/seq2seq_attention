from torch.utils.data import Dataset
import torch
import os
import sys
from tqdm import tqdm
import copy
data_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(data_dir)


class dailyDataset(Dataset):
    """

    """
    def __init__(self, path, word2idx, flag='train'):

        cached_features_file = os.path.join(
            data_dir,
            "data/cached_{}_{}".format('daily', flag)
        )

        if os.path.exists(cached_features_file):
            self.feature = torch.load(cached_features_file)
        else:
            with open(path, 'rb') as f:
                data = f.read().decode("utf-8")
            data = data.split('\n')
            self.feature = []
            for dialogue_index, dialogue in enumerate(tqdm(data)):
                utterances = dialogue.split("__eou__")
                dialogue_ids = [word2idx['[CLS]']]

                dia = {}
                label_list = []
                for idx, utterance in enumerate(utterances):
                    utterance = utterance.strip().lower()
                    if len(utterance) == 0:
                        continue
                    ids = [word2idx[word] for word in utterance.split()]

                    if idx % 2 == 0:
                        dialogue_ids.extend(ids)
                        dialogue_ids = dialogue_ids[:255]

                        dia['input_ids'] = dialogue_ids
                    else:
                        ids.append(word2idx['[SEP]'])
                        dia['label'] = ids
                        label_list.append(utterance)
                        temp = copy.deepcopy(dia)
                        self.feature.append(temp)
                        dialogue_ids.append(word2idx['[SEP]'])
                        dialogue_ids.extend(ids)

            self.feature.sort(key=lambda dic: len(dic['input_ids']), reverse=True)
            torch.save(self.feature, cached_features_file)

    def __getitem__(self, index):
        return self.feature[index]

    def __len__(self):
        return len(self.feature)