import pickle
import argparse
import collections

import tree
import torch
import models
import data_utils
import train_helper

import numpy as np

from tqdm import tqdm

MAX_LEN = 30
batch_size = 1000


parser = argparse.ArgumentParser()
parser.add_argument('--save_file', '-s', type=str)
parser.add_argument('--vocab_file', '-v', type=str)
parser.add_argument('--data_dir', '-d', type=str)
args = parser.parse_args()

def cosine_similarity(v1, v2):
    prod = (v1 * v2).sum(-1)
    v1_norm = (v1 ** 2).sum(-1) ** 0.5
    v2_norm = (v2 ** 2).sum(-1) ** 0.5
    return prod / (v1_norm * v2_norm)

save_dict = torch.load(
    args.save_file,
    map_location=lambda storage,
    loc: storage)

config = save_dict['config']
checkpoint = save_dict['state_dict']
config.debug = True

with open(args.vocab_file, "rb") as fp:
    W, vocab = pickle.load(fp)
    
with train_helper.experiment(config, config.save_prefix) as e:
    e.log.info("vocab loaded from: {}".format(args.vocab_file))
    model = models.vgvae(
        vocab_size=len(vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)
    model.eval()
    model.load(checkpointed_state_dict=checkpoint)
    e.log.info(model)

    def encode(d):
        global vocab, batch_size
        new_d = [[vocab.get(w, 0) for w in s.split(" ")] for s in d]
        all_y_vecs = []
        all_z_vecs = []

        for s1, m1, s2, m2, _, _, _, _, \
                _, _, _, _, _, _, _, _, _ in \
                tqdm(data_utils.minibatcher(
                    data1=np.array(new_d),
                    data2=np.array(new_d),
                    batch_size=100,
                    score_func=None,
                    shuffle=False,
                    mega_batch=0,
                    p_scramble=0.)):
                s1, m1, s2, m2 = \
                    model.to_vars(s1, m1, s2, m2)
                _, yvecs = model.yencode(s1, m1)
                _, zvecs = model.zencode(s2, m2)

                ymean = model.mean1(yvecs)
                ymean = ymean / ymean.norm(dim=-1, keepdim=True)
                zmean = model.mean2(zvecs)

                all_y_vecs.append(ymean.cpu().data.numpy())
                all_z_vecs.append(zmean.cpu().data.numpy())
        return np.concatenate(all_y_vecs), np.concatenate(all_z_vecs)
    
    
    #load sents 
    with open('bible_train.txt', 'r') as f:
        
    