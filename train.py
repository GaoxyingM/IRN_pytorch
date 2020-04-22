from IPython import embed
import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
from preprocess import process_data
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split

def train(args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    KB_file = 'data/2H-kb.txt'
    data_file = 'data/2H.txt'
    start = time.time()
    Q,A,P,S,Triples,args.query_size, word2id, ent2id, rel2id = process_data(KB_file, data_file)
    args.path_size = len(P[0])
    args.nhop = args.path_size / 2
    
    print ("read data cost %f seconds" %(time.time()-start))
    args.nwords = len(word2id) 
    args.nrels = len(rel2id) 
    args.nents = len(ent2id)

    trainQ, testQ, trainA, testA, trainP, testP, trainS, testS = train_test_split(Q, A, P, S, test_size=.1, random_state=123)
    trainQ, validQ, trainA, validA, trainP, validP, trainS, validS = train_test_split(trainQ, trainA, trainP, trainS, test_size=.11, random_state=0)

    # for UNSEEN relations (incomplete kb setting, change data_utils.py)
    if args.unseen:
        id_c=[]
        for idx in range(trainQ.shape[0]):
            if trainP[idx][-4] == 1 or trainP[idx][-4]==2 or trainP[idx][-4]==3:
                id_c.append(idx)
        trainQ = np.delete(trainQ,id_c,axis=0)
        trainA = np.delete(trainA,id_c,axis=0) 
        trainP = np.delete(trainP,id_c,axis=0)
        trainS = np.delete(trainS,id_c,axis=0) 
    
    n_train = trainQ.shape[0]     
    n_test = testQ.shape[0]
    n_val = validQ.shape[0]
    print("Training Size", n_train) 
    print("Validation Size", n_val) 
    print("Testing Size", n_test) 

    #
    #other data and some flags
    #
    id2word = dict(zip(word2id.values(), word2id.keys()))
    id2rel = dict(zip(rel2id.values(), rel2id.keys())) #{0: '<end>', 1: 'cause_of_death', 2: 'gender', 3: 'profession', 4: 'institution', 5: 'religion', 6: 'parents', 7: 'location', 8: 'place_of_birth', 9: 'nationality', 10: 'place_of_death', 11: 'spouse', 12: 'children', 13: 'ethnicity'} 
    
    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    valid_labels = np.argmax(validA, axis=1)

    batches = zip(range(0, n_train-args.batch_size, args.batch_size), range(args.batch_size, n_train, args.batch_size))
    r = np.arange(n_train) # instance idx to be shuffled
    l = n_train / args.batch_size * args.batch_size #total instances used in training 
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--edim', default=50, type=int, help="words vector dimension [50]")
    parser.add_argument('--nhop', default=3, type=int, help="number of hops [2/3+1]")
    parser.add_argument('--nepoch', default=5000, type=int, help="number of epoch to use during training [1000]")
    parser.add_argument('--inner_nepoch', default=3, type=int, help="PRN inner loop [5]")
    parser.add_argument('--init_lr', default=0.001, type=float, help="initial learning rate")
    parser.add_argument('--epsilon', default=1e-8, type=float, help="Epsilon value for Adam Optimizer")
    parser.add_argument('--max_grad_norm', default=20, type=int, help="clip gradients to this norm [20]")
    parser.add_argument('--dataset', default="pq2h", type=str, help="pq/pql/wc/")
    parser.add_argument('--checkpoint_dir', default="checkpoint", type=str, help="checkpoint directory")
    parser.add_argument('--unseen', default=False, type=bool, help="True to hide 3 relations when training [False]")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
