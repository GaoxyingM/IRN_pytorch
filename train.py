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
from model import IRN

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

    n_train = trainQ.shape[0]     
    n_test = testQ.shape[0]
    n_val = validQ.shape[0]
    print(trainQ.shape, trainA.shape,trainP.shape,trainS.shape)
    
    # 找到答案所在的坐标
    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    valid_labels = np.argmax(validA, axis=1)
    batches = zip(range(0, n_train-args.batch_size, args.batch_size), range(args.batch_size, n_train, args.batch_size))

    model = IRN(args)
    optimizer = optim.Adam(model.parameters(), args.lr,weight_decay=1e-5)
    pre_batches = zip(range(0, Triples.shape[0]-args.batch_size, args.batch_size), range(args.batch_size, Triples.shape[0], args.batch_size))
    pre_val_preds = model.predict(Triples, validQ, validP)
    pre_test_preds = model.predict(Triples, testQ, testP)
    criterion = nn.CrossEntropyLoss()
    for t in range(0, args.nepoch):
        model.train()
        logits = model(Triples, trainQ,trainA,trainP, batches,pre_batches)
        loss = criterion(logits, trainA)
        optimizer.zero_grad()
        loss.backward()
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--edim', default=50, type=int, help="words vector dimension [50]")
    parser.add_argument('--nhop', default=3, type=int, help="number of hops [2/3+1]")
    parser.add_argument('--batch_size', default=50, type = int, help = "batch size to use during training [50]")
    parser.add_argument('--nepoch', default=5000, type=int, help="number of epoch to use during training [1000]")
    parser.add_argument('--inner_nepoch', default=3, type=int, help="PRN inner loop [5]")
    parser.add_argument('--init_lr', default=0.001, type=float, help="initial learning rate")
    parser.add_argument('--epsilon', default=1e-8, type=float, help="Epsilon value for Adam Optimizer")
    parser.add_argument('--max_grad_norm', default=20, type=int, help="clip gradients to this norm [20]")
    parser.add_argument('--dataset', default="pq2h", type=str, help="pq/pql/wc/")
    parser.add_argument('--checkpoint_dir', default="checkpoint", type=str, help="checkpoint directory")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
