import torch
import torch.nn as nn
from utils import position_encoding
import os


class IRN(nn.Module):
    def __init__(self, args):
        self._margin = 4  # 外边距
        self._batch_size = args.batch_size
        self._vocab_size = args.nwords
        self._rel_size = args.nrels
        self._ent_size = args.nents
        self._sentence_size = args.query_size
        self._embedding_size = args.edim
        self._path_size = args.path_size
        self._hops = args.nhop
        self._max_grad_norm = args.max_grad_norm
        self._name = "IRN"
        self._inner_epochs = args.inner_nepoch
        self._checkpoint_dir = args.checkpoint_dir+'/'+self._name
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
        self.build_vars()

    def forward(self, Triples, Q, A, P, batches, pre_batches):
        np.random_shuffle(batches)
        for i in range(self._inner_epoch):
            np.random_shuffle(pre_batches)
            pre_total_cost  = 0.0
            for s,e in pre_batches:
                pre_total_cost += self.batch_pretrain(Triples[s:e],Q[0:FLAGS.batch_size],A[0:FLAGS.batch_size],np.argmax(A[0:FLAGS.batch_size], axis=1),P[0:FLAGS.batch_size])
        total_cost = 0.0
        for s,e in batches:
            total_cost += self.batch_fit(Triples[s:e],Q[s:e],A[s:e],np.argmax(A[s:e], axis=1),P[s:e])
        train_pres = self.predict(Triples, Q, P)
        return train_pres

    def build_vars(self):
        nil_word_slot = torch.zeros(1, self._embedding_size)
        nil_rel_slot = torch.zeros(1, self._embedding_size)
        self.E = torch.cat((nil_word_slot, nn.init.xavier_normal_(
            torch.Tensor(self._ent_size-1, self._embedding_size))), dim=0)
        self.Q = torch.cat((nil_word_slot, nn.init.xavier_normal_(
            torch.Tensor(self._vocab_size-1, self._embedding_size))), dim=0)
        self.R = torch.cat((nil_rel_slot, nn.init.xavier_normal_(
            torch.Tensor(self._rel_size-1, self._embedding_size))), dim=0)

        self.E.requires_grad = True
        self.Q.requires_grad = True
        self.R.requires_grad = True

        self.Mrq = nn.init.xavier_normal_(torch.Tensor(
            self._embedding_size, self._embedding_size))
        self.Mrs = nn.init.xavier_normal_(torch.Tensor(
            self._embedding_size, self._embedding_size))
        self.Mse = nn.init.xavier_normal_(torch.Tensor(
            self._embedding_size, self._embedding_size))

        self.Mrq.requires_grad = True
        self.Mrs.requires_grad = True
        self.Mse.requires_grad = True

        self._zeros = torch.zeros(1)

    def _pretrance(self, _KBs, _paddings):
        h = _KBs[:, 0]
        r = _KBs[:, 1]
        t = _KBs[:, 2]
        tt = _paddings

        h_emb = self.E[h.long()]
        r_emb = self.R[r.long()]
        t_emb = self.E[t.long()]
        tt_emb = self.E[tt.long()]
        l_emb = torch.matmul((h_emb+r_emb), self.Mse)
        s = (l_emb-t_emb)*(l_emb-t_emb)
        ss = (l_emb-tt_emb)*(l_emb-tt_emb)

        loss = self._margin + torch.sum(s, dim=1) - torch.sum(ss, dim=1)
        if loss > 0:
            return loss
        else:
            return 0

    def _inference(self, _paths, _queries):
        loss = self._zeros.unsqueeze(0)
        s_index = _paths[:, 0].unsqueeze(0)
        q_emb = self.Q[_queries.long()]
        q = torch.sum(q_emb, dim=1)
        state = self.E[s_index.long()].squeeze(1)
        p = s_index
        for hop in range(self._hops):
            step = 2 * hop
            gate = torch.matmul(q, torch.matmul(self.R, self.Mrq.t(
            ))) + torch.matmul(state, torch.matmul(self.R, self.Mrs.t()))
            rel_logits = gate
            r_index = torch.argmax(rel_logits, dim=1)
            gate = torch.softmax(gate)
            # real_rel_onehot = torch.nn.functional.one_hot(
            #     _paths[:, step+1], num_classes=self._rel_size)
            real_rel_onehot = _paths[:, step+1]
            predict_rel_onehot = torch.nn.functional.one_hot(
                r_index, num_classes=self._rel_size)
            state = state + torch.matmul(gate, torch.matmul(self.R, self.Mrs))
            critrion = nn.CrossEntropyLoss(reduce=False)
            loss += critrion(rel_logits, real_rel_onehot)
            q = q-torch.matmul(gate, torch.matmul(self.R, self.Mrq))
            value = torch.matmul(state, self.Mse)
            ans = torch.matmul(value, self.E.t())
            t_index = torch.argmax(ans, dim=1).float()
            r_index = r_index.float()
            t_index = r_index/(r_index+1e-15)*t_index + \
                (1-r_index/(r_index+1e-15)) * p[:, -1].float()

            p = torch.cat((p, r_index.int().view(-1, 1)), dim=1)
            p = torch.cat((p, t_index.int().view(-1, 1)), dim=1)

            real_ans_onehot = torch.nn.one_hot(
                _paths[:, step+2], num_class=self._ent_size)
            loss += critrion(ans, real_ans_onehot)

            return loss, p

    def match(self):
        Similar = torch.matmul(torch.matmul(self.R,self.Mrq),self.Q.t())
        _, idx = torch.topk(Similar, 5)
        return idx
    
    def batch_pretrain(self, KBs, queries, answers, answers_id, paths):
        nexample = KBs.shape[0]
        keys = np.repeat(np.reshape(
            np.arange(self._rel_size), [1, -1]), nexample, axis=0)
        pad = np.random.randint(low=0, high=self._ent_size, size=nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        KB_batch_loss = self.pretrance(KBs,pad)
        loss = torch.sum(KB_batch_loss)
        return loss
    
    def batch_fit(self, KBs, queries, answers, answers_id, paths):
        nexample = queries.shape[0]
        keys = np.repeat(np.reshape(
            np.arange(self._rel_size), [1, -1]), nexample, axis=0)
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        QA_batch_loss,p = self._inference(paths, queries)
        loss = torch.sum(QA_batch_loss)
        self.E = self.E/torch.pow(self.E, 2).sum(dim=1, keepdim= True)
        self.R = self.E/torch.pow(self.E, 2).sum(dim=1, keepdim= True)
        self.Q = self.E/torch.pow(self.E, 2).sum(dim=1, keepdim= True)
        return loss

    def predict(self, KBs, queries, paths):
        nexample = queries.shape[0]
        keys = np.repeat(np.reshape(
            np.arange(self._rel_size), [1, -1]), nexample, axis=0)
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        self.QA_predict_op  = self._inference(paths, queries)
        return self.QA_predict_op
