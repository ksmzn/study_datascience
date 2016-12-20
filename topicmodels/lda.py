#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Variational Bayes Zero

import time
import numpy as np
from scipy.special import psi

from .utils import Perplexity, output_word_topic_dist

class LDA:
    """
    Latent Dirichlet allocation, using collapsed Variational Bayesian Zero.

    Parameters
    --------------------
    w_di_li : array-like, shape = (docs, words_per_doc)
        The document-term matrix.

    calc_perplexity : Boolean
        If set to True, split the dateset and calculdate perplexity.

    K : int
        num of topics.

    init_alpha : float
        initial Dirichlet parameter.

    init_beta : float
        initial Dirichlet parameter.

    smartinit : Boolean
        If set to True, use Gibbs initialization.

    seed : int
        random seed.
    """
    def __init__(self, w_di_li, calc_perplexity=True, K=10, init_alpha=0.1, init_beta=0.01, smartinit=True, dictionary=None, seed=1024):
        self.V = np.hstack(w_di_li).max() + 1
        if calc_perplexity:
            self.w_di_li, self.w_di_li_test = self._split_docs(w_di_li, 0.9)
        else:
            self.w_di_li = w_di_li
            self.w_di_li_test = None
        self.K = K
        self.D = len(w_di_li)
        self.n_d_li = np.array([len(w_d) for w_d in self.w_di_li])
        self.alpha = np.zeros(K) + init_alpha
        self.beta = np.zeros(self.V) + init_beta
        np.random.seed(seed)
        self.q_z_di_li = self._initialize_q(smartinit)
        self.E_n_d_k_li = self._calc_E_n_d_k()
        self.E_n_k_v_li = self._calc_E_n_k_v()
        self.dictionary = None

    def _split_docs(self, docs, train_rate=.9):
        train_data_length = [int(len(d) * train_rate) for d in docs]
        split_docs = [(d[:i], d[i:]) for d, i in zip(docs, train_data_length)]
        w_di_li, w_di_li_test = zip(*split_docs)
        return w_di_li, w_di_li_test

    def _initialize_q(self, smartinit):
        if smartinit:
            q_z_di_li = self._smartinit()
        else:
            q_z_di_li = np.array([np.array([np.random.dirichlet(self.alpha) for n in range(n_d)]) for n_d in self.n_d_li])
        return q_z_di_li

    def _smartinit(self):
        self.E_n_d_k_li = np.zeros((self.D, self.K))# + self.alpha
        self.E_n_k_v_li = np.zeros((self.K, self.V))# + self.beta
        q_z_di_li = np.array([np.array([self._sampling_q(w, d) for w in doc]) for d, doc in enumerate(self.w_di_li)])
        return np.array(q_z_di_li)

    def _sampling_q(self, w, d):
        E_n_k_v_li = self.E_n_k_v_li[:, w]
        E_n_d_k_li = self.E_n_d_k_li[d]
        p_k = (E_n_k_v_li + self.beta[w]) * (E_n_d_k_li + self.alpha) / (E_n_k_v_li.sum(axis=0) + self.beta.sum())
        q_z = np.random.mtrand.dirichlet(p_k / p_k.sum() * self.alpha)
        E_n_d_k_li += q_z
        E_n_k_v_li += q_z
        return q_z

    def _calc_E_n_d_k(self):
        return np.array([q_z_di.sum(axis=0) for q_z_di in self.q_z_di_li])

    def __E_n_k_v(self, v):
        E_n_k_v_per_doc = [q_z_di[w_di==v].sum(axis=0) for q_z_di, w_di in zip(self.q_z_di_li, self.w_di_li) if (w_di==v).any()]
        if E_n_k_v_per_doc == []:
            E_n_k_v_per_doc = [np.zeros(self.K)]
        E_n_k_v = np.array(E_n_k_v_per_doc).sum(axis=0)
        return E_n_k_v

    def _calc_E_n_k_v(self):
        return np.array([self.__E_n_k_v(v) for v in range(self.V)]).T

    def _get_alpha(self):
        numer = (psi(self.E_n_d_k_li + self.alpha) - psi(self.alpha)).sum(axis=0) * self.alpha
        denom = (psi((self.E_n_d_k_li + self.alpha).sum(axis=1)) - psi(self.alpha.sum())).sum()
        #denom = psi((self.E_n_d_k_li + self.alpha).sum(axis=1)).sum() - psi(self.alpha.sum())
        #print("numer: ", numer)
        #print("denom: ", denom)
        return numer / denom

    def update_alpha(self, iter_maxnum=1000, debug=False):
        for i in range(iter_maxnum):
            old_alpha = self.alpha
            if debug and (iter_maxnum <=200 or i%10==0):
                debug_print = True
            else:
                debug_print = False

            if debug_print:
                print("==========Alpha Iter: {}==========".format(i))
            self.alpha = self._get_alpha()
            if debug_print:
                print("alpha: ", self.alpha)
            diff = np.abs((self.alpha - old_alpha) / old_alpha).sum()
            if debug_print:
                print("diff: ", round(diff, 6))
            #if diff <= 1e-15:
            if diff <= 1e-08:
                print("alpha: convergence! Alpha_Iter: ", i)
                break
        else:
            print("WARNING: alpha do not converge. Iter: ", i)

    def _get_beta_v(self, debug=False):
        beta_v = self.beta[0]
        V = self.V
        numer = (psi(self.E_n_k_v_li + beta_v) - psi(beta_v)).sum() * beta_v
        denom = (psi((self.E_n_k_v_li + beta_v).sum(axis=1)) - psi(beta_v * V)).sum() * V
        return numer / denom

    def update_beta(self, iter_maxnum=100, debug=False):
        for i in range(iter_maxnum):
            if i==0:
                beta_v = self.beta[0]
            if debug and (iter_maxnum <=200 or i%10==0):
                debug_print = True
            else:
                debug_print = False
            old_beta_v = beta_v
            if debug_print:
                print("==========beta==========")
            beta_v = self._get_beta_v(debug_print)
            if debug_print:
                print("beta: ", beta_v)
            self.beta = np.zeros(len(self.beta)) + beta_v

            diff = np.abs(beta_v - old_beta_v)
            if debug:
                print("Diff: ", diff)
            if diff <= 1e-08:
                print("beta: convergence! Beta_Iter: ", i)
                break
        else:
            print("WARNING: beta do not converge. Iter: ", i)

    def inference(self, debug=False):
        self.diff = 0
        Enkv_beta_sum = (self.E_n_k_v_li.sum(axis=1) + self.beta.sum())
        for d, doc in enumerate(self.w_di_li):
            per_d_start = time.time()
            q_z_di = self.q_z_di_li[d]
            for i, w in enumerate(doc):
                E_n_k_v_li = self.E_n_k_v_li[:, w]
                E_n_d_k_li = self.E_n_d_k_li[d] 

                # compute q(z_di)
                old_q = q_z_di[i]
                q = (E_n_k_v_li - old_q + self.beta[w]) * (E_n_d_k_li - old_q + self.alpha) / (Enkv_beta_sum - old_q)
                q /= q.sum()

                # update E(n_dk), E(n_kv)
                q_diff = q - old_q
                E_n_d_k_li += q_diff
                E_n_k_v_li += q_diff
                Enkv_beta_sum += q_diff

                # update q(z_di)
                q_z_di[i] = q

                self.diff += np.abs(q_diff).sum()

            per_d_end = time.time()
            if debug:
                print("Per D:", d, ", Time: ", per_d_end - per_d_start)

        # update params
        iter_maxnum = 2000
        update_alpha_start = time.time()
        self.update_alpha(iter_maxnum, debug)
        update_alpha_end = time.time()
        if debug:
            print("Alpha: Time: ", update_alpha_end - update_alpha_start)
        update_beta_start = time.time()
        self.update_beta(iter_maxnum, debug)
        update_beta_end = time.time()
        if debug:
            print("Beta: Time: ", update_beta_end - update_beta_start)

    def _calc_q_diff(self, new_q, old_q):
        return sum(sum(map(sum, np.abs(new_q, old_q))))

    def train(self, iter_num=1000,  debug=False):
        self.beta_li = []
        self.perps = []
        print_num = 2
        for i in range(iter_num):
            print("*"*100)
            print("Iter: ", i)
            start = time.time()
            old_q_z_di_li = self.q_z_di_li.copy()

            self.inference(debug=debug)

            self.beta_li.append(self.beta[0])
            diff = self.diff.sum()
            print("Diff q(z): ", diff)

            if self.w_di_li_test is not None:
                px = Perplexity(self, self.w_di_li_test)
                perplexity = px.calc_perplexity()
                print("Perplexity: ", perplexity)

                # Peplが下がって上昇したときの単語分布を見るため
                # パッケージ化の際には消す
                if self.dictionary and self.perps and (perplexity > self.perps[-1]) and print_num >= 0:
                    phi = self._calc_E_n_k_v()
                    phi /= phi.sum()
                    output_word_topic_dist(phi, self.K, dictionary)
                    print_num -= 1
                self.perps.append(perplexity)
            if diff <= 1e-25:
                print("q(z): convergence! Iter: ", i)
                break

            end = time.time()
            print("Time: ", end - start)
        else:
            print("WARNING: q(z) do not converge. Iter: ", i)
