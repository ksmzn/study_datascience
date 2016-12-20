#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Hierarchical Dirichlet Process + collapsed Variational Bayes Zero
# Like scikit-learn

import numpy as np
from scipy.special import psi
from collections import Counter

class PCVB0:
        np.random.seed(seed)
        self.T = T
        self.alpha = init_alpha
        self.beta = init_beta
        self.gamma = init_gamma
        self.e_pi = np.ones(T) / T
        self.evaluate = evaluate
        self.max_iter = max_iter
        self.max_iter_fpi = max_iter_fpi

    def _split_docs(self, docs, train_rate=.9):
        """ split dataset for calculating perplexity"""
        train_data_length = [int(len(d) * train_rate) for d in docs]
        split_docs = [(d[:i], d[i:]) for d, i in zip(docs, train_data_length)]
        w_di_li, w_di_li_test = zip(*split_docs)
        return np.array(w_di_li), np.array(w_di_li_test)

    def initial_q_z(self, size):
        q_z = np.random.uniform(0, 1, size=size) + 0.1
        #q_z = np.random.uniform(0, 1, size=size) + 10.0 # より一様にしてみる
        q_z = q_z / q_z.sum()
        return q_z

    def update_q_z(self, d, i, w, verbose=False):
        old_q = self.q_z_d[d][i]

        if verbose:
            print("Old_q:", old_q)

        n_dk_nodi = self.n_dk[d] - old_q
        n_kw_nodi = self.n_kw[:, w] - old_q
        n_k_nodi = self.n_kw.sum(axis=1) - old_q

        q = (n_dk_nodi + self.alpha * self.e_pi) * (n_kw_nodi + self.beta * self.tau[w]) / (n_k_nodi + self.beta)
        if verbose:
            self.tmp = q.copy()
            print("new_q:", q)
        q /= q.sum()

        diff = q - old_q
        self.q_z_d[d][i] = q
        self.n_dk[d] += diff
        self.n_kw[:, w] += diff
        self.q_diff += (np.abs(diff / old_q).sum())
        if verbose:
            print("Updated!!!")


    def _k_assign_over1time(self, q_z_d):
        return (1 - (1 - q_z_d).prod(axis=0))

    def update_endk_over1_sum(self):
        self.endk_over1_sum = np.array([self._k_assign_over1time(q_z_d) for q_z_d in self.q_z_d]).sum(axis=0)
        return self.endk_over1_sum

    def _qumsum(self, li):
        return np.array([li[i+1:].sum() for i, _ in enumerate(li)]) 

    def update_a(self):
        self.a = 1 + self.endk_over1_sum
        return self.a

    def update_b(self):
        self.b = self.gamma + self._qumsum(self.endk_over1_sum)
        return self.b

    def update_pi(self):
        pi_bar = self.a / (self.a + self.b)
        self.e_pi = np.array([pi_bar_k * (1-pi_bar[:k]).prod() for k, pi_bar_k in enumerate(pi_bar)])
        return self.e_pi

    def _update_n_dk(self):
        self.n_dk = np.array([q_z_d.sum(axis=0) for q_z_d in self.q_z_d])
        return self.n_dk

    def _num_topic_assigned(self, v):
        per_doc = np.array([q_z_di[w_di == v].sum(axis=0) for w_di, q_z_di in zip(self.w_di_li, self.q_z_d)])
        return per_doc

    def _update_n_kw(self):
        self.n_kw = np.array([self._num_topic_assigned(v).sum(axis=0) for v in range(self.V)]).T
        return self.n_kw

    def _v_assign_over1time(self, v):
        return 1 - np.array([(1 - q_z[w_d==v]).prod(axis=0) for w_d, q_z in zip(self.w_di_li, self.q_z_d)]).prod(axis=0)

    def update_enkw_over1_sum(self):
        self.enkw_over1_sum = np.array([self._v_assign_over1time(v).sum() for v in range(self.V)])
        return self.enkw_over1_sum

    def update_n_kw(self, d, i, w):
        self.n_kw *= (1 - self.rho_c)
        self.n_kw[:, w] += self.rho_c * self.N * self.q_z_d[d][i]
        return self.n_kw

    def update_alpha(self, max_iter=100):
        new_alpha = self.alpha
        numer = self.endk_over1_sum.sum()
        for i in range(max_iter):
            old_alpha = new_alpha
            new_alpha = numer / (psi(self.n_d_li + old_alpha) - psi(old_alpha)).sum()
            diff = np.abs((new_alpha - old_alpha) / old_alpha)
            if diff <= 1e-08:
                print("Alpha: convergence! Iter: ", i)
                self.alpha = new_alpha
                break
        else:
            print("WARNING: alpha do not converge. Iter: ", i)
            new_alpha = self.alpha # don't update alpha value
        return new_alpha

    def update_gamma(self):
        self.gamma = (self.T - 1) / (psi(self.a[:-1] + self.b[:-1]) - psi(self.b[:-1])).sum()
        return self.gamma

    def update_tau(self):
        self.tau = self.enkw_over1_sum + 1e-100 # to avoid dividing by zero
        self.tau /= self.enkw_over1_sum.sum()
        return self.tau

    def update_beta(self, max_iter=100):
        new_beta = self.beta
        self.update_enkw_over1_sum()
        numer = self.enkw_over1_sum.sum()

        for i in range(max_iter):
            old_beta = new_beta
            denom = (psi(self.n_kw.sum(axis=1) + old_beta) - psi(old_beta)).sum()
            new_beta = numer / denom
            diff = np.abs((new_beta - old_beta) / old_beta)
            if diff <= 1e-08:
                print("Beta: convergence! Iter: ", i)
                self.beta = new_beta
                break
        else:
            print("WARNING: beta do not converge. Iter: ", i)
            new_beta = self.beta # don't update beta value

        return new_beta

    def inference(self, max_iter_fpi):
        self.q_diff = 0
        for d in range(self.D):
            for i, w in enumerate(self.w_di_li[d]):
                self.update_q_z(d, i, w)

        # update pi, alpha, gamma
        self.update_endk_over1_sum()
        self.update_a()
        self.update_b()
        self.update_pi()
        self.update_alpha(max_iter_fpi)
        self.update_gamma()

        # update beta(, tau)
        self.update_beta(max_iter_fpi)

        # if beta is symmetry, should not update tau?
        self.update_tau()

    def learning(self, max_iter=100, max_iter_fpi=100):
        c = Counter()
        c.update([x.argmax() for y in self.q_z_d for x in y])
        print(c)
        print("Initial Alpha", self.alpha)
        print("Initial Gamma", self.gamma)
        print("Initial Beta", self.beta)
        print("Initial E[alpha*pi]", (self.alpha*self.e_pi).mean())
        print("Initial E[beta*tau]", (self.beta*self.tau).mean())
        self.perplexity = []
        self.q_diff_list = []
        append = self.perplexity.append
        q_append = self.q_diff_list.append
        for j in range(max_iter):
            print("="*100)
            print("Iter: ", j)
            self.inference(max_iter_fpi)
            c = Counter()
            c.update([x.argmax() for y in self.q_z_d for x in y])
            print(c)
            print("Alpha", self.alpha)
            print("Gamma", self.gamma)
            print("Beta", self.beta)
            print("E[alpha*pi]", (self.alpha*self.e_pi).mean())
            print("E[beta*tau]", (self.beta*self.tau).mean())
            q_append(self.q_diff)
            print("q(z) diff-rate:", self.q_diff)
            if self.w_di_li_test is not None:
                perplexity = self.calc_perplexity(self.w_di_li_test)
                append(perplexity)
                print("Perplexity: ", perplexity)

    def fit(self, X, y=None):
        # Setting
        print("Setting...")
        self.V = int(np.hstack(X).max() + 1)
        self.tau = np.ones(self.V) / self.T
        if self.evaluate:
            self.w_di_li, self.w_di_li_test = self._split_docs(X, 0.9)
        else:
            self.w_di_li = X
            self.w_di_li_test = None
        self.D = len(X)
        self.n_d_li = np.array([len(w_d) for w_d in self.w_di_li])
        self.q_z_d = np.array([np.array([self.initial_q_z(self.T) for _ in w_di]) for w_di in self.w_di_li])

        self._update_n_dk()
        self._update_n_kw()

        self.update_endk_over1_sum()
        self.update_enkw_over1_sum()
        self.update_a()
        self.update_b()

        # learning
        print("Start learning...")
        self.learning(self.max_iter, self.max_iter_fpi)


    def predict_word_probability(self, d, v):
        expected_theta = (self.n_dk[d] + self.e_pi * self.alpha) / (self.n_d_li[d] + self.alpha)
        expected_phi = (self.n_kw[:, v] + self.beta * self.tau[v]) / (self.n_kw.sum(axis=1) + (self.beta))
#         print("D: ", d, ", v: ", v, ", p=", (expected_theta * expected_phi).sum())
        return (expected_theta * expected_phi).sum()

    def log_likelihood_p(self, w_di_li_test):
        p_w_di = [self.predict_word_probability(d, v) for d, word_list in enumerate(w_di_li_test) for v in word_list]
        return np.log(p_w_di).sum()

    def calc_perplexity(self, w_di_li_test):
        logl = self.log_likelihood_p(w_di_li_test) 
        n_d_test = sum(map(len, w_di_li_test))
        perplexity = np.exp(- logl / n_d_test)
        return perplexity
