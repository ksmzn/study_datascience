#!/usr/bin/env python
# -*- coding: utf-8 -*-

# utility scripts.

import numpy as np

def output_word_topic_dist(phi, K, dictionary):
    for k in range(K):
        phi_k = phi[k].sum()
        print("\n-- topic: %d" % k)
        for w in np.argsort(-phi[k])[:20]:
            print("%s: %f" % (dictionary[w], phi[k,w]/phi_k))

class Perplexity:
    def __init__(self, model, w_di_li_test):
        self.w_di_li_test = w_di_li_test
        self.n_d_test = [len(w_di) for w_di in w_di_li_test]
        self.model = model
        self.expected_phi = self._calc_expected_phi()
        self.expected_theta = self._calc_expected_theta()

    def calc_perplexity(self):
        logl = self._log_likelihood()
        perplexity = np.exp(- logl / sum(self.n_d_test))
        return perplexity

    def _calc_expected_phi(self):
        numer = self.model.E_n_k_v_li + self.model.beta
        denom = numer.sum(axis=1)
        return (numer.T / denom).T

    def _calc_expected_theta(self):
        numer = self.model.E_n_d_k_li + self.model.alpha
        denom = numer.sum(axis=1)
        return (numer.T / denom).T

    def _log_likelihood(self):
        p_w_di = np.array([self._log_p_w_di(d, w_di)  for d, w_di in enumerate(self.w_di_li_test)])
        return p_w_di.sum()

    def _log_p_w_di(self, d, w_di):
        log_p_li = [np.log(self._predict_word_probability(d, w)) for w in w_di]
        return sum(log_p_li)

    def _predict_word_probability(self, d, w):
        return (self.expected_phi[:, w] * self.expected_theta[d]).sum()
