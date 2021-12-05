from os import pardir
import torch

from itertools import count
from queue import PriorityQueue
from collections import defaultdict

import pdb


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """
    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node):
        """ Adds a new beam search node to the queue of current nodes """
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        # ensure all node paths have the same length for batch ops
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        self.final.put((score, next(self._counter), node))

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        node = merged.get()
        node = (node[0], node[2])

        return node

    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        pdb.set_trace()
        nodes = PriorityQueue()
        # Keep track of how many search paths are already finished (EOS)
        finished = self.final.qsize()
        for _ in range(self.beam_size-finished):
            node = self.nodes.get()
            nodes.put(node)
        self.nodes = nodes


class DiverseBeamSearch(BeamSearch):

    def __init__(self, beam_size, max_len, pad, gamma, n=3):
        BeamSearch.__init__(self, beam_size, max_len, pad)
        self.n_best = n
        self.gamma = gamma

    def get_best_n(self):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        nodes = []
        # pdb.set_trace()
        for i in range(self.n_best):
            node = merged.get()
            node = node[2]
            nodes.append(node)

        return nodes
    
    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        diversified_nodes = PriorityQueue()
        common_parent = {}
        # Keep track of how many search paths are already finished (EOS)
        for _ in range(self.nodes.qsize()):
            # change tuple to list so that probability can be changed
            # node: [6.134, 75, <seq2seq.beam.BeamSearchNode object at 0x7f743075f910>]
            node = list(self.nodes.get())
            parent = int(node[2].sequence[-2])
            # current rank of sibling
            try:
                sibling_rank = common_parent[parent]
            except KeyError:
                common_parent[parent] = 1
                sibling_rank = 1
            prob = node[0]
            new_prob = prob + sibling_rank*self.gamma
            node[0] = new_prob
            diversified_nodes.put(tuple(node))
            
            # increase sibling counter
            common_parent[parent] += 1
        
        nodes = PriorityQueue()
        finished = self.final.qsize()

        # Keep track of how many search paths are already finished (EOS)
        for _ in range(self.beam_size-finished):
            node = diversified_nodes.get()
            nodes.put(node)
        
        self.nodes = nodes
        

class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):

        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search

    def eval(self, alpha=0.0):
        """ Returns score of sequence up to this node 

        params: 
            :alpha float (default=0.0): hyperparameter for
            length normalization described in in
            https://arxiv.org/pdf/1609.08144.pdf (equation
            14 as lp), default setting of 0.0 has no effect
        
        """
        normalizer = (5 + self.length)**alpha / (5 + 1)**alpha
        return self.logp / normalizer
        