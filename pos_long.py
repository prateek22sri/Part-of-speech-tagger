"""
Long way of POS Tagging using Viterbi Algorithm
This method is good to understand how the transition probabilities, emission probabilities and prior probabilities work

Author: Prateek Srivastava
Date: October 5,2017

References : Indiana University CSCI B659 - by Damir Cavar
"""

from nltk.corpus import brown
from collections import Counter
from collections import defaultdict
from nltk.util import ngrams
from math import log
from copy import deepcopy


class posCorpus:
    def __init__(self):
        """
        token  = list of all the tokens
        tag = list of all the tags
        """
        self.tokens, self.tags = zip(*brown.tagged_words())
        # print(tokens)
        # print(tags)

        """ tagCounter has count of all the tags used for eg. {'NN': 152470, 'IN': 120557, .. }"""
        self.tagCounter = Counter(self.tags)
        # print(tagCounter)

        """ tokenCounter has count of all the tokens used for eg. {'the': 62713, 'of': 36080, 'and': 27915 .. } """
        self.tokenCounter = Counter(self.tokens)
        # print(tokenCounter)

        """ 
        tokenTags is a dict with emission counts for 
        eg. {'laendler': Counter({'FW-NN': 1}), 'Viennese': Counter({'JJ': 1}) ..} 
        """
        self.tokenTags = defaultdict(Counter)
        for token, tag in brown.tagged_words():
            self.tokenTags[token][tag] += 1
        # print(tokenTags)

        """ 
        tagTags is a dict with transition counts 
        for eg. { 'AT': Counter({'NN': 48376, 'JJ': 19488, 'NNS': 7215, 'AP': 3007, ..}
        """
        self.tagTags = defaultdict(Counter)
        posBigrams = ngrams(self.tags, 2)
        for tag1, tag2 in posBigrams:
            self.tagTags[tag1][tag2] += 1
            # print(tagTags)

        """
        add the "_bs_" begining of sentence tag to each tag as the transition
        """
        self.tagSents = brown.tagged_sents()
        for sent in self.tagSents:
            self.tagTags['_BS_'][sent[0][1]] += 1
            # print(self.tagTags['_BS_'])

    def viterbi(self, text):
        textList = text.split()
        globalDict = {}
        seq = []
        newfringe = []
        for t in range(0, len(textList)):
            fringe = deepcopy(newfringe)
            newfringe = []

            if (t == 0):
                init_tmp = {}
                for probableTag in self.tokenTags[textList[t]]:
                    priorProbability = log((self.tagCounter[probableTag] / len(self.tokens)), 2)
                    transitionProbability = log((self.tagTags['_BS_'][probableTag] / len(self.tagSents)), 2)
                    emissionProbability = log(
                        (self.tokenTags[textList[t]][probableTag] / self.tokenCounter[textList[t]]), 2)
                    init_tmp[probableTag] = priorProbability + transitionProbability + emissionProbability
                globalDict[textList[t]] = init_tmp
                for k, v in globalDict[textList[t]].items():
                    newfringe.append([[k], v])
            else:
                while len(fringe) != 0:
                    s = fringe.pop(0)
                    tmp = {}
                    for probableTag in self.tokenTags[textList[t]]:
                        priorProbability = s[1]
                        emissionProbability = log(
                            (self.tokenTags[textList[t]][probableTag] / self.tokenCounter[textList[t]]), 2)
                        if s[0][len(s[0]) - 1] in self.tagTags:
                            if probableTag in self.tagTags[s[0][len(s[0]) - 1]]:
                                if self.tagTags[s[0][len(s[0]) - 1]][probableTag] != 0:
                                    transitionProbability = log(
                                        (self.tagTags[s[0][len(s[0]) - 1]][probableTag] / len(self.tokens)), 2)
                                    tmp[probableTag] = priorProbability + transitionProbability + emissionProbability
                    for k, v in tmp.items():
                        newfringe.append([s[0] + [k], v])
                        if t == len(textList) - 1:
                            seq.append([s[0] + [k], v])

        min = seq[0][1]
        r = []
        for s in range(0, len(seq)):
            if seq[s][1] < min:
                min = seq[s][1]
                r = seq[s][0]
        return r

    def findPOS(self, text, kind):
        if kind == 'viterbi':
            seq = self.viterbi(text)
            print(list(zip(text.split(), seq)))
        else:
            print("Error")
            exit(1)


if __name__ == '__main__':
    p = posCorpus()
    text = "Time flies like an arrow ."
    p.findPOS(text, 'viterbi')
