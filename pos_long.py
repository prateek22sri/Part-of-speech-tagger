"""
Long way of POS Tagging using Viterbi Algorithm
This method is good to understand how the transition probabilities, emission probabilities and prior probabilities work

Author: Prateek Srivastava
Date: October 5,2017

References : CSCI B659 - Damir Cavar
"""

from nltk.corpus import brown
from collections import Counter
from collections import defaultdict
from nltk.util import ngrams
from math import log


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

        """
        add the "_es_" end of sentence tag to each tag as the transition
        """
        # for sent in brown.tagged_sents():
        #     self.tagTags['_BS_'][sent[0][1]] += 1
        # print(self.tagTags['_BS_'])

    def viterbi(self, text):
        textList = text.split()
        globalDict = {}
        seq = []
        for t in range(0, len(textList)):
            tmp = {}
            if (t == 0):
                for probableTag in self.tokenTags[textList[t]]:
                    priorProbability = log((self.tagCounter[probableTag] / len(self.tokens)), 2)
                    transitionProbability = log((self.tagTags['_BS_'][probableTag] / len(self.tagSents)), 2)
                    emissionProbability = log(
                        (self.tokenTags[textList[t]][probableTag] / self.tokenCounter[textList[t]]), 2)
                    tmp[probableTag] = priorProbability + transitionProbability + emissionProbability
                globalDict[textList[t]] = tmp
                minimum = min(globalDict[textList[t]], key=globalDict[textList[t]].get)
                # print(minimum,globalDict[textList[t]][minimum])
                seq.append(minimum)
            else:
                for probableTag in self.tokenTags[textList[t]]:
                    priorProbability = min(
                        [globalDict[textList[t - 1]][previousTag] for previousTag in globalDict[textList[t - 1]]])
                    emissionProbability = log(
                        (self.tokenTags[textList[t]][probableTag] / self.tokenCounter[textList[t]]), 2)
                    # print(emissionProbability)
                    for previousTag in globalDict[textList[t - 1]]:
                        if self.tagTags[previousTag][probableTag] != 0:
                            transitionProbability = log((self.tagTags[previousTag][probableTag] / len(self.tokens)), 2)
                            tmp[probableTag] = priorProbability + transitionProbability + emissionProbability
                globalDict[textList[t]] = tmp
                minimum = min(globalDict[textList[t]], key=globalDict[textList[t]].get)
                seq.append(minimum)
        print(seq)

    def forwardBackward(self, text):
        pass

    def findPOS(self, text, kind):
        if kind == 'viterbi':
            self.viterbi(text)
        elif kind == 'forward-backward':
            self.forwardBackward(text)


if __name__ == '__main__':
    p = posCorpus()
    text = "Time flies like an arrow ."
    p.findPOS(text, 'viterbi')
