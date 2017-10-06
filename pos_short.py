"""
Short way of POS Tagging using Hidden Markov Model Trainer by nltk using Brown Corpus

Author : Prateek Srivastava
Date : October 5,2017
"""

from nltk.corpus import brown
from nltk.tag import hmm
train_data = brown.tagged_sents()
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)
print(tagger.tag("Time flies like an arrow .".split()))
