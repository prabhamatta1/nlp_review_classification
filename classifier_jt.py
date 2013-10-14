#!/usr/bin/python

__author__ = 'JT Huang'
__email__ = 'jthuang@ischool.berkeley.edu'
__python_version = '2.7.3'

import re
import nltk
from nltk.corpus import stopwords

# file path
trainfile = 'trainingfile.txt'
heldout = 'heldoutfile.txt'

feature_word_cnt = 500
en_stop_words = stopwords.words('english')
stemmer = nltk.PorterStemmer()


# get score from attribute labels
def get_score_from_attr(attr_labels):
  score_pattern = '\[(.*?)\]'
  scores = re.findall(score_pattern, attr_labels)

  review_score = []
  sum_score = 0;
  for s in scores:
    try:
      s = int(s)
      if s > 0:
        s = 1
      elif s < 0:
        s = -1
      else:
        s = 0
    except ValueError:
      s = 0
    sum_score += s

  if sum_score > 0:
    return 1
  elif sum_score < 0:
    return (-1)
  else:
    return 0


# load the training/heldout file and extract the text
def load_review_from_file(fname):
  reviews = []
  
  with open(fname, 'r') as f:
    # read in lines
    flines = f.read().splitlines()
    for lines in flines:
      line = lines.rstrip('\r\n').split('\t')
      # split each line into 'product', 'score', 'text'
      reviews.append((line[0], get_score_from_attr(line[1]), line[2].strip()))
    f.close()

  return reviews


# get feature words (can only choose either to get stem or to get tag. if both, only get stem)
def get_feature_words(reviews, get_all=False, get_stem=False, get_tag=False, flt_stop_words=True):
  # get positive and negative review words
  pos_words = []
  neg_words = []
  neu_words = []
  all_words = []

  for review in reviews:
    score = review[1]
    words = nltk.word_tokenize(review[2])

    if get_all:
      target_words = all_words
    else:
      if score > 0:
        target_words = pos_words
      elif score < 0:
        target_words = neg_words
      else:
        target_words = neu_words

    if get_stem:
      for word in words:
        target_words.append(stemmer.stem(word))
    elif get_tag:
      for word, tag in nltk.pos_tag(words):
        if tag.startswith('V') or tag == 'ADJ' or tag == 'ADV':
          target_words.append(word)
    else:
      target_words.extend(words)

  if flt_stop_words:
    stop_words = en_stop_words
  else:
    stop_words = []

  if get_all:
    return nltk.FreqDist(w.lower() for w in all_words if w.isalpha() and w not in stop_words).keys()[:feature_word_cnt]
  else:
    pos_feature_words = nltk.FreqDist(w.lower() for w in pos_words if w.isalpha() and w not in stop_words).keys()[:feature_word_cnt]
    neg_feature_words = nltk.FreqDist(w.lower() for w in neg_words if w.isalpha() and w not in stop_words).keys()[:feature_word_cnt]
    neu_feature_words = nltk.FreqDist(w.lower() for w in neu_words if w.isalpha() and w not in stop_words).keys()[:feature_word_cnt]
    return list(set(pos_feature_words + neg_feature_words + neu_feature_words))


# extract unigram feature
def extract_unigram_feature(reviews, get_all=False, get_stem=False, get_tag=False, flt_stop_words=True):
  feature_set = []

  # get feature words from review texts
  feature_words = get_feature_words(reviews, get_all, get_stem, get_tag, flt_stop_words)

  # get features
  for review in reviews:
    score = review[1]
    text = review[2]
    features = {}
    word_list = nltk.word_tokenize(text)
    # contain the feature words or not
    for feature_word in feature_words:
      features['contains(%s)' % feature_word] = feature_word in word_list

    feature_set.append((features, score))

  return feature_set



if __name__=='__main__':
    # read the train and heldout file
    train_reviews = load_review_from_file(trainfile)
    heldout_reviews = load_review_from_file(heldout)

    # train the classifier
    train_set = extract_unigram_feature(train_reviews)
    model = nltk.NaiveBayesClassifier.train(train_set)

    # test the model on the heldout file
    heldout_set = extract_unigram_feature(heldout_reviews)
    print "Accuracy: ", nltk.classify.accuracy(model, heldout_set)
    print model.show_most_informative_features(5)

    # train the classifier
    train_set = extract_unigram_feature(train_reviews, get_all=True)
    model = nltk.NaiveBayesClassifier.train(train_set)
    # test the model on the heldout file
    heldout_set = extract_unigram_feature(heldout_reviews, get_all=True)
    print "Accuracy: ", nltk.classify.accuracy(model, heldout_set)
    print model.show_most_informative_features(5)

##############################################################################
# (i) Each individual's code should be in the form of functions that
# produce output in this agreed-upon format
##############################################################################
# ANS (i):
# The function 'extract_unigram_feature()' is the main feature extraction function
# which will match most frequent words in the training set,
# and it will produce the output in our agreed-upon format described in (ii).
# The function also need to call 'get_feature_words()' to get most frequent words.


##############################################################################
# (ii) to get full credit the code must state what this format is in their documentation.
##############################################################################
# ANS (ii):
# Our agreed-upon return format is basically the same as the format of
# the argument 'labeled_featuresets' of 'nltk.NaiveBayesClassifier.train()'.
# The produced output, a feature set, is in the format of list of tuples.
# Each tuple consists of a feature dictionary and a score of a sentence.
# In the feature dictionary, it keeps the keys of frequent unigrams, and the values of
# the presence of those keys in the sentence.
#
# Ex: [({'frequent_word1': True, 'frequent_word2': False, ...}, 1),
#      ({'frequent_word1': False, 'frequent_word2': False, ...}, 0),
#      ({'frequent_word1': True, 'frequent_word2': True, ...}, -1),
#      ......
#     ]


##############################################################################
# (iii) You must show for each feature that you tried to optimize it;
# for example, if you used unigrams, you tested for the effects of using stopwords and/or stemming,
##############################################################################
# ANS (iii):
# I have tried several ways to improve the accuracy:
# a. filter stopwords
# b. get stem of the words
# c. get only special part-of-speech words (verbs, adjectives, adverbs)
# d. get most frequent words from positive, negative and neutral reviews, respectively.
#    and merge into a new fequent-word list (try to get more distince postive words and negative words)
#
# And the best combination result for this training data and held-out data is:
# 'a. filter stopwords' and 'd. get most frequent words from postive, negative, and neutral reviews respectively'
# It got the accuracy of 0.731213872832
#
# Note: Maybe the result that stemming and POS tags did not help to improve the accuracy a lot
# is because I only use simple stemmer nltk.PorterStemmer() and simple tagger nltk.pos_tag() 


##############################################################################
# Each individual feature must be tried out on the classification task
# to see how well it performs on the training data and on the held-out data.
# (iv) These results must be reported and included in the writeup.
# It is expected that each person individually is able to write code
# to run and test the classifier in this manner on the features they produce.
##############################################################################

# ANS (iv):
# I have tried the feature on the classification task using 'nltk.NaiveBayesClassifier'
# and the result is:
# 
# Accuracy: 0.731213872832
# Most Informative Features
#        contains(suggest) = True                0 : 1      =     34.8 : 1.0
#           contains(hype) = True                0 : 1      =     26.4 : 1.0
#           contains(mind) = True                0 : 1      =     26.4 : 1.0
#          contains(bulky) = True                0 : 1      =     26.4 : 1.0
#        contains(desktop) = True                0 : 1      =     26.4 : 1.0
