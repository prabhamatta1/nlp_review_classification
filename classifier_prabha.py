__author__ = "Prabha Matta"

import re
import nltk
from nltk.classify.scikitlearn import SklearnClassifier 
from sklearn.svm import LinearSVC #I used scikit-learn here to compare the performance between nltk's Naive Bayes and SVM
pattern='\[(.*?)\]'
import string

"""
Parser Plus Cleanup:
=====================
parser.py cleans up the individual data files and creates a single file "trainingfile.txt" and "heldoutfile.txt" which are read by classifier_prabha.py. These files, "trainingfile.txt", are of the format:

[ipod]	features[-2]size[+1]	 There isn't much features on the iPod at all, except games. Size is small and good.

Syntax: python parser.py "<all file names separated by commas>"

eg: python parser.py "training/Canon PowerShot SD500.txt,training/Hitachi router.txt,training/Nokia 6600.txt,training/Canon S100.txt,training/Linksys Router.txt,training/ipod.txt,training/Diaper Champ.txt,training/MicroMP3.txt,training/norton.txt"


How to come up with Feature Sets:
===========================
Some of my feature sets are based on pure glancing at the data. For eg: reviews containing -ve conjunctions have -ve labels, reviews containing negations also have -ve labels and  reviews containing positive emotions have +ve labels. These observations are in conjunction with Cheng et.al research paper: "Gender identification from e-mails."


Some of the followings feature sets I used from Cheng's paper :
-----------------------------------------------------------

Psycho-linguistic features:
===========================
Negations--> no, not, never
Positive emotion --->	 love, nice, sweet
positives -->	awesome, good, great, yes
Negative emotion  -->	hurt, ugly, nasty
Anxiety---> worried, fearful, nervous
Anger--> hate, kill, annoyed
Sadness--> crying, grief, sad



Syntactic features: 
=================
number of exclamations, punctuations,etc

Note:
Initially, I used nltk.PorterStemmer() to verify if any stemmed words of relevance are present or not. However, this made the code run very slow...therefore I added the stemmed words in the comparison list directly

Both Naive Bayes Classifier and SVM(support Vector Machine) are used in the code to train the classifer based on the feature sets. Finally, Accuracy based on both SVM model and Naive Bayes Model are shown in the output

"""



#lambda function to count multiple char matching --> used to find how many punctuation marks are present
count_multiple_char_matching = lambda text, stringsToBeMatched: len(list(filter(lambda c: c in stringsToBeMatched, text)))

#Functions to extract features
def has_negativeconjunctions(review):
    """ reviews containing -ve conjunctions have -ve labels"""    
    negativeconjunctions=set(['but', 'however', 'though', 'although', 'until' , 'unless'])
    num = 0
    text = review.translate(None, string.punctuation) #clean punctuations from review
    num = len(set(text.split()).intersection(negativeconjunctions))
    if num > 0:
	num =1
    return 'has_negativeconjunctions',num

def num_negations(review):
    """ reviews containing negations have -ve labels"""    
    negations=set(['no', 'not', 'never'])
    num = 0
    text = review.translate(None, string.punctuation) #clean punctuations from review
    num = len(set(text.split()).intersection(negations))
    return 'num_negations',num


def has_pos_emotions(review):
    pos_emotions=set(['love', 'nice', 'sweet', 'awesome', 'enjoy', 'enjoyed'])
    num = 0
    text = review.translate(None, string.punctuation) #clean punctuations from review
    num = len(set(text.split()).intersection(pos_emotions))
    if num > 0:
	num =1
    return 'has_pos_emotions',num

def num_pos_emotions(review):
    """  reviews containing positive emotions have +ve labels"""
    #stemmer=nltk.PorterStemmer() #this made code run very slow...therefore added the stemmed words in the list itself
    
    pos_emotions=set(['love', 'nice', 'sweet', 'awesome', 'enjoy', 'enjoyed'])
    num = 0
    text = review.translate(None, string.punctuation) #clean punctuations from review
    num = len(set(text.split()).intersection(pos_emotions))
    return 'num_pos_emotions',num

def num_positives(review):
    positives=set(['awesome','yes', 'good', 'great'])
    num = 0
    text = review.translate(None, string.punctuation) #clean punctuations from review
    num = len(set(text.split()).intersection(positives))
    return 'num_positives',num

def num_neg_emotions(review):
    neg_emotions=set(['hurt', 'ugly','nasty'])
    num = 0
    text = review.translate(None, string.punctuation) #clean punctuations from review
    num = len(set(text.split()).intersection(neg_emotions))
    return 'num_neg_emotions',num


def num_anger_sadness(review):
    sadness=set(['hate', 'annoy','sad'])
    num = 0
    text = review.translate(None, string.punctuation) #clean punctuations from review
    num = len(set(text.split()).intersection(sadness))
    return 'num_anger',num

def num_exclamationmarks(review):
    val=0
    for word in review.split():
	val+=word.count('!')
    return 'num_exclamations',val

def hasexclamationmarks(review):
    val=0
    for word in review.split():
        if word.find('!') != -1:
	    val=1
	    return 'hasexclamation',val
    return 'hasexclamation',val



#Load the training/heldout file and extract the text
def load_text_from_file(filename):
    products=[]
    scores=[]
    reviews=[]
    with open(filename,'r') as f:
        #Change the labels to -1,0,+1
        filelines=f.read().splitlines()
        for lines in filelines:
            line=lines.rstrip('\r\n').split('\t')
            product=line[0]
            products.append(product)
            attr_labels=line[1]
            review_score=[]
            labels=re.findall(pattern,attr_labels)
            
            for l in labels:
                try:
                    l=int(l)
                    if l>0:
                        l=1
                    elif l<0:
                        l=-1
                    else:
                        l=0
                except ValueError:
		    # ignore the label when it is [u], [cs],etc which are not true labels
                    continue
                review_score.append(l)
            mean_score=0
            if len(review_score)!=0:
                mean_score=sum(review_score)/len(review_score)
                if mean_score>0:
           	        mean_score=1
                elif mean_score<0:
                    mean_score=-1
                else:
                    mean_score=0
            
            scores.append(mean_score)       
            review=line[2].strip()
            reviews.append(review.lower())
    f.close()
    return products,scores,reviews
    
#Feature Extraction
def extract_features(reviews):
    train_set=[]
    i=0
    for review in reviews: #Extract features for each review
        features={} 
	key,val=has_negativeconjunctions(review)
	features[key]=val	
	key,val=num_negations(review)
	features[key]=val
	key,val=num_pos_emotions(review)
	features[key]=val
	key,val=num_positives(review)
	features[key]=val
	key,val=num_exclamationmarks(review)
	features[key]=val
	key,val=hasexclamationmarks(review)
	features[key]=val
	key,val=num_neg_emotions(review)
	features[key]=val	
	
	key,val=num_anger_sadness(review)
	features[key]=val
	
	key,val=has_pos_emotions(review)
	features[key]=val	
	      
        train_set.append((features,scores[i]))
        i+=1
    return train_set
            
if __name__=='__main__':
    import time
    st = time.time()
    trainfile='trainingfile.txt'
    heldout='heldoutfile.txt'
    products,scores,reviews=load_text_from_file(trainfile)
    train_set=extract_features(reviews)
    #Training the classifier. 
    clf=SklearnClassifier(LinearSVC())
    svm_model=clf.train(train_set)
    model=nltk.NaiveBayesClassifier.train(train_set)
    
    #Testing the model on the heldout file
    products,scores,reviews=load_text_from_file(heldout)
    heldout_set=extract_features(reviews)
    et = time.time()
    print "Time taken for classifier to run in seconds: ", et-st
    print "Naive Bayes Accuracy: ",nltk.classify.accuracy(model,heldout_set)
    print "SVM Accuracy: ",nltk.classify.accuracy(svm_model,heldout_set)    
    print model.show_most_informative_features(6)    
    #print model.show_most_informative_features(5)
    
"""''
Output:
Time taken for classifier to run in seconds:  1.09572315216
Naive Bayes Accuracy:  0.665317919075
SVM Accuracy:  0.640462427746
Most Informative Features
        num_neg_emotions = 1                   0 : 1      =     15.1 : 1.0
        num_exclamations = 3                   0 : -1     =      8.1 : 1.0
        has_pos_emotions = 1                   1 : -1     =      6.1 : 1.0
        num_pos_emotions = 1                   1 : -1     =      6.0 : 1.0
           num_positives = 1                   1 : -1     =      4.3 : 1.0
               num_anger = 1                  -1 : 1      =      4.3 : 1.0
'''"""