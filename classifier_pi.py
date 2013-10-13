__author__ = "Priya Iyer"

import re
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
pattern='\[(.*?)\]'

      
#Functions to extract features
def num_nt(review):
    val=0
    for word in review:
        word=word.lower()
        if word.endswith('n\'t'):
            val+=1
    return 'isnt',val
    
def num_ques(review):
    val=0
    for word in review:
        word=word.lower()
        val+=word.count('?')
    return 'num_ques',val
    
def isinsight(review):
    stemmer=nltk.PorterStemmer()
    insight=[ 'think', 'know', 'consider']
    val=0
    for word in review:
        word=word.lower()
        word=stemmer.stem(word)
        if word in insight:
            val+=1
    return 'isinsight',val
    
def istentative(review):
    tent=['maybe', 'perhaps', 'guess']
    val=0
    for word in review:
        word=word.lower()
        if word in tent:
            val+=1
    return 'istentative',val
    
def iscertainty(review):
    certain=['always', 'never']
    val=0
    for word in review:
        word=word.lower()
        if word in certain:
            val+=1
    return 'iscertainty',val
    
def isinhibition(review):
    stemmer=nltk.PorterStemmer()
    inhibit=['block', 'constrain', 'stop']
    val=0
    for word in review:
        word=word.lower()
        word=stemmer.stem(word)
        if word in inhibit:
            val+=1
    return 'isinhibition',val
    
def isassent(review):
    stemmer=nltk.PorterStemmer()
    assent=['agree','ok','yes']
    val=0
    for word in review:
        word=word.lower()
        word=stemmer.stem(word)
        if word in assent:
            val+=1
    return 'isassent',val


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
                    l=0
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
            reviews.append(review)
    f.close()
    return products,scores,reviews
    
#Feature Extraction
def extract_features(reviews):
    train_set=[]
    i=0
    #Extract features for each review
    for review in reviews:
        features={}
        review=review.split()
        key,val=num_nt(review)
        features[key]=val
        key,val=num_ques(review)
        features[key]=val
        key,val=isinsight(review)
        features[key]=val
        key,val=istentative(review)
        features[key]=val
        key,val=isassent(review)
        features[key]=val
        key,val=isinhibition(review)
        features[key]=val
        key,val=iscertainty(review)
        features[key]=val
        train_set.append((features,scores[i]))
        i+=1
    return train_set
            
if __name__=='__main__':
    trainfile='trainingfile.txt'
    heldout='heldoutfile.txt'
    products,scores,reviews=load_text_from_file(trainfile)
    train_set=extract_features(reviews)
    #Training the classifier. 
    clf=SklearnClassifier(LinearSVC())
    trainlen=int(len(train_set)*0.9)
    #model=clf.train(train_set[:trainlen])
    model=nltk.NaiveBayesClassifier.train(train_set)
    #Testing the model on the heldout file
    products,scores,reviews=load_text_from_file(heldout)
    heldout_set=extract_features(reviews)
    print nltk.classify.accuracy(model,heldout_set)
    print model.show_most_informative_features(5)

'''
Output:
Accuracy: 0.546242774566
Most Informative Features
                num_ques = 1                  -1 : 0      =      4.5 : 1.0
                    isnt = 2                  -1 : 0      =      3.4 : 1.0
             istentative = 1                  -1 : 0      =      2.7 : 1.0
                    isnt = 1                  -1 : 1      =      2.0 : 1.0
             iscertainty = 1                   1 : 0      =      1.9 : 1.0

'''
