__author__ = "Priya Iyer"

import re
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
import os       
import pickle as pk
pattern='\[(.*?)\]'

      
#Start: Functions to extract features
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
#End: Functions to extract features

#Load the training/heldout file and extract the text
def load_text_from_file(filename):
    '''Parses the file to extract reviews from them'''
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
                    review_score.append(l)
                except ValueError:
                    l=0    
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
def extract_features(reviews,scores=[],mode='train'):
    '''Extracts features from the reviews'''
    train_set=[]
    i=0
    #Extract features for each review
    for review in reviews:
        features={}
        if mode=='train':
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
        if mode=='train':
            train_set.append((features,scores[i]))
        else:
            train_set.append(features)
        i+=1
    
    return train_set
            
            
def load_test(filename):
    linenos=[]
    reviews=[]
    with open(filename,'r') as test:
        for line in test:
            line=line.replace('##','')
            line=line.rstrip('\r\n').split('\t')
            linenos.append(line[0])
            reviews.append(line[1])
    
    test.close()
    return linenos,reviews


def train_classifier(trainfile):
    '''Training the classifier '''
    products,scores,reviews=load_text_from_file(trainfile)
    train_set=extract_features(reviews,scores)
    clf=SklearnClassifier(LinearSVC())
    #trainlen=int(len(train_set)*0.9)
    model=clf.train(train_set)
    #model=nltk.NaiveBayesClassifier.train(train_set)
    pk.dump(model,open('classifier.p','wb'))
    print 'Accuracy for the training set: ',nltk.classify.accuracy(model,train_set)
    #print model.show_most_informative_features(5)
    
def evaluate_clf(heldout):
    '''Testing the model on the heldout file'''
    products,scores,reviews=load_text_from_file(heldout)
    heldout_set=extract_features(reviews,scores)
    model=pk.load(open('classifier.p','rb'))
    print 'Accuracy for the heldout set: ',nltk.classify.accuracy(model,heldout_set)
    #print model.show_most_informative_features(5)
    
    
def classify_reviews(testfolder):
    '''Classifying the actual test data'''
    model=pk.load(open('classifier.p','rb'))
    outputf=open('g_4_output.txt','w+')
    for testfile in os.listdir(testfolder):
        testpath=os.path.join(testfolder,testfile)
        test_reviews,linenos=load_test(testpath)
        test_set=extract_features(test_reviews,mode='test')
        i=0
        for each_res in test_set:
            result=model.classify(each_res)
            outputf.write(str(testfile)+'\t'+str(i)+'\t'+str(result)+'\n')
            i+=1
    outputf.close()
    
   
if __name__=='__main__':
    trainfile='trainingfile.txt' #Name of the training file
    heldout='heldoutfile.txt' #Name of the heldout file
    testfolder='../testset' #Folder which contains the test sets
    train_classifier(trainfile) #function that trains the model
    evaluate_clf(heldout) #function that evaluates the mdoel on the heldout set
    classify_reviews(testfolder) #function that loads the model and classifies

'''
Write-up:
(i) Some of the features that I decided to work on were: 
    a. Function: num_nt()
    Number of words ending in "n't". Eg: isn't, doesn't, hasn't etc. This is because from my eyeballing of the training file, I found this recurring pattern in negative reviews.
    b. Function: num_ques()
    How many question marks in the reviews? Same reason for choice as above.
    c. Function: is_insight()
    Does the review contain words that are 'insight'? (Cheng et.al)
    d. Function: is_tentative()
    Does the review contain words that are 'tentative'? (Cheng et.al)
    e. Function: is_certainty()
    Does the review contain words that are 'certainty'? (Cheng et.al)
    f. Function: is_inhibition()
    Does the review contain words that are 'inhibition'? (Cheng et.al)
    g. Function: is_assent()
    Does the review contain words that are 'assent'? (Cheng et.al)
    
(ii) As a group, we decided to output our results for each of the feature extraction functions as a tuple of (feature_name,feature_value)

(iii) In some of the feature functions such as 'isassent' and 'isinsight', since the feature I was trying to measure was a verb, I had to stem each of the words
in the reviews to its root form.

(iv)

Output:

Accuracy for the training set:  0.629298162977
Most Informative Features
                num_ques = 3                   0 : -1     =      4.9 : 1.0
             istentative = 1                   0 : 1      =      3.5 : 1.0
                    isnt = 2                  -1 : 1      =      2.3 : 1.0
                num_ques = 1                  -1 : 1      =      2.2 : 1.0
                    isnt = 1                  -1 : 1      =      1.9 : 1.0

Accuracy for the heldout set:  0.634682080925
Most Informative Features
                num_ques = 3                   0 : -1     =      4.9 : 1.0
             istentative = 1                   0 : 1      =      3.5 : 1.0
                    isnt = 2                  -1 : 1      =      2.3 : 1.0
                num_ques = 1                  -1 : 1      =      2.2 : 1.0
                    isnt = 1                  -1 : 1      =      1.9 : 1.0

'''

