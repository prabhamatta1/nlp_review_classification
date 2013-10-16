__author__ = "G4"

import re
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
import os       
import pickle as pk
pattern='\[(.*?)\]'

import classifier_prabha
import classifier_jt
import extract_features_david

# gloabl variable for feature words of unigram model
feature_words = []

      
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

def combine_sets(set1, set2, set3, set4, mode='train'):
  final_set = []
  if mode=="train":
    for i in range(len(set1)):
      dict1 = set1[i][0]
      dict2 = set2[i][0]
      dict3 = set3[i][0]
      dict4 = set4[i][0]
      score = set1[i][1]
      final_set.append((dict(dict1.items() + dict2.items() + dict3.items() + dict4.items()), score))
  else:
    for i in range(len(set1)):
      dict1 = set1[i]
      dict2 = set2[i]
      dict3 = set3[i]
      dict4 = set4[i]
      final_set.append(dict(dict1.items() + dict2.items() + dict3.items() + dict4.items()))

  return final_set



def train_classifier(trainfile):
    '''Training the classifier '''
    products,scores,reviews=load_text_from_file(trainfile)

    train_set_pi=extract_features(reviews,scores)
    train_set_prabha=classifier_prabha.extract_features(reviews,scores)
    train_set_david=extract_features_david.extract_features_david(reviews,scores)
    # get feature words from review texts
    feature_words = classifier_jt.get_feature_words(reviews, scores)
    train_set_jt=classifier_jt.extract_unigram_feature(reviews,feature_words,scores)

    train_set = combine_sets(train_set_pi, train_set_prabha, train_set_david, train_set_jt)

    clf=SklearnClassifier(LinearSVC())
    #trainlen=int(len(train_set)*0.9)
    #model=clf.train(train_set)
    model=nltk.NaiveBayesClassifier.train(train_set)
    pk.dump(model,open('classifier.p','wb'))
    #print 'Accuracy for the training set: ',nltk.classify.accuracy(model,train_set)
    #print model.show_most_informative_features(5)
    
def evaluate_clf(heldout):
    '''Testing the model on the heldout file'''
    products,scores,reviews=load_text_from_file(heldout)

    heldout_set_pi=extract_features(reviews,scores)
    heldout_set_prabha=classifier_prabha.extract_features(reviews,scores)
    heldout_set_david=extract_features_david.extract_features_david(reviews,scores)
    heldout_set_jt=classifier_jt.extract_unigram_feature(reviews,feature_words,scores)

    heldout_set = combine_sets(heldout_set_pi, heldout_set_prabha, heldout_set_david, heldout_set_jt)
    model=pk.load(open('classifier.p','rb'))
    print 'Accuracy for the heldout set: ',nltk.classify.accuracy(model,heldout_set)
    print model.show_most_informative_features(5)
    
    
def classify_reviews(testfolder):
    '''Classifying the actual test data'''
    model=pk.load(open('classifier.p','rb'))
    outputf=open('g4_output.txt','w+')
    for testfile in os.listdir(testfolder):
        # skip the file like '.DS_store' in Mac OS
        if testfile.startswith('.'):
          continue
        testpath=os.path.join(testfolder,testfile)
        linenos,test_reviews=load_test(testpath)
        test_set_pi=extract_features(test_reviews,mode='test')
        test_set_prabha=classifier_prabha.extract_features(test_reviews,mode='test')
        test_set_david=extract_features_david.extract_features_david(test_reviews,mode='test')
        test_set_jt=classifier_jt.extract_unigram_feature(test_reviews,feature_words,mode='test')
        test_set = combine_sets(test_set_pi, test_set_prabha, test_set_david, test_set_jt, mode='test')
        i=0
        for each_res in test_set:
            # TODO: [t] as netural
            if test_reviews[i].startswith('[t]'):
              outputf.write(str(testfile)+'\t'+str(i+1)+'\t0\n')
            else:
              result=model.classify(each_res)
              outputf.write(str(testfile)+'\t'+str(i+1)+'\t'+str(result)+'\n')
            i+=1
    outputf.close()
    
   
if __name__=='__main__':
    trainfile='trainingfile.txt' #Name of the training file
    heldout='heldoutfile.txt' #Name of the heldout file
    print "training classifier..."
    train_classifier(trainfile) #function that trains the model
    
    testfolder='./testset' #Folder which contains the test sets
    print "classifying reviews...."
    #evaluate_clf(heldout) #function that evaluates the mdoel on the heldout set
    classify_reviews(testfolder) #function that loads the model and classifies
