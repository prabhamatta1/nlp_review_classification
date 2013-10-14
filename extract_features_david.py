#!/usr/bin/env python

#dgreis feature extractor

def extract_features_david(reviews,scores,mode='train'):
    train_set=[]
    i=0
    articles = 'a an the'
    prosentwords = "yes no okay OK" 
    pronouns = "all another any anybody anyone anything both each each_other either everybody his everyone I everything it few its he \
    itself her little hers many herself me him mine himself more most much myself neither no one nobody none nothing one one_another other that\
    what others theirs whatever ours them which ourselves themselves whichever several these who she they whoever some this whom somebody those\
    whomever someone us whose something we you our yours yourself yourselves"
    auxverbs = "are can aren\'t cannot ain\'t can\'t \'re could be couldn\'t been did didn\'t hadn\'t haven\'t do \'d \'ve don\'t has is does hasn\'t isn\'t doesn\'t \'s \'s had have may might shouldn\'t mightn\'t was mustn\'t wasn\'t shall were shan\'t weren\'t should will"
    conjs = "and or though because yet unless nor so when now_that even_though although if now_that only_if while whereas whether_or_not in_order_that in_case even_if until"
    inters = "adios ah aha ahem ahoy alack alas all_hail alleluia aloha amen attaboy aw ay bah dear begorra doh behold duh bejesus eh bingo encore bleep eureka boo fie bravo gee bye gee_whiz cheerio gesundheit cheers goodness ciao gosh crikey great cripes hah Ha-ha hail hallelujah heigh-ho hello hem hey hey_presto hi hip hmm ho ho_hum hot dog howdy hoy huh humph hurray hush indeed jeepers_creepers jeez lo_and_behold man my_word now ooh oops tush ouch tut phew Tut-tut phooey ugh pip-pip uh-huh pooh uh-oh pshaw uheuh rats viva righto voila scat wahoo shoo well shoot who so long whoopee Touch whoops whoosh wow yay yikes yippee yo yoicks yoo-hoo yuk yummy zap"
    #Extract features for each review 
    biglist = [(articles,'articles'),(prosentwords,'prosentwords'),(pronouns,'pronouns'),(auxverbs,'auxverbs'),(conjs,'conjs'),(inters,'inters')]
    for review in reviews:
      features={}
      if mode=='train':
        review=review.split()
      for j in biglist:
    	  key,val=functionTyper_group(j[0],j[1],review)
    	  features[key]=val
      if mode=='train':
        train_set.append((features,scores[i]))
      else:
        train_set.append(features)
      i+=1
    return train_set

def functionTyper_group(typeSet,typeName,line):
	funcSplit = typeSet.split()
	returnval = [typeName+'CNT',0]
	for i in range(len(funcSplit)):
		funcSplit[i] = funcSplit[i].split('_')
		funcSplit[i] = ' '+ ' '.join(funcSplit[i])+' '
		if funcSplit[i].startswith(' \'') == True:
			funcSplit[i] = funcSplit[i][1:]
		for w in line:
			returnval[1] = returnval[1] + w.lower().count(funcSplit[i])
	return(returnval[0],returnval[1])
