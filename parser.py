__author__ = "Prabhavathi Matta"

#from __future__ import division
import nltk
import codecs
import sys
import string
import re

def parse_files(filenames):
    all_attributes = {}
    all_reviews = {}
    pattern = re.compile(r"(\w+)(\[.*?\])(\[.*\])?")    
    for product_file in filenames:
        product_attr = []
        product_reviews = {}
        with codecs.open(product_file, 'r', 'utf-8') as fin:
            for line in fin:
                try:
                    attributes, review = line.strip('').split("##")
                    product_reviews[review.strip()] = attributes.strip().split()
                    attributes = attributes.strip()
                    if len(attributes) != 0:
                        for attr in attributes.split(','):
                            if pattern.match(attr) != None:
                                attrList = list(pattern.match(attr).groups())
                            product_attr.append(attrList[0])
                except Exception,e:
                    continue
                    #print "Error in line: ",line
                    
                
    all_attributes[product_file] = product_attr
    all_reviews[product_file] = product_reviews
    return all_attributes, all_reviews
                
    
if __name__ == '__main__':
    all_attr, all_reviews = parse_files(sys.argv[1:])