__author__ = "Prabhavathi Matta"

#from __future__ import division
import nltk
import codecs
import sys
import string
import re
from os.path import basename



def parse_files(filenames):
    all_attributes = {}
    all_reviews = {}
    pattern = re.compile(r"(\w+)(\[.*?\])(\[.*\])?")
    fout = open("trainingfile.txt", 'w')
    
    for product_file in filenames.split(","):
        product_attr = []
        product_reviews = {}
        
        with open(product_file, 'r') as fin:
            product_name = basename(product_file).split('.')[0]
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
                        #print"["+product_name+"]\t"+ attributes +"\t" review
                        output = "["+product_name+"]\t" + attributes +"\t"+ review +"\n"
                        #print output
                        fout.write(output)
                except Exception,e:
                    #continue
                    print "Error in line: ",line
                    
                
    all_attributes[product_file] = product_attr
    all_reviews[product_file] = product_reviews
    
    fout.close()
    return all_attributes, all_reviews


def write_trainingfile(filenames):
    fout = open("trainingfile.txt", 'w')
    for product_file in filenames.split(","):
        with open(product_file, 'r') as fin:
            product_name = basename(product_file).split('.')[0]
            for line in fin:
                try:
                    attributes, review = line.strip().split("##")
                    attributes = attributes.strip()
                    if len(attributes) != 0:
                        output = "["+product_name+"]\t" + attributes +"\t"+ review +"\n"
                        fout.write(output)
                except Exception,e:
                    continue
                    #print "Error in line: ",line
    fout.close()
    return 



                
    
if __name__ == '__main__':
    print "Argument to the parser given==",sys.argv[1]
    #all_attr, all_reviews = parse_files(sys.argv[1])
    write_trainingfile(sys.argv[1])