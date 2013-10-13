#!/usr/bin/env python

import os
import re


def labelMaker():
	cwd = os.getcwd()

	IN_FILE = cwd + "/trainingfile.txt"
	OUT_FILE = cwd + "/labelFile.txt"

	pattern = '''\[[+-].]'''

	f = open(IN_FILE,'r')
	o = open(OUT_FILE,'w')

	counter = 1
	for line in f:
		if line == "\n" :
			o.write('\n')
			#print(str(counter))
		else:
			labels = re.findall(pattern,line)
			numList = []
			if len(labels) > 0:
				for lab in labels:
					if lab[1] == "+":
						numList.append(int(lab[2]))
					if lab[1] == "-":
						numList.append(-1*int(lab[2]))						
				sentPol = min(numList)
				if sentPol <= -2:
					o.write('-1\n')
				if sentPol == 1:
					o.write('0\n')
				if sentPol == -1:
					o.write('0\n')
				if sentPol >= 2:
					o.write('1\n')
				print(' '.join([str(counter),str(numList),'min:',str(sentPol)]))
			else:
				o.write('\n')
		counter +=1 

	o.close
	f.close

labelMaker()


