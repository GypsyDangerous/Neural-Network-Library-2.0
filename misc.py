import numpy as np 
import time
import os

'''
helper module for misc functions
'''
def truncate(x, level=100):
	'''truncate decimals off a number, the number of decimals left will be the same as the number of 0's on the level '''
	return int(x*level)/level

def percent(x, total=100, level=100):
	'''return a percentage of the total and truncate the number with the level '''
	return truncate((x/total)*100, level)

def time_format(secs):
	'''format the input number into hr:min:sec and return a string'''
	h = int(secs / (60 * 60))
	m = int((secs % (60 * 60)) / 60)
	s = secs % 60
	return ("%a: %a: %a" % (h, m, truncate(s)))