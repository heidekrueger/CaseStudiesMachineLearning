
import os,sys

import Image

import glob
import subprocess

##
## USAGE
##
## e.g.
## name = "images/Our Photos/2015-04-22 19.01.17.jpg"
## res = load_standardised_image( name , 10, 10 )
##
def load_standardised_image( filename, width, height ):
	
	###
	### Load Image
	###

	image = Image.open(filename)

	### 
	### Scale Image
	###

	resied_image = image.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter

	return resied_image
	
