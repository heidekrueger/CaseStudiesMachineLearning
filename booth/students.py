
import os,sys

import Image

##
## USAGE
##
## e.g.
## name = "images/Our Photos/2015-04-22 19.01.17.jpg"
## res = load_standardised_image( name , 10, 10 )
##
def load_standardised_image( filename, width, height ):
	
	###
	### Load Image in Greyscale
	###

	image = Image.open(filename)#.convert('LA')

	### 
	### Scale Image
	###

	resized_image = image.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter

	return resized_image
	
