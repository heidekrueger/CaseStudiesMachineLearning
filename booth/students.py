
import os, sys, re

import numpy as np

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

	image = Image.open(filename).convert('F')

	### 
	### Scale Image
	###

	resized_image = image.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter

	return resized_image
	

###
### Load, standardize and label image data
###
### USAGE: X, Y = load_student_database( "../../faces/", 20, 20)
###
def load_student_database( folder, width, height):

	images = []
	labels = []
	
	##
	## Walk the dir and get all image files
	##
	
	for root, dirs, files in os.walk(folder):
		for name in files:
			
			## 
			## Get label
			##
			
			ext = name[-3:]
			label = re.sub("_\d+.%s" % ext, "", name)
			
			##
			## Load standardized image
			##
			
			try:
				image = load_standardised_image( root + name, width, height )					
				image = np.asarray(image)
			except:
				print "Error loading standardized Image:", root + name
				continue
				
			images.append(image)
			labels.append(label)
	
	return images, labels
	
def TEST_load_student_database():
	X, Y = load_student_database( "faces/", 20, 20)
	print "data shape:"
	print np.shape( X )
	print "labels:"
	print Y

TEST_load_student_database()

