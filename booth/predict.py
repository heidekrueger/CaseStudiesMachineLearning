from __future__ import print_function

import students

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from time import time
import sys
import re

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import Image, ImageDraw, ImageFont
from os import chdir, path

def depickle_classifier():
    with open("classifier.pkl", "rb") as keeptrace:
        clf = pkl.load(keeptrace)
    return clf

def create_labeled_image(filename, text):
    
    text = re.sub("_", " ", text)
    split = filename.split("/")
    img_name = "print/" + split[-1]

    ## prepare image
    img = Image.open(filename)
    # make an entirely black image
    imgbg = Image.new('F', img.size, "#000000") 
    # make a mask that masks out all
    mask = Image.new('L',img.size,"#000000")       
    # setup to draw on the main image
    draw = ImageDraw.Draw(img)                     
    # setup to draw on the mask
    drawmask = ImageDraw.Draw(mask)                
    
    ## configuration
    fontsize = int(img.size[1]*0.05)
    position = (10,img.size[1]*0.9)
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf", size=fontsize)
    
    # draw a line on the mask to allow some bg through
    drawmask.line((0, position[1] + fontsize/2, img.size[0],position[1] + fontsize/2),
                  fill="#999999", width=int(fontsize*2)) 
                         
    # put the (somewhat) transparent bg on the main
    img.paste(imgbg, mask=mask)                    
    # add some text to the main
    draw.text(position, text, font = font)      
    del draw 
    
    img.save(img_name,"JPEG",quality=100)  

if __name__ == "__main__":

    folder = "uglyfaces/"
    ext = ".jpg"

    ## check for command line arguments
    if len(sys.argv) > 1:
        name = sys.argv[-1]
        
        ## if absolute path given
        if ext in name:
            name = re.sub(ext, "", name)
        if folder in name:
            name = re.sub(folder, "", name)
    ## create prompt
    else:
        name = raw_input("Dateiname: ")
    
    ## create filename
    filename = folder + name + ext
    
    ## params
    w, h = 30, 50        
        
    clf = depickle_classifier()
    ugly_face = students.load_features_from_image(filename, w, h)      

    y_pred = clf.predict(ugly_face)
    y_prob = clf.predict_proba(ugly_face)
    
    print("")
    print("Predicted Name:", y_pred[0])
    print("With probability:", y_prob.max())
    print("")
    
    text = "Predicted Name: " + str(y_pred[0]) + " with probability " + str(int(y_prob.max()*100)) + "%"
    
    create_labeled_image(filename, text)
    
    
    
