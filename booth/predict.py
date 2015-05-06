from __future__ import print_function

import students

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from time import time

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def depickle_classifier():
    with open("classifier.pkl", "rb") as keeptrace:
        clf = pkl.load(keeptrace)
    return clf

if __name__ == "__main__":    
    
    folder = "./uglyfaces/"
    name = raw_input("Dateiname: ")
    ext = ".jpg"
    
    w, h = 30, 50        
        
    clf = depickle_classifier()
    filename = folder + name + ext
    ugly_face = students.load_features_from_image(filename, w, h)      

    y_pred = clf.predict(ugly_face)
    y_prob = clf.predict_proba(ugly_face)
    
    print("")
    print("Predicted Name:", y_pred[0])
    print("With probability:", y_prob.max())
    print("")

