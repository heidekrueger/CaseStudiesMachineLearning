"""
    This script is made for change the name of students photo
    just saved
"""

import os


def def_savename():
    '''
    This function ask you to enter your first and family name
    to produce a saving name used tor rename the photos just taken
    '''

    check = False

    s_name = raw_input("Enter your first name : ")
    f_name = raw_input("Enter your family name : ")
    check = input("Please type True or False to validate your choice : ")

    s = "empty"
    if check is True:
        s = s_name + "_" + f_name
    return s


def change_name(s):
    '''
    This function modif2015-05-06-151352ies the name of the photos just taken
    '''

    print "saving name : " + s

    # date of the photo
    date = '2015-05-07'

    # path to the webcam rep
    path_webcam = "../../random_rep/"

    # path to the faces rep
    path_faces = "faces/"

    # path to the ugly faces
    path_uglyfaces = "uglyfaces/"

    # extension
    f_ext = ".jpg"

    # gather all files beginning with the prefix date
    prefixed = [fn for fn in os.listdir(path_webcam) if fn.startswith(date)]

    prefixed.sort()

    for i in range(len(prefixed)):
        o_s = path_webcam + prefixed[i]
        print o_s

        l = prefixed[i]
        l = l.split("-")

        # if the file is the ugly face
        if len(l) == 4:
            l = l[-1]
            l = l.split(".")
            l = l[0]
            n_s = path_uglyfaces + l + f_ext
            print n_s

        # if the file is a normal face
        else:
            n_s = path_faces + s + "_" + str(i) + f_ext
            print n_s

        # move the file
        try:
            os.rename(o_s, n_s)
            print "file moved with succes"

        except:
            print("could not find file")

if __name__ == "__main__":
    s = "empty"

    while s == "empty":
        s = def_savename()

        if s == "empty":
            print "Please try again"
        else:
            print "Thanks !"

    change_name(s)
