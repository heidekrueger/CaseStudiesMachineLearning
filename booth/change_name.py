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

    date = '2015-05-06'
    path = "faces/"
    f_ext = ".jpg"

    prefixed = [fn for fn in os.listdir(path) if fn.startswith(date)]
    print len(prefixed)

    prefixed.sort()

    i = 1
    new_s = [s + "_" + str(i) + f_ext for i in range(1, 11)]
    new_s.sort()

    for i in range(10):
        try:
            o_s = path + prefixed[i]
            n_s = path + new_s[i]     
            os.rename(o_s, n_s)
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
