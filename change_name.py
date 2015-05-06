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
    This function modifies the name of the photos just taken
    '''

    print "saving name : " + s

    n = 10
    old_b = "s_c"
    f_ext = ".txt"
    for i in range(1, n + 1):
        old_s = old_b + str(i) + f_ext
        new_s = s + "_" + str(i) + f_ext
        os.rename(old_s, new_s)

if __name__ == "__main__":
    s = "empty"

    while s == "empty":
        s = def_savename()

        if s == "empty":
            print "Please try again"
        else:
            print "Thanks !"

    change_name(s)
