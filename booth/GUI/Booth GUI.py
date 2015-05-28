# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:14:16 2015

@author: Fin Bauer
"""

import easygui as g
import subprocess
#import change_name as cn
start = 0
for i in range(start, 1000): # damit wir nach einem Schüler wieder von vorne anfangen
    
    g.msgbox(msg = "Wir machen jetzt 10 Fotos für unsere Datenbank von dir!")
    
    for j in range(10):
        subprocess.call(["fswebcam", "-q", "-r", "1280x1024", "--jpeg", "85", "--no-banner", "-D", "0.2", "2015-05-29-%d_%d.jpg" %(i,j)])

    g.msgbox(msg = "Und noch ein Foto! Schaffst du es, den Computer auszutricksen??")
    
    msg = "Jetzt brauchen wir nur noch deinen Namen"
    title = "Wie heißt du?"
    fieldNames = ["Vorname", "Nachname"]
    fieldValues = []
    fieldValues = g.multenterbox(msg, title, fieldNames)
    
    fName, lName = fieldValues
    
    s = fName + "_" + lName
    #cn.change_name(s)
    
    """
    Hier noch die anderen Funktionen die dann rechnen
    """
    predName = "Jakob Heuke" # Hier dann der predictete Name
    uglyface = "url.jpg" # Hier kommt der path zum uglyface hin
    reply = g.buttonbox(msg = "Du heißt " + predName + "!", title = "Mein Tip", \
        image = uglyface, choices = ["Ja", "Nein"])
    
    if reply == "Ja":
        g.msgbox(msg = "Der Computer hat gewonnen!", image = "Frowny.jpg")
    else:
        g.msgbox(msg = "Glückwunsch du hast gewonnen!", image = "Smiley.jpg")
        
        
