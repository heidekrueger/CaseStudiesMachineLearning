# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:14:16 2015

@author: Fin Bauer, Roland Halbig
"""

import easygui as g
import subprocess
import change_name as cn
import predict
import train

if __name__ == "__main__":

    start = 0
    webcam = ["fswebcam", "-q", "-r", "640x512", "--jpeg", "85", "--no-banner", "-D", "0.2"]
    for i in range(start, 1000): # damit wir nach einem Schüler wieder von vorne anfangen     
        
        ## Namen
        msg = "Wir brauchen zunächst deinen Namen um einen Datenbankeintrag für dich anzulegen."
        title = "Wie heißt du?"
        fieldNames = ["Vorname", "Nachname"]
        fieldValues = []
        fieldValues = g.multenterbox(msg, title, fieldNames)

        fName, lName = fieldValues
        identifier = fName + "_" + lName
        cn.change_name(identifier)

        ## 10 Fotos
        msg = "Wir machen jetzt 10 Fotos für unsere Datenbank von dir!"
        title = "Datenbankeintrag"
#        g.msgbox(msg)
        if g.ccbox(msg, title):     # show a Continue/Cancel dialog
            for j in range(10):
                subprocess.call(webcam + ["Datenbank/%s-%d.jpg" %(identifier, j)])

        ## 1 Foto
        g.msgbox(msg = "Und noch ein Foto! Schaffst du es, den Computer auszutricksen??")
        subprocess.call(webcam + ["Grimassen/%s.jpg" %(identifier)])


        """
        Hier noch die anderen Funktionen die dann rechnen
        """
        
        ## train
        print "Training..."
        train_folder = "Datenbank/"
        w, h = 30, 50
        clf = train.get_trained_classifier(train_folder, w, h)
        
        ## config
        folder = "Grimassen/"
        ext = ".jpg"
        
        ## create filename
        filename = folder + identifier + ext
        name, prob = predict.predict_face(folder, identifier, ext, clf=clf)
        print("")
        print("Predicted Name:", name)
        print("With probability:", prob)
        print("")

        text = "Bist du " + str(name).replace("_", " ") + "? Ich bin mir zu " + str(int(prob*100)) + "% sicher!"

        predict.create_labeled_image(filename, text)

        '''
        predict_face(folder)

        predName = "Jakob Heuke" # Hier dann der predictete Name
        uglyface = "url.jpg" # Hier kommt der path zum uglyface hin
        reply = g.buttonbox(msg = "Du heißt " + predName + "!", title = "Mein Tip", \
            image = uglyface, choices = ["Ja", "Nein"])

        if reply == "Ja":
            g.msgbox(msg = "Der Computer hat gewonnen!", image = "Frowny.jpg")
        else:
            g.msgbox(msg = "Glückwunsch du hast gewonnen!", image = "Smiley.jpg")
        '''
        
        msg = "Soll ein Ausdruck erstellt werden?"
        title = "Drucken"
        if g.ccbox(msg, title):     # show a Continue/Cancel dialog
            subprocess.call(["lpr", "print/%s.jpg" %identifier])

        msg = "Noch ein Versuch?"
        title = "Programm Fortsetzen?"
        if g.ccbox(msg, title):     # show a Continue/Cancel dialog
            pass  # user chose Continue
        else:
            break
            #sys.exit(0)           # user chose Cancel
