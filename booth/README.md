
Pipeline:

--> Take 10 burstmode images of students using the program "cheese".
--> Take one image/ugly face
--> Run "python change_names.py" in the booth folder
--> Let the student enter his names
--> Train the classifier using "python train.py"
--> Run the prediction using "python predict.py [NR]" where NR is the postfix number of the cheese picture

TODO:
--> Create a file with the ugly face and the prediction value on it which can be printed
--> Print student's ugly face
