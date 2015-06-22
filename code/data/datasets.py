import numpy as np
import sklearn.datasets
import csv
#import MySQLdb

#### functions for reading from loooong file as stream

def getstuff(filename, rowlim):
    with open(filename, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        for row in datareader:
            if count < rowlim:
                yield row
                count += 1
            else:
                return

def getdata(filename, rowlim):
    for row in getstuff(filename, rowlim):
        yield row

#####

def normalize(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    for i in range(X.shape[0]):
	X[i,:] -= mu
	for j in range(X.shape[1]):
	    X[i,j] /= std[j]
    return X

def load_data1():
    dataset = np.genfromtxt(open('../datasets/ex2data1.txt','r'), delimiter=',', dtype='f8')
    X = dataset[:,:2]
    z = dataset[:,2][:,np.newaxis] 
    
    X = normalize(X)
    
    X = np.concatenate( (np.ones( (len(z), 1)), X), axis=1)
    X_new = []
    for i in range(len(z)):
	x = np.array(list(X[i,:].flatten()))
	X_new.append(x)
    return X_new, list(z)

def load_data2():
    dataset = np.genfromtxt(open('../datasets/ex2data2.txt','r'), delimiter=',', dtype='f8')
    X = dataset[:,:2]
    z = dataset[:,2][:,np.newaxis]
    
    X = normalize(X)
    
    X = np.concatenate( (np.ones( (len(z), 1)), X), axis=1)
    X_new = []
    for i in range(len(z)):
	x = np.array(list(X[i,:].flatten()))
	X_new.append(x)
    return X_new, list(z)

def load_iris():
	iris = sklearn.datasets.load_iris()
	X, y = [], []
	for i in range(len(iris.target)):
		if iris.target[i] != 2:
			X.append(np.array([1] + list(iris.data[i])))
			y.append(iris.target[i])
	return X, y


def split_into_files(src, dest_folder):
	counter = 1
	csvfile = open(src, "rb")
	line = csvfile.readline()
	while line is not None:
		line = csvfile.readline()
		with open(dest_folder + "/" + str(counter), "w+") as feature:
			feature.write(line)
		counter += 1
		if counter > 1e6:
			break
	csvfile.close()

def load_higgs_into_mysql():
    
	table_name = "TEST"
	dimensions = 6
	
	db = MySQLdb.connect(host="localhost", 
			    user="casestudies",
#			    passwd="megajonhy", # your password
			    db="data") # name of the data base	
	cur = db.cursor() 
	
	sql =   "create table if not exists " + table_name + "(ID INTEGER PRIMARY KEY"
	for i in range(dimensions):
		sql += "x_" + str(i) + " DOUBLE," 
	sql += "DOUBLE, IS_SET INTEGER);"

	cur.execute(sql)
	
	generic = "INSERT INTO " + table_name + " VALUES ("
	
	file_name = '../datasets/HIGGS.csv'
	with open(filename, "rb") as csvfile:
		
		line = csvfile.readline()
		entries = line.split(",")
		insert_statement = generic 
		for entry in entries:
		    insert_statement += entry + ","
		insert_statement = insert_statement[:-1]
		insert_statement += ')' 
		print insert_statement
		cur.execute(insert_statement)
	
	db.close()
	
	
	
def load_higgs(rowlim=1000):
    file_name = '../datasets/HIGGS.csv'
    X, y = [], []
    for row in getdata(file_name, rowlim):
        X.append(np.array([1.0] + [float(r) for r in row[1:]]))
        y.append(float(row[0])) 
    # print type(X[0])
    # print X[0][0]
    return X, y
