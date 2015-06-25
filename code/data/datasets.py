import numpy as np
import sklearn.datasets
import csv
import MySQLdb

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




"""
	Before you can use MySQL as database, you first have to create a user 'casestudies' using the admin/root account.
	Then you have to make sure that the database HIGGS is created and that the user has access to it.
	mysql -u root -p
	CREATE USER 'casestudies'@'localhost';
	CREATE DATABASE HIGGS;
	GRANT ALL PRIVILEGES ON HIGGS.* TO 'casestudies'@'localhost';
"""
	
def get_mysql(): 	
	db = MySQLdb.connect(
			    user="casestudies",
			    db="HIGGS"
			    ) # name of the data base	
	cur = db.cursor()
	
	table_name = "DATA"
	dimensions = 29
	
	return db, cur, table_name, dimensions


def create_higgs():
	db, cur, table_name, dimensions = get_mysql() 
	sql =   "CREATE TABLE IF NOT EXISTS " + table_name + " (ID INTEGER PRIMARY KEY, "
	for i in range(dimensions):
		sql += "x_" + str(i) + " DOUBLE, " 
	sql = sql[:-2]
	sql += ");"
	
	try:
		cur.execute(sql)
	except Warning, w:
		print w
	cur.close()
	db.close()
	

def load_higgs_into_mysql():
	
	create_higgs()
	db, cur, table_name, dimensions = get_mysql() 
	
	generic = "INSERT INTO " + table_name + " VALUES ("
	
	file_name = '../../datasets/HIGGS.csv'
	csvfile = open(file_name, "rb")
	
	for count, line in enumerate(iter(csvfile)):
		entries = line.split(",")
		insert_statement = generic 
		insert_statement += str(count) + ","
		for index, entry in enumerate(entries):
		    if index >= dimensions:
			    break
		    insert_statement += entry + ","
		    
		insert_statement = insert_statement[:-1]
		insert_statement += ')' 
		print count
		if count > 1e5:
		    break
		try:
			cur.execute(insert_statement)
		except:
			continue
	
	csvfile.close()
	
	db.commit()
	cur.close()
	db.close()

def get_higgs_mysql(ID_list):
	create_higgs()
	
	db, cur, table_name, dimensions = get_mysql() 
	
	query = "SELECT * FROM " + table_name + " WHERE ID IN ("
	for ID in ID_list:
		query += "'" + str(ID) + "'" + ","
	query = query[:-1]
	query += ");"
	#print query
	cur.execute(query)
	X, y = [], []
	for c in cur:
		X_tmp, y_tmp = [], None
		for index in range(len(c)):
			if index == 0: 
				continue
			elif index == 1:
				y_tmp = c[index]
				X_tmp.append(1.0)
			else:
				X_tmp.append(c[index])
		X.append(np.array(X_tmp))
		y.append(y_tmp)
	cur.close()
	#db.close()
	
	return X, y
	
	
def load_higgs(rowlim=1000):
    file_name = '../datasets/HIGGS.csv'
    X, y = [], []
    for row in getdata(file_name, rowlim):
        X.append(np.array([1.0] + [float(r) for r in row[1:]]))
        y.append(float(row[0])) 
    # print type(X[0])
    # print X[0][0]
    return X, y



if __name__ == "__main__":
	load_higgs_into_mysql()
	print get_higgs_mysql([1,2,5, 55, 332, 3456, 0])
	