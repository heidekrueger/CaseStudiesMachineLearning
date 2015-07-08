import re
import numpy as np
from matplotlib import pyplot as plt
from data.datasets import get_higgs_mysql as ghm
from SQN.LogisticRegression import LogisticRegression as LR

def get_batchsizes_from_name(filepath):
        filename = filepath.split("/")[-1]
        filename = filename.split("_")
        b_G = int(filename[0])
        b_H = int(filename[1])
        return b_G, b_H

def get_fixed_sample(size):
        logreg = LR()
        return ghm(range(size))

def get_fixed_F(size):
        X_fix, y_fix = get_fixed_sample(size)
        return lambda w: logreg.F(w,X_fix,y_fix)


def load_result_file(filepath):
        
        resfile = open(filepath, "r")
        
        iters, fevals, gevals, adp, f_S, g_norm_S, time = [], [], [], [], [], [], []
        for count, line in enumerate(iter(resfile)):
                line = re.sub('\s', '', str(line))
                entries = re.split(",", str(line))
                
                iters.append( int(entries[0]) )
                fevals.append( int(entries[1]) )
                gevals.append( int(entries[2]) )
                adp.append( int(entries[3]) )
                f_S.append( float(entries[4]) )
                g_norm_S.append( float(entries[5]) )
                time.append( float(entries[6]) )
                
        return iters, fevals, gevals, adp, f_S, g_norm_S, time
                
def load_result_file_w(filepath):
        
        resfile = open(filepath, "r")
        w = []
        for count, line in enumerate(iter(resfile)):
                line = re.sub('\s', '', str(line))
                entries = re.split(",", str(line))
                w.append(np.array([float(s) for s in entries]))
        return w


file_dir = '../outputs/'
file_name = '1000_0_1.txt'
file_path = file_dir + file_name

b_G, b_H = get_batchsizes_from_name(file_path)
iters, fevals, gevals, adp, f_S, g_norm_S, time = load_result_file(file_path)
w = load_result_file_w(file_path+'_w.txt')

import numpy as np
f_S_MA = [ np.mean(f_S[:i]) if i<100 else np.mean(f_S[i-100:i]) for i in range(len(f_S)) ]
fig = plt.figure()
plt.title("f_S and moving averages")
plt.plot(iters, f_S)
plt.plot(iters, f_S_MA)
plt.yscale("log")
# fig =plt.figure()
# plt.title("test.")
# plt.plot(f_S_MA,iters)
# plt.show()

F = get_fixed_F(1000)

logreg = LR()

X_f, y_f = get_fixed_sample(5)

logreg.w=w[-1]
yp = logreg.predict(X_f)

print yp, y_f

#v = [F(w_i) for w_i in w]

#fig = plt.figure()
#plt.plot(iters, v)
#plt.yscale('log')
#plt.show()

