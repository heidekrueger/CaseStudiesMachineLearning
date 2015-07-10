from sklearn.cross_validation import train_test_split
import data.datasets as datasets


X, y = datasets.load_eeg()

print "Data dim : ", X.shape
print "Label dim : ", y.shape

# a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.33, random_state=42)

