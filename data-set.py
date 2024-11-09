from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
# print(digits.keys())
# get dataSet of digits from excising directory
data = digits.data
target = digits.target
images = digits.images
index=10
print(target[index])
print(digits.DESCR)
# defining the classifier with param to determine how it will learn
classifier = svm.SVC(gamma=0.0001, C=100)

plt.imshow(images[index], cmap=plt.cm.gray_r)
plt.show()

























# getting the size of
# sizeOfSet = len(digits.data)

# # getting size of train
# sizeOfTrain = int(sizeOfSet * TRAIN_PERCENT / 100)
# sizeOfTest = sizeOfSet - sizeOfTrain
#
# print("Data set size: ", sizeOfTrain)
# print("Test set size: ", sizeOfTest)
