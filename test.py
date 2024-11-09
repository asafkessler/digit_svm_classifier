import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

FACTOR = 70
# get dataSet of digits from excising directory
digits = datasets.load_digits()
print("My Information : ", digits.keys())
my_data = digits['data']
my_target = digits['target']


# len(my_data) == len(my_target)
cutting_place = round((len(my_data) * FACTOR) / 100)

train_data = my_data[:cutting_place]
test_data = my_data[cutting_place:]
train_target = my_target[:cutting_place]
test_target = my_target[cutting_place:]
