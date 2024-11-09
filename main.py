from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
train = digits.data[:int(0.67 * len(digits.data))]
test = digits.data[int(0.67 * len(digits.data)):-1]
images = digits.images[int(0.67 * len(digits.data)):-1]

target_train = digits.target[:int(0.67 * len(digits.target))]
target_test = digits.target[int(0.67 * len(digits.target)):-1]

classifier = svm.SVC(gamma=0.008, C=1.5)
classifier.fit(train, target_train)
right = 0.0
wrong = 0.0

for i in range(0, len(test)):
    prediction = classifier.predict(test[[i]])
    if prediction == target_test[i]:
        right += 1
        print("Success ! ")
        print("prediction ", prediction[0])
        print("Actual", target_test[i])
        plt.imshow(images[i],cmap=plt.cm.gray_r)
        plt.show()
    else:
        wrong += 1
        print("Error !")
        print("prediction ", prediction[0])
        print("Actual", target_test[i])
        plt.imshow(digits.images[i],cmap=plt.cm.gray_r)
        plt.show()

right_Percent = float(right / (right + wrong)) * 100
wrong_Percent = float(wrong / (right + wrong)) * 100
print("right Percent :" , right_Percent)
print ("wrong Percent : " , wrong_Percent)
