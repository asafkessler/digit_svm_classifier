from sklearn import datasets, svm, neural_network
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt

def split_list(a_list):
    part = int(len(a_list) * 70/100)
    return a_list[:part], a_list[part:]

digits = datasets.load_digits()
data_train, data_test = split_list(digits["data"])
target_train, target_test = split_list(digits["target"])
c = [0,0,0,0,0,0,0,0,0,0]
wrong_nums=[]
wrong_images=[]
wrong_guesses=[]
z, images_test = split_list(digits["images"])
p = 0
# classifier = svm.SVC(gamma=0.001, C=10000)
classifier = knn(n_neighbors=1)
classifier.fit(data_train, target_train)
for i in range(len(data_test)):
    if classifier.predict(data_test[[i]]) == target_test[[i]]:
        p+=1
    else:
        c[int(target_test[[i]])]+=1
        wrong_images.append(images_test[i])
        wrong_nums.append(target_test[i])
        wrong_guesses.append(classifier.predict(data_test[[i]]))


p = round(p/len(data_test) *100,1)
print("Succes: "+str(p)+"%")
print("The wrongest digits are ")
for i in range(len(c)):
    if c[i] == max(c):
        print(str(i))
for i in range(len(wrong_images)):
    print("guessing  : " + str(wrong_guesses[i]))
    print("wrong picture is " + str(wrong_nums[i]))
    plt.imshow(wrong_images[i], cmap=plt.cm.gray_r)
    plt.show()
