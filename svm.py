#-------------------------------------------------------------------------
# AUTHOR:beize li
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
end_c = 0
end_degree = 0
end_kernel = ""
end_dfs = ""
for i in c: #iterates over c
    for de in degree : #iterates over degree
        for ker in kernel : #iterates kernel
           for dfs in decision_function_shape: #iterates over decision_function_shape
                cor = 0
                accuracy = 0
                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C = i, degree = de, kernel = ker, decision_function_shape = dfs)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                for data in dbTest:
                    true_label = data[-1]
                    class_predicted = clf.predict([data[:-1]])[0]
                    if true_label == class_predicted:
                        cor += 1
                accuracy = cor/len(dbTest)
                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if (accuracy > highestAccuracy):
                   highestAccuracy = accuracy
                   print("highest Svm accuracy so far",highestAccuracy,"Parameters:","a =",i,"degree=", de,"kernel=", ker,"decision_function_shape =", dfs)
                   end_c = i
                   end_dfs =dfs
                   end_degree = de
                   end_kernel = ker
#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
#--> add your Python code here
print("final highest Svm accuracy")
print("highest Svm accuracy so far",highestAccuracy,"Parameters:","a =",end_c,"degree=", end_degree,"kernel=", end_kernel,"decision_function_shape =", end_dfs)











