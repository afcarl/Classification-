# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:22:54 2016
@author: JayaramK
"""

import numpy as np
import pickle
import gzip
import os
from PIL import Image





'''
  Loading the MNIST data
'''
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')

training_data, validation_data, test_data = pickle.load(f)
f.close()

train_features, train_label = training_data
validation_features, validation_label = validation_data
test_features, test_label = test_data



'''
  Loading the USPS data
'''

uspsdata = []
uspslabel = []





for root,dirs,files in os.walk("USPS/Numerals"):
    for name in files:
         uspslabel.append(int(root[len(root) - 1]))

         path = os.path.join(root, name)
         img = Image.open(path)
         w = 28
         h = 28
         img = img.resize((w,h),Image.ANTIALIAS)
         myarray = np.array(img)
         myarray = myarray/np.sum(myarray)
         flattenedarray = myarray.flatten()
         uspsdata.append(flattenedarray)



'''
   Print Results
'''
def PrintResults(accuracy,displaystring):
      print(displaystring)
      print("Accuracy="+str(accuracy))



'''
   Method to calculate the accuracy
'''
def CalculateAccuracy(ykcollectmat,data_label):
    count = 0
    for i in range(0,len(data_label)):
        if np.argmax(ykcollectmat[i]) == data_label[i] :
            count = count+1
    accuracy = (float(count) /len(data_label))*100.0
    return accuracy


'''
   Util funciton used in both training and testing phases
'''
def LogisticRegressionUtilFunction(weight_mat,data_feature_sample):
    weight_mat_transpose = np.transpose(weight_mat)
    weight_mat_transpose_prod_x = np.dot(weight_mat_transpose, data_feature_sample)
    ak = np.zeros(10)
    ak = weight_mat_transpose_prod_x + bk
    exp_ak = np.exp(ak)
    exp_sum = np.sum(exp_ak)
    yk = exp_ak / exp_sum
    return yk



"""Implementing and Training  Logistic Regression Model using Stochastic Gradient Descent
"""
def LogisticRegressionModel_Train(data_features,data_label,weight_mat):
    learning_rate = 0.0099
    ykcollectmat = []
    for i in range(0,len(data_features)):
        yk = LogisticRegressionUtilFunction(weight_mat,data_features[i])
        ykcollectmat.append(yk)
        tk = np.zeros(10)
        tk[int(data_label[i])] = 1
        yktk_diff = yk-tk
        Gradient = np.dot(np.transpose(np.matrix(yktk_diff)),np.matrix(data_features[i]))
        eta_prod_gradient = learning_rate*Gradient
        weight_mat = np.array(weight_mat-np.transpose(eta_prod_gradient))
    return weight_mat,ykcollectmat


"""
    Testing Logistic Regression Model
"""
def LogisticRegressionModel_Test(testing_data_features,weight_mat):
    ykcollectmat = []
    for i in range(0,len(testing_data_features)):
        yk = LogisticRegressionUtilFunction(weight_mat,testing_data_features[i])
        ykcollectmat.append(yk)
    return ykcollectmat


bk = np.ones(10)
weight_mat = np.random.random((784,10))

weight_mat_train,ykcollectmat = LogisticRegressionModel_Train(train_features,train_label,weight_mat)
accuracy = CalculateAccuracy(ykcollectmat,train_label)
PrintResults(accuracy,"MNIST_training_dataset_LogisticRegression")

ykcollectmat =  LogisticRegressionModel_Test(validation_features,weight_mat_train)
accuracy = CalculateAccuracy(ykcollectmat,validation_label)
PrintResults(accuracy,"MNIST_validation_dataset_LogisticRegression")

ykcollectmat =  LogisticRegressionModel_Test(test_features,weight_mat_train)
accuracy = CalculateAccuracy(ykcollectmat,test_label)
PrintResults(accuracy,"MNIST_test_dataset_LogisticRegression")


ykcollectmat = LogisticRegressionModel_Test(uspsdata,weight_mat_train)
accuracy = CalculateAccuracy(ykcollectmat,uspslabel)
PrintResults(accuracy,"USPS_dataset_LogisticRegression")
























































































































































































































