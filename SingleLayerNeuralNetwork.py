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
   Method to calculate the accuracy
'''
def CalculateAccuracy(ykcollectmat, data_label):
    count = 0
    for i in range(0, len(data_label)):
        if np.argmax(ykcollectmat[i]) == data_label[i]:
            count = count + 1
    print count
    accuracy = (float(count) / len(data_label)) * 100.0
    return accuracy

'''
   Print Results
'''
def PrintResults(accuracy,displaystring):
      print(displaystring)
      print("Accuracy="+str(accuracy))




def sigmoid(input):
    return 1/ (1 + np.exp(-input))


def SingleLayerNeuralNetwork_Util(data_features_sample,wij,wkj,bj,bk,trainflag):
    h_input = np.dot(data_features_sample, wij) + bj
    zj = sigmoid(h_input)
    ak = np.dot(zj, wkj) + bk
    exp_ak = np.exp(ak)
    exp_sum = np.sum(exp_ak)
    yk = exp_ak / exp_sum
    if trainflag:
        return yk,ak,zj
    else:
        return yk


'''
   Testing Single Layer Neural Network
'''
def SingleLayerNeuralNetwork_Test(data_features,wij, wkj, bj, bk):
    ykcollectmat = []
    for i in range(0, len(data_features)):
        yk = SingleLayerNeuralNetwork_Util(data_features[i],wij, wkj, bj,bk,False)
        ykcollectmat.append(yk)
    return ykcollectmat


'''
   Implementation and Training  Single Layer Neural Network
'''
def SingleLayerNeuralNetwork_Train(data_features,data_label,M,wij,wkj,bj,bk):
    ykcollectmat = []
    eta_learningrate = 0.0099
    for i in range(0, len(data_features)):
        yk,aj,zj = SingleLayerNeuralNetwork_Util(data_features[i], wij, wkj, bj, bk,True)
        ykcollectmat.append(yk)
        tk = np.zeros(10)
        tk[int(data_label[i])] = 1
        deltak = yk - tk
        h_derivative = np.dot(sigmoid(zj), (1 - sigmoid(zj)))
        wkj_prod_deltak = np.dot(deltak, np.transpose(wkj))
        deltaj = np.dot(h_derivative, wkj_prod_deltak)
        deltaj_prod_xi = np.dot(np.transpose(np.matrix(data_features[i])), np.matrix(deltaj))
        deltak_prod_zj = np.dot(np.transpose(np.matrix(zj)), np.matrix(deltak))
        wij = np.array(wij - (eta_learningrate * deltaj_prod_xi))
        wkj = np.array(wkj - (eta_learningrate * deltak_prod_zj))
        bj = (bj - (eta_learningrate * deltaj))
        bk = (bk - (eta_learningrate * deltak))
    return wij, wkj, bj, bk, ykcollectmat


M = 1000
wij = np.random.randn(len(train_features[0]),M)
wij = wij/np.sum(wij)
wkj = np.random.randn(M,10)
wkj = wkj/np.sum(wkj)

bj = np.ones(M)
bk = np.ones(10)


wijtrain,wkjtrain,bjtrain,bktrain,ykcollectmat = SingleLayerNeuralNetwork_Train(train_features,train_label,M,wij,wkj,bj,bk)
accuracy = CalculateAccuracy(ykcollectmat,train_label)
PrintResults(accuracy,"MNIST_training_dataset_Sigle Layer Neural Network")


ykcollectmat = SingleLayerNeuralNetwork_Test(validation_features,wijtrain,wkjtrain,bjtrain,bktrain)
accuracy = CalculateAccuracy(ykcollectmat,validation_label)
PrintResults(accuracy,"MNIST_validation_dataset_Sigle Layer Neural Network")


ykcollectmat = SingleLayerNeuralNetwork_Test(test_features,wijtrain,wkjtrain,bjtrain,bktrain)
accuracy = CalculateAccuracy(ykcollectmat,test_label)
PrintResults(accuracy,"MNIST_test_dataset_Sigle Layer Neural Network")


ykcollectmat = SingleLayerNeuralNetwork_Test(uspsdata,wijtrain,wkjtrain,bjtrain,bktrain)
accuracy = CalculateAccuracy(ykcollectmat,uspslabel)
PrintResults(accuracy,"USPS_dataset_Sigle Layer Neural Network")



