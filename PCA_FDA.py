import numpy as np
import pandas as pd
import os
import kagglehub
import idx2numpy
import matplotlib.pyplot as plt
import random


############################# LOADING DATA
X_train = "C:/Users/Dhruv/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-images.idx3-ubyte"
y_train = "C:/Users/Dhruv/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-labels.idx1-ubyte"
X_test = "C:/Users/Dhruv/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-images.idx3-ubyte"
y_test = "C:/Users/Dhruv/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-labels.idx1-ubyte"

X_train = idx2numpy.convert_from_file(X_train)
y_train = idx2numpy.convert_from_file(y_train)
X_test = idx2numpy.convert_from_file(X_test)
y_test = idx2numpy.convert_from_file(y_test)

############################# SEGREGATION OF DATA
zeros = []
ones = []
twos = []
for i in range(60000):
    if y_train[i] == 0:
        zeros.append(X_train[i])
    if y_train[i] == 1:
        ones.append(X_train[i])
    if y_train[i] == 2:
        twos.append(X_train[i])

zeros_test = []
ones_test = []
twos_test = []
for i in range(10000):
    if y_test[i] == 0:
        zeros_test.append(X_test[i])
    if y_test[i] == 1:
        ones_test.append(X_test[i])
    if y_test[i] == 2:
        twos_test.append(X_test[i])

############################# SELECTION, FLATTENING, NORMALIZATION OF DATA

random.shuffle(zeros)
random.shuffle(ones)
random.shuffle(twos)
random.shuffle(zeros_test)
random.shuffle(ones_test)
random.shuffle(twos_test)

zeros = zeros[:100]
ones = ones[:100]
twos = twos[:100]

zeros_test = zeros_test[:100]
ones_test = ones_test[:100]
twos_test = twos_test[:100]

def normalizer(list_of_matrices):
    ans = []
    for i in list_of_matrices:
        i = i.reshape(-1)
        i = i/255
        ans.append(i)
    return ans

zeros = normalizer(zeros)
ones = normalizer(ones) 
twos = normalizer(twos)
zeros_test = normalizer(zeros_test)
ones_test = normalizer(ones_test) 
twos_test = normalizer(twos_test)

############################# PLOTTING

def display_images(feature_vectors, labels, images_per_row=10):
    num_images = len(feature_vectors)
    rows = (num_images + images_per_row - 1) // images_per_row 
    fig, axes = plt.subplots(rows, images_per_row, figsize=(images_per_row * 2, rows * 2))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image = feature_vectors[i].reshape(28, 28)
            ax.imshow(image, cmap="gray")
            ax.set_title(str(labels[i]))
        ax.axis("off")

    plt.tight_layout()
    plt.show()

############################# 2. Computing MLE estimates
#################### 2a. Estimating Mean and Covariance matrix for each class

def mean_finder(data_matrix):
    return np.mean(data_matrix, axis=0, keepdims=True).T

def centralizer_of_data(data_matrix):
    mean = mean_finder(data_matrix)
    data_matrix = np.vstack(data_matrix).T
    centered_data = data_matrix - mean
    return centered_data

def covariance_finder(data_matrix):
    a = centralizer_of_data(data_matrix)
    ans = (a @ a.T) / (a.shape[1] - 1)
    delta = 0.0000001
    ans = ans + (delta * np.eye(ans.shape[0]))
    return ans

zeros_data = {"data":zeros , "mean" : mean_finder(zeros) , "covariance" : covariance_finder(zeros)}
ones_data = {"data":ones , "mean" : mean_finder(ones) , "covariance" : covariance_finder(ones)}
twos_data = {"data":twos , "mean" : mean_finder(twos) , "covariance" : covariance_finder(twos)}

zeros_test_data = {"data":zeros_test , "mean" : mean_finder(zeros_test) , "covariance" : covariance_finder(zeros_test)}
ones_test_data = {"data":ones_test , "mean" : mean_finder(ones_test) , "covariance" : covariance_finder(ones_test)}
twos_test_data = {"data":twos_test , "mean" : mean_finder(twos_test) , "covariance" : covariance_finder(twos_test)}

#################### 2b. Creating estimates on Gaussian

def likelihood_calc(x , mean, cov):
    x = [x]
    x = mean_finder(x)
    d = x.shape[0]
    det_sigma = np.linalg.det(cov)
    if det_sigma == 0:
        det_sigma = np.e ** (-25)
    inv_sigma = np.linalg.inv(cov)
    
    centralized_data = (x - mean).reshape(784,1)
    exponent_term = -0.5 * (centralized_data.T @ inv_sigma @ centralized_data)
    norm_term = 1
    #ans = norm_term * np.exp(exponent_term)
    ans = -0.5 * (np.log(det_sigma)) - 0.5 * (centralized_data.T @ inv_sigma @ centralized_data)
    #print(type(ans), ans.shape, "ans")
    return float(ans.item())

def likelihood_finder(x):
    prob_of_0 = likelihood_calc(x , zeros_data["mean"] , zeros_data["covariance"])
    prob_of_1 = likelihood_calc(x , ones_data["mean"] , ones_data["covariance"])
    prob_of_2 = likelihood_calc(x , twos_data["mean"] , twos_data["covariance"])
    
    t = max(prob_of_0 , prob_of_1 , prob_of_2)
    #print(prob_of_0 , prob_of_1, prob_of_2)
    if t == prob_of_1 :
        return (1, t)
    if t == prob_of_0:
        return (0, t)
    if t == prob_of_2:
        return (2, t)

#################### 2c. Accuracy of MLE

zero_score = [0 , 0 , 0]
one_score = [0 , 0 , 0]
two_score = [0 , 0 , 0]
print ( " ****************************** MLE [TEST] ****************************** \n")
for i in zeros_test:
    a = likelihood_finder(i)
    if a[0] != 0:
        zero_score[1] += 1
    else:
        zero_score[0] += 1
    zero_score[2] += 1
print("accuracy for class 0 = " , zero_score[0] / zero_score[2] )

for i in ones_test:
    a = likelihood_finder(i)
    if a[0] != 1:
        one_score[1] += 1
    else:
        one_score[0] += 1
    one_score[2] += 1
print("accuracy for class 1 = " , one_score[0] / one_score[2] )
    
for i in twos_test:
    a = likelihood_finder(i)
    if a[0] != 2:
        two_score[1] += 1
    else:
        two_score[0] += 1
    two_score[2] += 1
print("accuracy for class 2 = " , two_score[0] / two_score[2] )

print("overall accuracy of MLE = " , (zero_score[0] + one_score[0] + two_score[0]) / (zero_score[2] + one_score[2] + two_score[2]))
print ( " ****************************** MLE [TEST] ****************************** \n")

############################# 3. Dimensionality Reduction using PCA [95% of variance]
#################### 3a. Getting U_p matrix for PCA

data = zeros + ones + twos
data = np.vstack(data).T

mean = np.mean(data, axis=1, keepdims=True)

centered_data = data - mean

centered_data = data - mean

X_c = centered_data

S = (X_c @ X_c.T ) / (X_c.shape[1] - 1) 

np.seterr(all="ignore")

eigenvalues, eigenvectors = np.linalg.eig(S)
eigenvalues = eigenvalues.real.astype(float)
eigenvectors = eigenvectors.real.astype(float)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

total_variance = np.sum(eigenvalues)
    
p = 0
var_p = 0
while (var_p / total_variance) < 0.95:
    var_p += eigenvalues[p]
    p += 1
print ( " ****************************** PCA [DEMO] ****************************** \n")
print("p = ", p)

U_p = eigenvectors[:, :p]

#testing
y_test = U_p.T @ zeros_test[64]
print("y_test shape = ", y_test.shape)
reconstruct = (U_p @ y_test).reshape(784,1)
print(reconstruct.shape)
display_images([zeros_test[64] , (reconstruct + mean)] , ["original" , "reconstructed"])
print ( " ****************************** PCA [DEMO] ****************************** \n")

############################# 4. Class projection using FDA
#################### 4a. Computing Global Data Matrix and Classwise Mean

data = zeros + ones + twos
data = np.vstack(data).T

mean_0 = mean_finder(zeros)
mean_1 = mean_finder(ones)
mean_2 = mean_finder(twos)

final_mean = mean_finder(zeros + ones + twos)
print(final_mean.shape)

def Classwise_scatter(data_C , mean_C):
    start = np.zeros((784,784))
    for i in data_C:
        start = start + ((i - mean_C) @ (i - mean_C).T)
    return start

S_1 = Classwise_scatter(zeros , mean_0)
S_2 = Classwise_scatter(ones , mean_1)
S_3 = Classwise_scatter(twos , mean_2)

S_W = S_1 + S_2 + S_3

S_T = Classwise_scatter(zeros + ones + twos , final_mean)

def Between_Class_Scatter(mu, class_mu , corresponding_num):
    start = np.zeros((784,784))
    for i in range(len(class_mu)):
        value = corresponding_num[i] * ((class_mu[i] - mu) @ (class_mu[i] - mu).T)
        start = start + value
    return start

S_B = Between_Class_Scatter(final_mean, [mean_0, mean_1, mean_2] , [100,100,100])

#S_B = S_T - S_W

#################### 4b. Solving Generalized Eigenvalue Problem
eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)

sorted_indices = np.argsort(-eigvals)
W = eigvecs[:, sorted_indices[:2]]

data_LDA = W.T @ data

y_test = W.T @ zeros[30]

#################### 4c. Plotting Output
def plot_2d_points(list1, list2, list3):
    plt.figure(figsize=(8, 6))
    list1 = np.array(list1)
    list2 = np.array(list2)
    list3 = np.array(list3)

    if list1.size > 0:
        plt.scatter(list1[:, 0], list1[:, 1], c='cyan', label='Class 0', alpha=0.7, edgecolors='black')
    if list2.size > 0:
        plt.scatter(list2[:, 0], list2[:, 1], c='magenta', label='Class 1', alpha=0.7, edgecolors='black')
    if list3.size > 0:
        plt.scatter(list3[:, 0], list3[:, 1], c='yellow', label='Class 2', alpha=0.7, edgecolors='black')

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Scatter Plot of Three Classes")
    plt.legend()
    plt.grid(True)
    plt.show()

print ( " ****************************** FDA [DEMO] ****************************** \n")

zeros_test_LDA, ones_test_LDA , twos_test_LDA = [] , [] , []
for i in range(100):
    a = zeros_test[i]
    zeros_test_LDA.append((W.T @ a).real)
for i in range(100):
    a = ones_test[i]
    ones_test_LDA.append((W.T @ a).real)
for i in range(100):
    a = twos_test[i]
    twos_test_LDA.append((W.T @ a).real)
    
plot_2d_points(zeros_test_LDA, ones_test_LDA, twos_test_LDA)

print ( " ****************************** FDA [DEMO] ****************************** \n")

############################# 5. FDA then LDA (train + test)

FDA_data_zeros , FDA_data_ones , FDA_data_twos = [] , [] , []

for i in zeros:
    FDA_data_zeros.append((W.T @ i).real)
for i in ones:
    FDA_data_ones.append((W.T @ i).real)
for i in twos:
    FDA_data_twos.append((W.T @ i).real)

FDA_test_zeros , FDA_test_ones , FDA_test_twos = [] , [] , []

for i in zeros_test:
    FDA_test_zeros.append((W.T @ i).real)
for i in ones_test:
    FDA_test_ones.append((W.T @ i).real)
for i in twos_test:
    FDA_test_twos.append((W.T @ i).real)

mean_0 = mean_finder(FDA_data_zeros)
mean_1 = mean_finder(FDA_data_ones)
mean_2 = mean_finder(FDA_data_twos)

cov_0 = covariance_finder(FDA_data_zeros)
cov_1 = covariance_finder(FDA_data_ones)
cov_2 = covariance_finder(FDA_data_twos)

combined_cov = (cov_0 + cov_1 + cov_2) / 3

def DA(x, mean, cov):
    x = [x]
    x = mean_finder(x)
    d = x.shape[0]
    inv_sigma = np.linalg.inv(cov)
    centralized_data = (x - mean).reshape(d,1)
    exponent_term = -0.5 * (centralized_data.T @ inv_sigma @ centralized_data)
    return float(exponent_term.item())

def LDA_ans(x, mean_0, mean_1, mean_2, cov):
    prob_of_0 = DA(x , mean_0 , cov)
    prob_of_1 = DA(x , mean_1 , cov)
    prob_of_2 = DA(x , mean_2 , cov)
    
    t = max(prob_of_0 , prob_of_1 , prob_of_2)
    #print(prob_of_0 , prob_of_1, prob_of_2)
    if t == prob_of_1 :
        return (1, t)
    if t == prob_of_0:
        return (0, t)
    if t == prob_of_2:
        return (2, t)
    
zero_score_FDA_LDA = [0 , 0 , 0]
one_score_FDA_LDA = [0 , 0 , 0]
two_score_FDA_LDA = [0 , 0 , 0]
print ( " ****************************** FDA + LDA [TRAIN] ****************************** \n")
for i in FDA_data_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, combined_cov)
    if a[0] != 0:
        zero_score_FDA_LDA[1] += 1
    else:
        zero_score_FDA_LDA[0] += 1
    zero_score_FDA_LDA[2] += 1
print("accuracy for class 0 ON TRAIN = " , zero_score_FDA_LDA[0] / zero_score_FDA_LDA[2] )

for i in FDA_data_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, combined_cov)
    if a[0] != 1:
        one_score_FDA_LDA[1] += 1
    else:
        one_score_FDA_LDA[0] += 1
    one_score_FDA_LDA[2] += 1
print("accuracy for class 1 ON TRAIN = " , one_score_FDA_LDA[0] / one_score_FDA_LDA[2] )
    
for i in FDA_data_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, combined_cov)
    if a[0] != 2:
        two_score_FDA_LDA[1] += 1
    else:
        two_score_FDA_LDA[0] += 1
    two_score_FDA_LDA[2] += 1
print("accuracy for class 2 ON TRAIN = " , two_score_FDA_LDA[0] / two_score_FDA_LDA[2] )

print("overall accuracy of FDA then LDA ON TRAIN = " , (zero_score_FDA_LDA[0] + one_score_FDA_LDA[0] + two_score_FDA_LDA[0]) / (zero_score_FDA_LDA[2] + one_score_FDA_LDA[2] + two_score_FDA_LDA[2]))

print ( " ****************************** FDA + LDA [TRAIN] ****************************** \n")

print ( " ****************************** FDA + LDA [TEST] ****************************** \n")

zero_score_FDA_LDA = [0 , 0 , 0]
one_score_FDA_LDA = [0 , 0 , 0]
two_score_FDA_LDA = [0 , 0 , 0]

for i in FDA_test_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, combined_cov)
    if a[0] != 0:
        zero_score_FDA_LDA[1] += 1
    else:
        zero_score_FDA_LDA[0] += 1
    zero_score_FDA_LDA[2] += 1
print("accuracy for class 0 ON TEST = " , zero_score_FDA_LDA[0] / zero_score_FDA_LDA[2] )

for i in FDA_test_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, combined_cov)
    if a[0] != 1:
        one_score_FDA_LDA[1] += 1
    else:
        one_score_FDA_LDA[0] += 1
    one_score_FDA_LDA[2] += 1
print("accuracy for class 1 ON TEST = " , one_score_FDA_LDA[0] / one_score_FDA_LDA[2] )
    
for i in FDA_test_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, combined_cov)
    if a[0] != 2:
        two_score_FDA_LDA[1] += 1
    else:
        two_score_FDA_LDA[0] += 1
    two_score_FDA_LDA[2] += 1
print("accuracy for class 2 ON TEST = " , two_score_FDA_LDA[0] / two_score_FDA_LDA[2] )

print("overall accuracy of FDA then LDA ON TEST = " , (zero_score_FDA_LDA[0] + one_score_FDA_LDA[0] + two_score_FDA_LDA[0]) / (zero_score_FDA_LDA[2] + one_score_FDA_LDA[2] + two_score_FDA_LDA[2]))

print ( " ****************************** FDA + LDA [TEST] ****************************** \n")

############################# 6. FDA then QDA (train + test)

FDA_data_zeros , FDA_data_ones , FDA_data_twos = [] , [] , []

for i in zeros:
    FDA_data_zeros.append((W.T @ i).real)
for i in ones:
    FDA_data_ones.append((W.T @ i).real)
for i in twos:
    FDA_data_twos.append((W.T @ i).real)
    
    
FDA_test_zeros , FDA_test_ones , FDA_test_twos = [] , [] , []

for i in zeros_test:
    FDA_test_zeros.append((W.T @ i).real)
for i in ones_test:
    FDA_test_ones.append((W.T @ i).real)
for i in twos_test:
    FDA_test_twos.append((W.T @ i).real)

mean_0 = mean_finder(FDA_data_zeros)
mean_1 = mean_finder(FDA_data_ones)
mean_2 = mean_finder(FDA_data_twos)

cov_0 = covariance_finder(FDA_data_zeros)
cov_1 = covariance_finder(FDA_data_ones)
cov_2 = covariance_finder(FDA_data_twos)

def DA_complex(x , mean, cov):
    x = [x]
    x = mean_finder(x)
    d = x.shape[0]
    det_sigma = np.linalg.det(cov)
    if det_sigma == 0:
        det_sigma = np.e ** (-25)
    inv_sigma = np.linalg.inv(cov)
    centralized_data = (x - mean).reshape(d,1)
    exponent_term = -0.5 * (centralized_data.T @ inv_sigma @ centralized_data)
    ans = -0.5 * (np.log(det_sigma)) - 0.5 * (centralized_data.T @ inv_sigma @ centralized_data)
    return float(ans.item())

def QDA_ans(x, mean_0, mean_1, mean_2, cov_0, cov_1, cov_2):
    prob_of_0 = DA_complex(x , mean_0 , cov_0)
    prob_of_1 = DA_complex(x , mean_1 , cov_1)
    prob_of_2 = DA_complex(x , mean_2 , cov_2)
    
    t = max(prob_of_0 , prob_of_1 , prob_of_2)
    #print(prob_of_0 , prob_of_1, prob_of_2)
    if t == prob_of_1 :
        return (1, t)
    if t == prob_of_0:
        return (0, t)
    if t == prob_of_2:
        return (2, t)

print ( " ****************************** FDA + QDA [TRAIN] ****************************** \n")

zero_score_FDA_QDA = [0 , 0 , 0]
one_score_FDA_QDA = [0 , 0 , 0]
two_score_FDA_QDA = [0 , 0 , 0]

for i in FDA_data_zeros:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 0:
        zero_score_FDA_QDA[1] += 1
    else:
        zero_score_FDA_QDA[0] += 1
    zero_score_FDA_QDA[2] += 1
print("accuracy for class 0 ON TRAIN = " , zero_score_FDA_QDA[0] / zero_score_FDA_QDA[2] )

for i in FDA_data_ones:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 1:
        one_score_FDA_QDA[1] += 1
    else:
        one_score_FDA_QDA[0] += 1
    one_score_FDA_QDA[2] += 1
print("accuracy for class 1 ON TRAIN = " , one_score_FDA_QDA[0] / one_score_FDA_QDA[2] )
    
for i in FDA_data_twos:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 2:
        two_score_FDA_QDA[1] += 1
    else:
        two_score_FDA_QDA[0] += 1
    two_score_FDA_QDA[2] += 1
print("accuracy for class 2 ON TRAIN = " , two_score_FDA_QDA[0] / two_score_FDA_QDA[2] )

print("overall accuracy of FDA then QDA ON TRAIN = " , (zero_score_FDA_QDA[0] + one_score_FDA_QDA[0] + two_score_FDA_QDA[0]) / (zero_score_FDA_QDA[2] + one_score_FDA_QDA[2] + two_score_FDA_QDA[2]))

print ( " ****************************** FDA + QDA [TRAIN] ****************************** \n")

print ( " ****************************** FDA + QDA [TEST] ****************************** \n")

zero_score_FDA_QDA = [0 , 0 , 0]
one_score_FDA_QDA = [0 , 0 , 0]
two_score_FDA_QDA = [0 , 0 , 0]

for i in FDA_test_zeros:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 0:
        zero_score_FDA_QDA[1] += 1
    else:
        zero_score_FDA_QDA[0] += 1
    zero_score_FDA_QDA[2] += 1
print("accuracy for class 0 ON TEST = " , zero_score_FDA_QDA[0] / zero_score_FDA_QDA[2] )

for i in FDA_test_ones:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 1:
        one_score_FDA_QDA[1] += 1
    else:
        one_score_FDA_QDA[0] += 1
    one_score_FDA_QDA[2] += 1
print("accuracy for class 1 ON TEST = " , one_score_FDA_QDA[0] / one_score_FDA_QDA[2] )
    
for i in FDA_test_twos:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 2:
        two_score_FDA_QDA[1] += 1
    else:
        two_score_FDA_QDA[0] += 1
    two_score_FDA_QDA[2] += 1
print("accuracy for class 2 ON TEST = " , two_score_FDA_QDA[0] / two_score_FDA_QDA[2] )

print("overall accuracy of FDA then QDA ON TEST = " , (zero_score_FDA_QDA[0] + one_score_FDA_QDA[0] + two_score_FDA_QDA[0]) / (zero_score_FDA_QDA[2] + one_score_FDA_QDA[2] + two_score_FDA_QDA[2]))

print ( " ****************************** FDA + QDA [TEST] ****************************** \n")

############################# 7. PCA then QDA (train + test)

PCA_data_zeros , PCA_data_ones , PCA_data_twos = [] , [] , []
PCA_test_zeros , PCA_test_ones , PCA_test_twos = [] , [] , []

for i in zeros:
    PCA_data_zeros.append(U_p.T @ i)
for i in ones:
    PCA_data_ones.append(U_p.T @ i)
for i in twos:
    PCA_data_twos.append(U_p.T @ i)
    
for i in zeros_test:
    PCA_test_zeros.append(U_p.T @ i)
for i in ones_test:
    PCA_test_ones.append(U_p.T @ i)
for i in twos_test:
    PCA_test_twos.append(U_p.T @ i)

mean_0 = mean_finder(PCA_data_zeros)
mean_1 = mean_finder(PCA_data_ones)
mean_2 = mean_finder(PCA_data_twos)

cov_0 = covariance_finder(PCA_data_zeros)
cov_1 = covariance_finder(PCA_data_ones)
cov_2 = covariance_finder(PCA_data_twos)

print ( " ****************************** PCA + QDA [TRAIN] ****************************** \n")
zero_score_PCA_QDA = [0 , 0 , 0]
one_score_PCA_QDA = [0 , 0 , 0]
two_score_PCA_QDA = [0 , 0 , 0]

for i in PCA_data_zeros:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 0:
        zero_score_PCA_QDA[1] += 1
    else:
        zero_score_PCA_QDA[0] += 1
    zero_score_PCA_QDA[2] += 1
print("accuracy for class 0 ON TRAIN = " , zero_score_PCA_QDA[0] / zero_score_PCA_QDA[2] )

for i in PCA_data_ones:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 1:
        one_score_PCA_QDA[1] += 1
    else:
        one_score_PCA_QDA[0] += 1
    one_score_PCA_QDA[2] += 1
print("accuracy for class 1 ON TRAIN = " , one_score_PCA_QDA[0] / one_score_PCA_QDA[2] )
    
for i in PCA_data_twos:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 2:
        two_score_PCA_QDA[1] += 1
    else:
        two_score_PCA_QDA[0] += 1
    two_score_PCA_QDA[2] += 1
print("accuracy for class 2 ON TRAIN = " , two_score_PCA_QDA[0] / two_score_PCA_QDA[2] )

print("overall accuracy of PCA then QDA ON TRAIN = " , (zero_score_PCA_QDA[0] + one_score_PCA_QDA[0] + two_score_PCA_QDA[0]) / (zero_score_PCA_QDA[2] + one_score_PCA_QDA[2] + two_score_PCA_QDA[2]))
print ( " ****************************** PCA + QDA [TRAIN] ****************************** \n")

print ( " ****************************** PCA + QDA [TEST] ****************************** \n")
zero_score_PCA_QDA = [0 , 0 , 0]
one_score_PCA_QDA = [0 , 0 , 0]
two_score_PCA_QDA = [0 , 0 , 0]

for i in PCA_test_zeros:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 0:
        zero_score_PCA_QDA[1] += 1
    else:
        zero_score_PCA_QDA[0] += 1
    zero_score_PCA_QDA[2] += 1
print("accuracy for class 0 ON TEST = " , zero_score_PCA_QDA[0] / zero_score_PCA_QDA[2] )

for i in PCA_test_ones:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 1:
        one_score_PCA_QDA[1] += 1
    else:
        one_score_PCA_QDA[0] += 1
    one_score_PCA_QDA[2] += 1
print("accuracy for class 1 ON TEST = " , one_score_PCA_QDA[0] / one_score_PCA_QDA[2] )
    
for i in PCA_test_twos:
    a = QDA_ans(i , mean_0, mean_1, mean_2, cov_0, cov_1, cov_2)
    if a[0] != 2:
        two_score_PCA_QDA[1] += 1
    else:
        two_score_PCA_QDA[0] += 1
    two_score_PCA_QDA[2] += 1
print("accuracy for class 2 ON TEST = " , two_score_PCA_QDA[0] / two_score_PCA_QDA[2] )

print("overall accuracy of PCA then QDA ON TEST = " , (zero_score_PCA_QDA[0] + one_score_PCA_QDA[0] + two_score_PCA_QDA[0]) / (zero_score_PCA_QDA[2] + one_score_PCA_QDA[2] + two_score_PCA_QDA[2]))
print ( " ****************************** PCA + QDA [TEST] ****************************** \n")

############################# 8. PCA then LDA (train + test)

PCA_data_zeros , PCA_data_ones , PCA_data_twos = [] , [] , []
PCA_test_zeros , PCA_test_ones , PCA_test_twos = [] , [] , []

for i in zeros:
    PCA_data_zeros.append(U_p.T @ i)
for i in ones:
    PCA_data_ones.append(U_p.T @ i)
for i in twos:
    PCA_data_twos.append(U_p.T @ i)
    
for i in zeros_test:
    PCA_test_zeros.append(U_p.T @ i)
for i in ones_test:
    PCA_test_ones.append(U_p.T @ i)
for i in twos_test:
    PCA_test_twos.append(U_p.T @ i)

mean_0 = mean_finder(PCA_data_zeros)
mean_1 = mean_finder(PCA_data_ones)
mean_2 = mean_finder(PCA_data_twos)

cov_0 = covariance_finder(PCA_data_zeros)
cov_1 = covariance_finder(PCA_data_ones)
cov_2 = covariance_finder(PCA_data_twos)

covariance_final = (cov_0 + cov_1 + cov_2) / 3

print ( " ****************************** PCA + LDA [TRAIN] ****************************** \n")
zero_score_PCA_LDA = [0 , 0 , 0]
one_score_PCA_LDA = [0 , 0 , 0]
two_score_PCA_LDA = [0 , 0 , 0]

for i in PCA_data_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 0:
        zero_score_PCA_LDA[1] += 1
    else:
        zero_score_PCA_LDA[0] += 1
    zero_score_PCA_LDA[2] += 1
print("accuracy for class 0 ON TRAIN = " , zero_score_PCA_LDA[0] / zero_score_PCA_LDA[2] )

for i in PCA_data_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 1:
        one_score_PCA_LDA[1] += 1
    else:
        one_score_PCA_LDA[0] += 1
    one_score_PCA_LDA[2] += 1
print("accuracy for class 1 ON TRAIN = " , one_score_PCA_LDA[0] / one_score_PCA_LDA[2] )
    
for i in PCA_data_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 2:
        two_score_PCA_LDA[1] += 1
    else:
        two_score_PCA_LDA[0] += 1
    two_score_PCA_LDA[2] += 1
print("accuracy for class 2 ON TRAIN = " , two_score_PCA_LDA[0] / two_score_PCA_LDA[2] )

print("overall accuracy of PCA then LDA ON TRAIN = " , (zero_score_PCA_LDA[0] + one_score_PCA_LDA[0] + two_score_PCA_LDA[0]) / (zero_score_PCA_LDA[2] + one_score_PCA_LDA[2] + two_score_PCA_LDA[2]))
print ( " ****************************** PCA + LDA [TRAIN] ****************************** \n")

print ( " ****************************** PCA + LDA [TEST] ****************************** \n")
zero_score_PCA_LDA = [0 , 0 , 0]
one_score_PCA_LDA = [0 , 0 , 0]
two_score_PCA_LDA = [0 , 0 , 0]

for i in PCA_test_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 0:
        zero_score_PCA_LDA[1] += 1
    else:
        zero_score_PCA_LDA[0] += 1
    zero_score_PCA_LDA[2] += 1
print("accuracy for class 0 ON TEST = " , zero_score_PCA_LDA[0] / zero_score_PCA_LDA[2] )

for i in PCA_test_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 1:
        one_score_PCA_LDA[1] += 1
    else:
        one_score_PCA_LDA[0] += 1
    one_score_PCA_LDA[2] += 1
print("accuracy for class 1 ON TEST = " , one_score_PCA_LDA[0] / one_score_PCA_LDA[2] )
    
for i in PCA_test_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 2:
        two_score_PCA_LDA[1] += 1
    else:
        two_score_PCA_LDA[0] += 1
    two_score_PCA_LDA[2] += 1
print("accuracy for class 2 ON TEST = " , two_score_PCA_LDA[0] / two_score_PCA_LDA[2] )

print("overall accuracy of PCA then LDA ON TEST = " , (zero_score_PCA_LDA[0] + one_score_PCA_LDA[0] + two_score_PCA_LDA[0]) / (zero_score_PCA_LDA[2] + one_score_PCA_LDA[2] + two_score_PCA_LDA[2]))
print ( " ****************************** PCA + LDA [TEST] ****************************** \n")

############################# 9. PCA (90%) then LDA (train + test)
data = zeros + ones + twos
data = np.vstack(data).T

mean = np.mean(data, axis=1, keepdims=True)

centered_data = data - mean

centered_data = data - mean

X_c = centered_data

S = (X_c @ X_c.T ) / (X_c.shape[1] - 1) 

np.seterr(all="ignore")

eigenvalues, eigenvectors = np.linalg.eig(S)
eigenvalues = eigenvalues.real.astype(float)
eigenvectors = eigenvectors.real.astype(float)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

total_variance = np.sum(eigenvalues)
    
p = 0
var_p = 0
while (var_p / total_variance) < 0.9:
    var_p += eigenvalues[p]
    p += 1

print ( " ****************************** PCA (90%) [DEMO] ****************************** \n")
print("p = ", p)

U9_p = eigenvectors[:, :p]

#testing
y_test = U9_p.T @ ones[38]
print("y_test shape = ", y_test.shape)
reconstruct = (U9_p @ y_test).reshape(784,1)
print(reconstruct.shape)
display_images([ones[38] , (reconstruct + mean)] , ["original" , "reconstructed"])
print ( " ****************************** PCA (90%) [DEMO] ****************************** \n")

PCA9_data_zeros , PCA9_data_ones , PCA9_data_twos = [] , [] , []
PCA9_test_zeros , PCA9_test_ones , PCA9_test_twos = [] , [] , []

for i in zeros:
    PCA9_data_zeros.append(U9_p.T @ i)
for i in ones:
    PCA9_data_ones.append(U9_p.T @ i)
for i in twos:
    PCA9_data_twos.append(U9_p.T @ i)
    
for i in zeros_test:
    PCA9_test_zeros.append(U9_p.T @ i)
for i in ones_test:
    PCA9_test_ones.append(U9_p.T @ i)
for i in twos_test:
    PCA9_test_twos.append(U9_p.T @ i)

mean_0 = mean_finder(PCA9_data_zeros)
mean_1 = mean_finder(PCA9_data_ones)
mean_2 = mean_finder(PCA9_data_twos)

cov_0 = covariance_finder(PCA9_data_zeros)
cov_1 = covariance_finder(PCA9_data_ones)
cov_2 = covariance_finder(PCA9_data_twos)

covariance_final = (cov_0 + cov_1 + cov_2) / 3

print ( " ****************************** PCA (90%) + LDA [TRAIN] ****************************** \n")
zero_score_PCA9_LDA = [0 , 0 , 0]
one_score_PCA9_LDA = [0 , 0 , 0]
two_score_PCA9_LDA = [0 , 0 , 0]

for i in PCA9_data_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 0:
        zero_score_PCA9_LDA[1] += 1
    else:
        zero_score_PCA9_LDA[0] += 1
    zero_score_PCA9_LDA[2] += 1
print("accuracy for class 0 ON TRAIN = " , zero_score_PCA9_LDA[0] / zero_score_PCA9_LDA[2] )

for i in PCA9_data_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 1:
        one_score_PCA9_LDA[1] += 1
    else:
        one_score_PCA9_LDA[0] += 1
    one_score_PCA9_LDA[2] += 1
print("accuracy for class 1 ON TRAIN = " , one_score_PCA9_LDA[0] / one_score_PCA9_LDA[2] )
    
for i in PCA9_data_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 2:
        two_score_PCA9_LDA[1] += 1
    else:
        two_score_PCA9_LDA[0] += 1
    two_score_PCA9_LDA[2] += 1
print("accuracy for class 2 ON TRAIN = " , two_score_PCA9_LDA[0] / two_score_PCA9_LDA[2] )

print("overall accuracy of PCA(90) then LDA ON TRAIN = " , (zero_score_PCA9_LDA[0] + one_score_PCA9_LDA[0] + two_score_PCA9_LDA[0]) / (zero_score_PCA9_LDA[2] + one_score_PCA9_LDA[2] + two_score_PCA9_LDA[2]))
print ( " ****************************** PCA (90%) + LDA [TRAIN] ****************************** \n")

print ( " ****************************** PCA (90%) + LDA [TEST] ****************************** \n")
zero_score_PCA9_LDA = [0 , 0 , 0]
one_score_PCA9_LDA = [0 , 0 , 0]
two_score_PCA9_LDA = [0 , 0 , 0]

for i in PCA9_test_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 0:
        zero_score_PCA9_LDA[1] += 1
    else:
        zero_score_PCA9_LDA[0] += 1
    zero_score_PCA9_LDA[2] += 1
print("accuracy for class 0 ON TEST = " , zero_score_PCA9_LDA[0] / zero_score_PCA9_LDA[2] )

for i in PCA9_test_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 1:
        one_score_PCA9_LDA[1] += 1
    else:
        one_score_PCA9_LDA[0] += 1
    one_score_PCA9_LDA[2] += 1
print("accuracy for class 1 ON TEST = " , one_score_PCA9_LDA[0] / one_score_PCA9_LDA[2] )
    
for i in PCA9_test_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 2:
        two_score_PCA9_LDA[1] += 1
    else:
        two_score_PCA9_LDA[0] += 1
    two_score_PCA9_LDA[2] += 1
print("accuracy for class 2 ON TEST = " , two_score_PCA9_LDA[0] / two_score_PCA9_LDA[2] )

print("overall accuracy of PCA(90) then LDA ON TEST = " , (zero_score_PCA9_LDA[0] + one_score_PCA9_LDA[0] + two_score_PCA9_LDA[0]) / (zero_score_PCA9_LDA[2] + one_score_PCA9_LDA[2] + two_score_PCA9_LDA[2]))
print ( " ****************************** PCA (90%) + LDA [TEST] ****************************** \n")

############################# 10. PCA (2d) then LDA (train + test)
data = zeros + ones + twos
data = np.vstack(data).T

mean = np.mean(data, axis=1, keepdims=True)

centered_data = data - mean

centered_data = data - mean

X_c = centered_data

S = (X_c @ X_c.T ) / (X_c.shape[1] - 1) 

np.seterr(all="ignore")

eigenvalues, eigenvectors = np.linalg.eig(S)
eigenvalues = eigenvalues.real.astype(float)
eigenvectors = eigenvectors.real.astype(float)
#eigenvalues.sort()
#eigenvalues = eigenvalues[::-1]

#eigenvalues , eigenvectors = zip(*sorted(zip(eigenvalues, eigenvectors), key=lambda x: -x[0]))
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]  # Ensure correct column order

total_variance = np.sum(eigenvalues)
    
p = 2

print ( " ****************************** PCA (2D) [DEMO] ****************************** \n")
print("p = ", p)

U2_p = eigenvectors[:, :p]

zeros_test_PCA, ones_test_PCA , twos_test_PCA = [] , [] , []
for i in range(100):
    a = zeros_test[i]
    zeros_test_PCA.append((U2_p.T  @ a).real)
for i in range(100):
    a = ones_test[i]
    ones_test_PCA.append((U2_p.T  @ a).real)
for i in range(100):
    a = twos_test[i]
    twos_test_PCA.append((U2_p.T  @ a).real)
    
plot_2d_points(zeros_test_PCA, ones_test_PCA, twos_test_PCA)

#testing
y_test = U2_p.T @ zeros[10]
print("y_test shape = ", y_test.shape)
reconstruct = (U2_p @ y_test).reshape(784,1)
print(reconstruct.shape)
display_images([zeros[10] , (reconstruct + mean)] , ["original" , "reconstructed"])
print ( " ****************************** PCA (2D) [DEMO] ****************************** \n")

PCA2_data_zeros , PCA2_data_ones , PCA2_data_twos = [] , [] , []
PCA2_test_zeros , PCA2_test_ones , PCA2_test_twos = [] , [] , []

for i in zeros:
    PCA2_data_zeros.append(U2_p.T @ i)
for i in ones:
    PCA2_data_ones.append(U2_p.T @ i)
for i in twos:
    PCA2_data_twos.append(U2_p.T @ i)
    
for i in zeros_test:
    PCA2_test_zeros.append(U2_p.T @ i)
for i in ones_test:
    PCA2_test_ones.append(U2_p.T @ i)
for i in twos_test:
    PCA2_test_twos.append(U2_p.T @ i)

mean_0 = mean_finder(PCA2_data_zeros)
mean_1 = mean_finder(PCA2_data_ones)
mean_2 = mean_finder(PCA2_data_twos)

cov_0 = covariance_finder(PCA2_data_zeros)
cov_1 = covariance_finder(PCA2_data_ones)
cov_2 = covariance_finder(PCA2_data_twos)

covariance_final = (cov_0 + cov_1 + cov_2) / 3

print ( " ****************************** PCA (2D) + LDA [TRAIN] ****************************** \n")
zero_score_PCA2_LDA = [0 , 0 , 0]
one_score_PCA2_LDA = [0 , 0 , 0]
two_score_PCA2_LDA = [0 , 0 , 0]

for i in PCA2_data_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 0:
        zero_score_PCA2_LDA[1] += 1
    else:
        zero_score_PCA2_LDA[0] += 1
    zero_score_PCA2_LDA[2] += 1
print("accuracy for class 0 ON TRAIN = " , zero_score_PCA2_LDA[0] / zero_score_PCA2_LDA[2] )

for i in PCA2_data_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 1:
        one_score_PCA2_LDA[1] += 1
    else:
        one_score_PCA2_LDA[0] += 1
    one_score_PCA2_LDA[2] += 1
print("accuracy for class 1 ON TRAIN = " , one_score_PCA2_LDA[0] / one_score_PCA2_LDA[2] )
    
for i in PCA2_data_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 2:
        two_score_PCA2_LDA[1] += 1
    else:
        two_score_PCA2_LDA[0] += 1
    two_score_PCA2_LDA[2] += 1
print("accuracy for class 2 ON TRAIN = " , two_score_PCA2_LDA[0] / two_score_PCA2_LDA[2] )

print("overall accuracy of PCA(2 class) then LDA ON TRAIN = " , (zero_score_PCA2_LDA[0] + one_score_PCA2_LDA[0] + two_score_PCA2_LDA[0]) / (zero_score_PCA2_LDA[2] + one_score_PCA2_LDA[2] + two_score_PCA2_LDA[2]))

print ( " ****************************** PCA (2D) + LDA [TRAIN] ****************************** \n")

print ( " ****************************** PCA (2D) + LDA [TEST] ****************************** \n")
zero_score_PCA2_LDA = [0 , 0 , 0]
one_score_PCA2_LDA = [0 , 0 , 0]
two_score_PCA2_LDA = [0 , 0 , 0]

for i in PCA2_test_zeros:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 0:
        zero_score_PCA2_LDA[1] += 1
    else:
        zero_score_PCA2_LDA[0] += 1
    zero_score_PCA2_LDA[2] += 1
print("accuracy for class 0 ON TEST = " , zero_score_PCA2_LDA[0] / zero_score_PCA2_LDA[2] )

for i in PCA2_test_ones:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 1:
        one_score_PCA2_LDA[1] += 1
    else:
        one_score_PCA2_LDA[0] += 1
    one_score_PCA2_LDA[2] += 1
print("accuracy for class 1 ON TEST = " , one_score_PCA2_LDA[0] / one_score_PCA2_LDA[2] )
    
for i in PCA2_test_twos:
    a = LDA_ans(i , mean_0, mean_1, mean_2, covariance_final)
    if a[0] != 2:
        two_score_PCA2_LDA[1] += 1
    else:
        two_score_PCA2_LDA[0] += 1
    two_score_PCA2_LDA[2] += 1
print("accuracy for class 2 ON TEST = " , two_score_PCA2_LDA[0] / two_score_PCA2_LDA[2] )

print("overall accuracy of PCA(2 class) then LDA ON TEST = " , (zero_score_PCA2_LDA[0] + one_score_PCA2_LDA[0] + two_score_PCA2_LDA[0]) / (zero_score_PCA2_LDA[2] + one_score_PCA2_LDA[2] + two_score_PCA2_LDA[2]))

print ( " ****************************** PCA (2D) + LDA [TEST] ****************************** \n")
