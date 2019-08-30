# library
library(e1071)
library(glasso)
library(SparseM)
library(R.matlab)

# read data
data = read.matrix.csr('D:\\Proximal average\\data\\graph_guided_logistic_regression\\cod-rna')
# data = read.csv('D:\\Proximal average\\data\\graph_guided_logistic_regression\\sido\\sido2_train.data')
# data = readMat('D:\\Proximal average\\data\\graph_guided_logistic_regression\\sido2_matlab\\sido2_train.mat')
x_train = data$x
# y_train = data$y
x_train = as.matrix(x_train)


# scale
range01 <- function(x){
  k = 2/(max(x) - min(x))
  range01 = -1 + k * (x - min(x))
}
x_train[, 1] = range01(x_train[, 1])
x_train[, 2] = range01(x_train[, 2])

# Estimate covariance matrix
cov_x = cov(x_train)

# Get the matrix
E = glasso(cov_x, rho = 0.002)
W = E$wi

