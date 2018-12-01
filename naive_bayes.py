import os, sys, glob
import numpy as np
import PIL.Image as Image

##reading test and train file from command line
train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
if not os.path.exists(train_file_name):
	print("not exists file " + os.path.basename(train_file_name))
if not os.path.exists(test_file_name):
	print("not exists file " + os.path.basename(test_file_name))

train_file = open(train_file_name, 'r') 
paths = train_file.readlines();

for i in range(0, len(paths)):
    paths[i] = paths[i].rstrip()

train_paths = []
for i in paths:
    train_paths.append([i.split(' ')[0], i.split(' ')[1]])

X = []
train_data = []
for i in train_paths:
    image = np.array(Image.open(i[0]).convert('L'), dtype='uint8')
    X.append(image)
    image = np.reshape(image, (256*256, 1))
    train_data.append([image, i[1]])

test_file = open(test_file_name, 'r') 
paths = test_file.readlines();

for i in range(0, len(paths)):
    paths[i] = paths[i].rstrip()
    
test_data = []

for i in paths:
    image = np.array(Image.open(i).convert('L'), dtype='uint8')
    image = np.reshape(image, (256*256, 1))
    test_data.append(image)
##----------------------------------------------
##----------------------------------------------
### getting eigen faces and weights by PCA
def read_images(path, sz=None):
    X = []
    for files in glob.glob(path + "/*.jpg"):
        image = np.array(Image.open(files).convert('L'), dtype='uint8')
        X.append(image)
    return X

def normalize(X, low, high):
    X = np.array(X)
    min_X, max_X = np.min(X), np.max(X)
    X = X - float(min_X)
    X = X / float((max_X - min_X))
    X = X * (high - low)
    X = X + low
    return np.array(X)

def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = []
    for i in range(0, len(X)):
        mat.append(np.reshape(X[i], (X[0].shape[0] * X[0].shape[0], )))
    return np.array(mat)

def pca(X):
    [n,d] = X.shape
    mu = X.mean(axis=0)
    X = X - mu
    if n>d:
        C = np.dot(X.T,X)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X,X.T)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T,eigenvectors)
        for i in range(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
            
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    return [eigenvalues, eigenvectors, mu]

def project(W, X, mu=None):
    if mu is None:
        return np.dot(X,W)
    return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu

def weight_img(image, components):
    numEvs = components
    P = project(eig_vecs[:,0:numEvs], image.reshape(1,-1), mu)
    return P

def reconstruct_img(image):
    steps = [i for i in range(10, min(len(X), 320), 20)]
    E = []
    for i in range(len(steps)):
        numEvs = steps[i]
        P = project(eig_vecs[:,0:numEvs], image.reshape(1,-1), mu)
        R = reconstruct(eig_vecs[:,0:numEvs], P, mu)
        # reshape and append
        R = R.reshape(image.shape)
        E.append(normalize(R,0,255))
    return E, steps

def MSE(X, Y):
    m, n = X.shape
    error = 0
    for i in range(m):
        for j in range(n):
            error += (X[i, j] - Y[i, j])**2
    error /= (m * n)
    return error
###----------------------------------------------
###----------------------------------------------
##getting eigenfaces from dataset
[eig_val, eig_vecs, mu] = pca(asRowMatrix(X))
components = 32
Eig_face = []
for i in range(min(len(X), components)):
    e = eig_vecs[:,i].reshape(X[0].shape)
    Eig_face.append(normalize(e, 0, 255))

train_data_X = []
train_data_Y = []
for i in range(0, len(train_data)):
    train_data_X.append(weight_img(train_data[i][0], components).flatten())
    train_data_Y.append(train_data[i][1])

train_data_X = np.array(train_data_X)
train_data_Y = np.array(train_data_Y)

test_data_X = []
for i in range(0, len(test_data)):
    test_data_X.append(weight_img(test_data[i], components).flatten())

test_data_X = np.array(test_data_X)
##-------------------------------------------------
##-------------------------------------------------

##Naive Bayes Classfier
def separate_By_Class(data_x, data_y):
    separated_dict = {}
    for i in range(len(data_x)):
        key = data_x[i]
        if (data_y[i] not in separated_dict):
            separated_dict[data_y[i]] = []
        separated_dict[data_y[i]].append(key)
    return separated_dict

def mean(numbers):
    num = sum(numbers)
    deno = float(len(numbers))
    return num/deno

def stdev(numbers):
    avg = mean(numbers)
    var_num = sum([pow(x-avg,2) for x in numbers])
    var_deno = float(len(numbers)-1)
    variance = var_num / var_deno
    return np.sqrt(variance)

def get_distribution(data_x, data_y):
    distribution = [(mean(attribute), stdev(attribute)) for attribute in zip(*data_x)]
    return distribution

def distribution_by_class(data_x, data_y):
    separated = separate_By_Class(data_x, data_y)
    distributions = {}
    for classes, keys in separated.items():
        distributions[classes] = get_distribution(keys, data_y)
    return distributions

def calculate_Probability(x, mean, stdev):
    exponent = np.exp(-(pow(x-mean,2)/(2*pow(stdev,2))))
    return np.log((1 / (np.sqrt(2*np.pi) * stdev)) * exponent)

def calculate_Class_Probabilities(distributions, X):
    probabilities = {}
    for Value, dist in distributions.items():
        probabilities[Value] = 1
        for i in range(len(dist)):
            mean, stdev = dist[i]
            x = X[i]
            probabilities[Value] += calculate_Probability(x, mean, stdev)
    return probabilities

def predict(distributions, X):
    probabilities = calculate_Class_Probabilities(distributions, X)
    bestLabel, bestProb = None, -100
    for label, probability in probabilities.items():
        if probability > bestProb or bestLabel is None:
            bestProb = probability
            bestLabel = label
    return bestLabel

def get_Predictions(distributions, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(distributions, testSet[i])
        predictions.append(result)
    return predictions

def classifier(X_train, Y_train, X_test):
    summary = distribution_by_class(X_train, Y_train)
    prediction = get_Predictions(summary, X_test)
    return prediction
##--------------------------------------------------------------
##--------------------------------------------------------------

p = classifier(train_data_X, train_data_Y, test_data_X)
for i in range(0, len(p)):
    print(p[i])
