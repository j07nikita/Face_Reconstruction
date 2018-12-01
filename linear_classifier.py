import numpy as np
import os, sys, glob
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
    
train_data = []
X = []
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

def project(W, X, mu = None):
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

###-----------------------------------------------
##-------------------------------------------------

###Linear classifier : Softmax
def getLoss(w,x,y,lam):
    m = x.shape[0] 
    y_mat = oneHotIt(y) 
    scores = np.dot(x,w)
    prob = softmax(scores) #perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return loss, grad

def create_lable(Y):
    l_dict = dict(zip(set(Y), range(len(Y))))
    return l_dict, np.array([l_dict[x] for x in Y])

def oneHotIt(Y):
    n_values = np.max(Y) + 1
    return np.eye(n_values)[Y] 

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def get_value_from_lable(p, l_dict):
	t = []
	for i in p:
		for k, v in l_dict.items():
			if v == i:
				t.append(k)
	return t
##----------------------------------------------------------------------
##---------------------------------------------------------------------

x = train_data_X
l_dict, y = create_lable(train_data_Y)
w = np.zeros([x.shape[1],len(np.unique(y))])
lam = 1
iterations = 100
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,x,y,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
#print("loss - ", loss)
prob, prede = getProbsAndPreds(test_data_X)
class_prede = get_value_from_lable(prede, l_dict)
for i in range(0, len(class_prede)):
    print(class_prede[i])
