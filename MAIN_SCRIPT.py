import numpy as np
import pandas as pd
from tqdm import tqdm
from cvxopt import solvers
from cvxopt import matrix
from cvxopt import spmatrix
from cvxopt import sparse
from collections import deque
#################

"""
Group: Kernels Samplers ( Francis Anokye & Aissatou Ndoye)
Data Challenge :  kernel methods in Machine learning
Professors: Jean Phillpe Vert & Julien Mairal
TA: Romain Menegaux 

Description:
-------------
This main script file implements a kernel SVM which uses vanilla k-spectrum feature embedding and 
(k,m)-mismatch embedding as a multi[le kernel. This multiple kernel handles all the operations (gram matrix computation, 
function evaluation) and then after we predict on the test data and generate our submission file. 
"""
############################################# KERNELS ##############################################
class Kernel():
       
    def gaussian(sigma):
        return lambda x, y : 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-np.linalg.norm(x - y)**2/(2*sigma**2))
    
    def linear():
        return lambda x, y: np.dot(x, y)
    
    def polynomial(c, n):
        return lambda x, y : (np.dot(x, y) + c)**n
    
    def spectrum():
        def f(x, y):
            prod_scal = 0
            for kmer in x:
                if kmer in y:
                    prod_scal += x[kmer]*y[kmer]
            return prod_scal
        return f
    
    def mismatch():
        def f(x, y):
            prod_scal = 0
            for idx in x:
                if idx in y:
                    prod_scal += x[idx]*y[idx]
            return prod_scal
        return f
    
    def sparse_gaussian(sigma):
        def f(x, y):
            ps = Kernel.mismatch()
            norm = ps(x, x) - 2*ps(x, y) + ps(y,y)
            return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-norm/(2*sigma**2))
        return f
    
    def sparse_poly(c, n):
        def f(x, y):
            ps = Kernel.mismatch()
            return (ps(x,y) + c)**n
        return f
    
    def __init__(self, func, normalized = False):
        self.kernel = func
        self.normalized = normalized
        self.diag = np.array([])
        
    def gram(self, data):
        n = len(data)
        K = np.zeros((n, n))
        print("Computing Gram Matrix")
        for i in tqdm(range(n)):
            for j in range(i+1):
                prod_scal = self.kernel(data[i], data[j])
                K[i, j] = prod_scal
                K[j, i] = prod_scal
        
        if self.normalized:
            self.diag = np.sqrt(np.diag(K))
            print(self.diag.shape)
            for i in range(n):
                K[i, :] = K[i,:]/self.diag[i]
                K[:, i] = K[:, i]/self.diag[i]
        return K
    
    def eval_f(self, x, alpha, data):
        if self.normalized:
            square_norm_x = np.sqrt(self.kernel(x, x))
            result = np.sum([(alpha[i]*self.kernel(x, xi))/(square_norm_x * self.diag[i]) for i, xi in enumerate(data)])
        else:
            result =  np.sum([alpha[i]*self.kernel(x, xi) for i, xi in enumerate(data)])
        return result 

###################################### Kernel_SVM #################################################
class Kernel_SVM():
    
    def SVM(K, y, lmda):
        print("Optimizing")
        solvers.options['show_progress'] = False
        n = len(y)
        q = -matrix(y, (n, 1), tc='d')
        h = matrix(np.concatenate([np.ones(n)/(2*lmda*n), np.zeros(n)]).reshape((2*n, 1)))
        P = matrix(K)
        Gtop = spmatrix(y, range(n), range(n))
        G = sparse([Gtop, -Gtop])
        sol = solvers.qp(P, q, G, h)['x']
    
        return sol
############################################ Multiple Spectrum Kernel ##################################
def project(v):
        mu = list(v)
        mu.sort()
        cumul_sum = np.cumsum(mu)
        rho = np.max([j for j in range(0, len(mu)) if mu[j] - 1/(j+1)*(cumul_sum[j] - 1) > 0])
        
        theta = 1/(rho+1)*(cumul_sum[rho] - 1)
        return np.array([max(0, vi - theta) for vi in v])

def MKL(kernels, y, lmda, T):
    m = len(kernels)
    d = np.array([1/m for k in range(m)])
    for t in range(T):
        K = np.zeros_like(kernels[0])
        for i, Km in enumerate(kernels):
            K = K + d[i]*Km
        alpha = Kernel_SVM.SVM(K, y, lmda) 
        grad = [-0.5*lmda*np.dot(alpha.T, np.dot(Km, alpha))[0][0] for Km in kernels]
        step = 0.01
        d = project(d - step*np.array(grad)) 
    return d
        
#########################################
def write_predictions(predictions, out_fname):
    data = [[int(np.abs((pred+1)//2))] for i, pred in enumerate(predictions)]
    data = np.concatenate([[['Bound']], data])
    data_frame = pd.DataFrame(data=data[1:,:], columns=data[0])
    data_frame.index.name = 'Id'
    data_frame.to_csv(out_fname)
    
def kernel_train(kernel, training_data, ytrain, lmda):
    K = kernel.gram(training_data)
    alpha = Kernel_SVM.SVM(K, ytrain, lmda)
    return alpha

def kernel_predict(kernel, alpha, training, test):
    predict = []
    for x in tqdm(test):
        predict.append(np.sign(kernel.eval_f(x, alpha, training)))
    return predict

def score(predict, truth):
    return sum([int(predict[i]==truth[i]) for i in range(len(truth))])/len(truth)

def split_data(dataset, y, k, m):
    dataset.populate_kmer_set(k)
    dataset.mismatch_preprocess(k, m)
    idx = range(len(dataset.data))
    pairs = []
    data_tranches = [idx[500*i : 500*i+ 500] for i in range(4)]
    label_tranches = [y[500*i: 500*i + 500] for i in range(4)]
    for i in range(4):
        test, ytest = data_tranches[i], label_tranches[i]
        train = np.concatenate([data_tranches[j] for j in range(4) if j != i])
        ytrain = np.concatenate([label_tranches[j] for j in range(4) if j != i])
        
        pairs.append((train, ytrain, test, ytest))
    return pairs

############################################### Data Loader #######################################
class DataLoader():
    
    def __init__(self, fname):
        self.X = pd.read_csv(fname)['seq']
        self.data = self.X
        self.kmer_set = {}
        self.neigborhoods = {}
        self.alph = "GATC"
        self.precomputed = {}
        
    def spectrum_preprocess(self, k):
        n = self.X.shape[0]
        d = len(self.X[0])
        embedding = [{} for x in self.X]
        print("Computing kmer embedding")
        for i,x in enumerate(tqdm(self.X)):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer in embedding[i]:
                    embedding[i][kmer] += 1
                else:
                    embedding[i][kmer] = 1
        self.data = embedding
        
    def populate_kmer_set(self, k):
        d = len(self.X[0])
        idx = 0
        print("Populating kmer set")
        for x in tqdm(self.X):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer not in self.kmer_set:
                    self.kmer_set[kmer] = idx
                    idx +=1  
            
    def mismatch_preprocess(self, k, m):
        n = self.X.shape[0]
        d = len(self.X[0])
        embedding = [{} for x in self.X]
        print("Computing mismatch embedding")
        for i,x in enumerate(tqdm(self.X)):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer not in self.precomputed:
                    Mneighborhood = self.m_neighborhood(kmer, m)
                    self.precomputed[kmer] = [self.kmer_set[neighbor] for neighbor in Mneighborhood if neighbor in self.kmer_set]
                    
                for idx in self.precomputed[kmer]:
                    if idx in embedding[i]:
                        embedding[i][idx] += 1
                    else:
                        embedding[i][idx] = 1
        self.data = embedding
            
    def m_neighborhood(self, kmer, m):
        mismatch_list = deque([(0, "")])
        for letter in kmer:
            num_candidates = len(mismatch_list)
            for i in range(num_candidates):
                mismatches, candidate = mismatch_list.popleft()
                if mismatches < m :
                    for a in self.alph:
                        if a == letter :
                            mismatch_list.append((mismatches, candidate + a))
                        else:
                            mismatch_list.append((mismatches + 1, candidate + a))
                if mismatches == m:
                    mismatch_list.append((mismatches, candidate + letter))
        return [candidate for mismatches, candidate in mismatch_list]

##################################################################################################################

print('''
---------------------------------------------------------------------------------
------Generating Submission File: This may take some time. Please be patient-----
---------------------------------------------------------------------------------
''')

dataset = DataLoader('data/Xtr.csv')

labels = pd.read_csv('data/Ytr.csv')
y = 2.0*np.array(labels['Bound']) - 1

test = DataLoader('data/Xte.csv')

dataset.X = pd.concat([dataset.X, test.X], axis = 0, ignore_index = True)

dataset.populate_kmer_set(k = 12) #12
dataset.mismatch_preprocess(k=12, m=1)#12,1
Kernell_1 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.populate_kmer_set(k = 13)#13
dataset.mismatch_preprocess(k=13, m=1)#13,1
Kernell_2 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.populate_kmer_set(k = 15)#15
dataset.mismatch_preprocess(k=15, m=1)#15,1
Kernell_3 = Kernel(Kernel.mismatch()).gram(dataset.data)


# Add kernels together
K = Kernell_1 + Kernell_2 + Kernell_3 

training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]

lmda = 1.0#0.8

alpha = Kernel_SVM.SVM(K[training][:, training], y, lmda)

predictions = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += alpha[k]*K[i, j]
    predictions.append(np.sign(val))
submission_file = "Yte_final_last6.csv"
write_predictions(predictions, submission_file)
print('')
print('All Done...!!!')
print('''
---------------------------------------------------------------------------------------------------
---------Submission File Generated: Thank you for your time and knowledge shared with us.----------
---------------------------------------------------------------------------------------------------
''')


# ###############
# dataset.populate_kmer_set(12)
# test.kmer_set = dataset.kmer_set
# dataset.mismatch_preprocess(12 , 0)
# test.mismatch_preprocess(12, 0)
# kernel = Kernel(Kernel.sparse_gaussian(7.8))
# lmda = 0.00000001
# alpha = kernel_train(kernel, dataset.data, y, lmda)
# predictions = kernel_predict(kernel, alpha, dataset.data, test.data)
# write_predictions(predictions, "Yte_mismatch_plus_gauss.csv")
# print('All Done')
# ################
