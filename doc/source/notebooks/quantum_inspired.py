import numpy as np
import scipy as sp
import scipy.linalg
import random
import cmath, math

class SegmentTree:
    def __init__(self, n):
        self.size   = 2**(math.ceil(math.log2(n)) + 1)
        self.height = math.ceil(math.log2(n))
        self.data   = np.zeros(self.size)

    def root(self):
        return self.data[1] 
               
    def leaf(self, i):
        index = self.size//2 + i
        return self.data[index]
    
    def update(self, k, val):
        size = self.size
        for _ in (range(self.height+1)):
            size = size//2
            self.data[size + k] += val
            k = k // 2 
            
    def __str__(self):
        return str(self.data)
    
class VectorBasedDataStructure:    
    def __init__(self, vector):
        self.n = vector.size
        self.segment_tree = SegmentTree(self.n)
        self.sgn = np.sign(vector) #[vector[i]/abs(vector[i]) for i in range(self.n)]
        for i in range(self.n):
            self.segment_tree.update(i, vector[i]**2)
                            
    def sample(self):
        height = self.segment_tree.height    
        size   = self.segment_tree.size
        k = 1
        for _ in range(height):
            if random.random()*self.segment_tree.data[k] < self.segment_tree.data[2*k]:
                k = 2*k
            else:
                k = 2*k+1
        return k - size//2   
    
    def query(self, i):
        val = self.segment_tree.leaf(i)
        return self.sgn[i]*math.sqrt(val)    
            
    def norm(self):
        return math.sqrt(self.segment_tree.root())
    
    def print_structure(self):
        data = self.segment_tree.data
        height = self.segment_tree.height
        print("height =", height)
        k = 1
        for i in range(height+1):
            lst = []
            for j in range(2**i,2**(i+1)):  
                lst.append(data[j])
            print(lst)   
        print(self.sgn)
        
class MatrixBasedDataStructure:    
    def __init__(self, matrix):
        m, n = matrix.shape
        self.shape = matrix.shape
        
        self.vecFro = np.array([np.linalg.norm(matrix[i,:]) for i in range(m)])
        self.SQvecFro = VectorBasedDataStructure(self.vecFro)
        self.SQrowlist = [VectorBasedDataStructure(matrix[i,:]) for i in range(m)]

    def sample1(self):
        return self.SQvecFro.sample()

    def sample2(self, i):
        return self.SQrowlist[i].sample()
    
    def query(self, i, j):
        return self.SQrowlist[i].query(j)
    
    def norm(self, i):
        return self.SQrowlist[i].norm()
    
    def normF(self):
        return self.SQvecFro.norm()
    

        
### ---- qi_svd ----
        
def sample_rows(SQA, r): 
    row_indices = []
    for s in range(r):
        i_s = SQA.sample1()
        row_indices.append(i_s)    
    return row_indices

# 確認用
def construct_R(SQA, A, row_indices):
    r = len(row_indices)
    R = np.zeros((r, A.shape[1]))
    for s in range(r):
        i_s = row_indices[s]
        R[s,:] = (SQA.normF()/math.sqrt(r))*A[i_s,:]/SQA.norm(i_s)
    return R

def sample_cols(SQA, row_indices, c):
    r = len(row_indices)

    col_indices = []
    for t in range(c):
        s   = np.random.randint(r)
        i_s = row_indices[s]
        j_t = SQA.sample2(i_s)
        col_indices.append(j_t)
        
    return col_indices

def construct_C(SQA, row_indices, col_indices):
    normAF = SQA.normF()
    
    r = len(row_indices)
    c = len(col_indices)

    # 行列Cを構成する
    C = np.zeros((r, c))
    for t in range(c):
        
        # Rのj_t列を構成する
        R_jt = np.zeros(r)        
        j_t = col_indices[t]
        for s in range(r):
            i_s = row_indices[s]
            R_jt[s] = (normAF/(math.sqrt(r) * SQA.norm(i_s))) * SQA.query(i_s, j_t)
            
        # Cのt列を構成
        C[:, t] = (normAF/math.sqrt(c)) * R_jt/np.linalg.norm(R_jt)
        
    return C

def qi_svd(SQA, r, c):
    
    # Step 1
    row_indices = sample_rows(SQA, r)
    
    # Step 2
    col_indices = sample_cols(SQA, row_indices, c)
    C = construct_C(SQA, row_indices, col_indices)
    
    # Step 3
    W, S, _ = np.linalg.svd(C)
    
    return row_indices, W, S


### ---- SQ(x) ----

def R_col(SQA, k, row_indices):
    r = len(row_indices)
    Rk = np.zeros(r)
    for s in range(r):
        i_s = row_indices[s]
        Rk[s] = SQA.query(i_s, k)/SQA.norm(i_s)
    return (SQA.normF()/math.sqrt(r)) * Rk

def estimate_lambda(SQA, b, row_indices, W, S, ell, sample_size=100):    
    lam = 0
    for _ in range(sample_size):
        # A
        i = SQA.sample1()
        j = SQA.sample2(i)
        Aij = SQA.query(i,j)

        # B
        Rj = R_col(SQA, j, row_indices) 
        vj = (1/S[ell]) * np.vdot(Rj, W[:, ell])
        Bij = b[i]*vj

        # Zij 
        Zij =  Bij/Aij 
        
        lam += Zij
    lam = lam/sample_size
    return lam

def construct_w(SQA, b, row_indices, W, S, rank, lambda_sample_size=100):
    lams = [0]*rank
    for i in range(rank):        
        lams[i] = estimate_lambda(SQA, b, row_indices, W, S, i, lambda_sample_size)
    w = sum(lams[i] * W[:,i]/S[i]**3 for i in range(rank))
    return w

# Query(k)
def SQx_query(SQA, k, row_indices, w):
    """
    wがすでにある時
    """
    
    Rk = R_col(SQA, k, row_indices)
    return np.vdot(Rk, w)

def SQx_query1(SQA, b, k, row_indices, W, S, rank=2, lambda_sample_size=100):
    """
    wを構成するとき
    """
    w  = construct_w(SQA, b, row_indices, W, S, rank, lambda_sample_size)
    Rk = R_col(SQA, k, row_indices)
    return np.vdot(Rk, w)

def SQx_query2(SQA, b, k, r, c, rank=2, sample_size=100):
    """
    qi_svdも含めて行うとき
    """

    # qi_svd実行
    row_indices, W, S = qi_svd(SQA, r, c)
    
    lams = [0]*rank
    for i in range(rank):        
        lams[i] = estimate_lambda(SQA, b, row_indices, W, S, i, sample_size)
    w = sum(lams[i] * W[:,i]/S[i]**3 for i in range(rank))
    
    Rk = R_col(SQA, k, row_indices)
    
    return np.vdot(Rk, w)


# Sample()
def SQx_sample(SQA, b, w, row_indices, W, S):
    r = len(w)

    normw2 = np.linalg.norm(w)**2
    prob_   = [w[s]**2/normw2 for s in range(r)]
    
    accept = False
    while not accept : 
        
        # 提案分布
        s_ = np.random.choice(range(r), 1, p=prob_)[0]   
        i_s = row_indices[s_]
        j = SQA.sample2(i_s)

        # xj = x[j]
        xj = SQx_query(SQA, j, row_indices, w)

        # rj を計算
        val = 0
        for s in range(r):
            i_s = row_indices[s]
            val += (w[s] * SQA.query(i_s, j)/SQA.norm(i_s))**2

        rj = xj**2 / (SQA.normF()**2 * val)
        
        
        y = random.random()
        if y < rj : 
            accept = True
            
    return j

# Norm()
def _SQx_norm(SQA, b, w, row_indices, W, S):
    r = len(w)

    normw2 = np.linalg.norm(w)**2
    prob_   = [w[s]**2/normw2 for s in range(r)]
    
    s_ = np.random.choice(range(r), 1, p=prob_)[0]   
    i_s = row_indices[s_]
    j = SQA.sample2(i_s)

    xj = SQx_query(SQA, j, row_indices, w)
    
    val = 0
    for s in range(r):
        i_s = row_indices[s]
        val += (w[s] * SQA.query(i_s, j)/SQA.norm(i_s))**2

    rj = xj**2 / (SQA.normF()**2 * val)

    y = random.random()
    if y < rj : 
        return 1
    else:
        return 0
    

def SQx_norm(SQA, b, w, row_indices, W, S, norm_sample_size=100):
    normw2 = np.linalg.norm(w)**2
    normAF2 = SQA.normF()**2
    cnt = sum(_SQx_norm(SQA, b, w, row_indices, W, S) for _ in range(norm_sample_size))
    val = cnt/norm_sample_size
    
    return math.sqrt(val * normAF2 * normw2)
    

class SQx:
    def __init__(self, SQA, b, r, c, rank=2, lambda_sample_size=100):   
        self.SQA = SQA
        self.b = b
        self.r, self.c = r, c
        self.row_indices, self.W, self.S = qi_svd(SQA, r, c)
        self.w = construct_w(SQA, b, self.row_indices, self.W, self.S, rank=2, lambda_sample_size=100)
        
    def sample(self):
        return SQx_sample(self.SQA, self.b, self.w, self.row_indices, self.W, self.S)
    
    def query(self, k):
        return SQx_query(self.SQA, k, self.row_indices, self.w)
    
    def norm(self, norm_sample_size=100):
        return SQx_norm(self.SQA, self.b, self.w, self.row_indices, self.W, self.S, norm_sample_size)
    
    
### ---- others -----

def qi_vdot(SQa, Qb, sample_size=100):
    val = 0
    for _ in range(sample_size):
        i  = SQa.sample()
        ai = SQa.query(i)
        bi = Qb[i]
        val += bi/ai 
    return val/sample_size * SQa.norm()**2

def qi_HS(SQA, QB, sample_size=100):
    """ヒルベルト・シュミット内積 (QBは配列)
    """
    val = 0
    for _ in range(sample_size):
        i = SQA.sample1()
        j = SQA.sample2(i)
        Aij = SQA.query(i,j)
        Bij = QB[i,j] # or 
        val += Bij/Aij
    return val/sample_size * SQA.normF()**2


def low_rank_matrix(m, n, rank, cond=10):

    """
    ランクが`rank`で，条件数が`cond`であるような m x n の実行列を生成する．
    ここで条件数とは最大特異値/最小特異値である．したがってrank=1のときは条件数の値は1になる．
    
    """
    
    U, _  = np.linalg.qr(np.random.rand(m, m))
    Vh, _ = np.linalg.qr(np.random.rand(n, n))
    
    if rank==1:
        diag = np.array([cond])
        
    elif rank==2: 
        diag = np.array([1, 1/cond])
    else:
        diag = [random.uniform(1/cond, 1) for i in range(rank)]
        diag[1] = 1
        diag[2] = 1/cond
    
    return U[:,:rank] @ np.diag(diag) @ Vh[:rank,:]
