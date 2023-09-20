import numpy as np
import scipy as sp
import scipy.linalg
import random
import cmath, math


# def low_rank_approximation(A, rank):
#     U, S, V = np.linalg.svd(A)
#     Ur = U[:,:rank]
#     Sr = S[:rank]
#     Vr = V[:rank,:]
#     return Ur @ np.diag(Sr) @ Vr 
        
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
   

def cond(matrix, rank):
    svdvals = sp.linalg.svdvals(matrix)[:rank]
    return max(svdvals)/min(svdvals)
    