import cv2
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def WLSFilter(IN,lamb=1, alpha=1.2, L=None):
    output = np.zeros_like(IN)
    if L is None:
        L = np.log(IN)

    def wlsFilter_helper(IN, lamb=1, alpha=1.2, L=None): 

        smallNum = 0.0001

        r,c = IN.shape
        k = r*c

        dy = np.diff(L,1,0)

        dy = -lamb/(np.abs(dy)**alpha + smallNum)
        dy = np.pad(dy, ((0 ,1),(0,0)),mode='constant')
        dy = dy.flatten()

        dx = np.diff(L, 1, 1); 
        dx = -lamb/(np.abs(dx)**alpha + smallNum);
        dx = np.pad(dx, ((0 ,0),(0,1)),mode='constant')
        dx = dx.flatten()

        B = np.zeros(shape=(dx.shape[0],2))
        B[:,0] = dx
        B[:,1] = dy;
        d = np.array([-r,-1])

        A = spdiags(B.T,d,k,k)

        e = dx
        w = np.pad(dx, ((r,0)) ,mode= 'constant')
        w = w[0:-r]

        s = dy
        n = np.pad(dy, ((1,0)), mode= 'constant')
        n = n[0:-1]

        D = 1-(e+w+s+n)
        A = A + A.T + spdiags(D.T, 0, k, k)
        OUT = spsolve(A,IN.flatten())

        OUT = np.reshape(OUT, (r, c))

        return OUT
    output[:,:,0] = wlsFilter_helper(IN[:,:,0],lamb=lamb,alpha=alpha,L=L[:,:,0])
    output[:,:,1] = wlsFilter_helper(IN[:,:,1],lamb=lamb,alpha=alpha,L=L[:,:,1])
    output[:,:,2] = wlsFilter_helper(IN[:,:,2],lamb=lamb,alpha=alpha,L=L[:,:,2])
    return output
    
