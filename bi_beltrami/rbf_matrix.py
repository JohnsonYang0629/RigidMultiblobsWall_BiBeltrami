import random
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from cvxopt import matrix, solvers
from estimate_filter_P import sphere_P0Cheat_v2
import scipy.io


def generatemultiindex(N, dim):
    """
    Compute index of polynomials.

    Inputs:
    - N:  max degree of polynomials.
    - dim: dimension.

    Returns:
    - index
    """

    P = math.comb(N + dim, dim)
    index = np.zeros((dim, P))
    Ntotal = (N + 1) ** dim
    allindex = np.zeros((dim, Ntotal))

    for i in range(dim):
        nskip = (N + 1) ** (dim - i - 1)
        for k in range(int(Ntotal / nskip / (N + 1))):
            for j in range(N + 1):
                allindex[i, k * nskip * (N + 1) + j * nskip : k * nskip * (N + 1) + (j + 1) * nskip] = j
    index1 = np.where(np.sum(allindex, axis=0) <= N)
    index = allindex[:,index1]
    return index

def Laplace_v1p1_gmls_ext_weightK(x, k, degr, P, tvec, operator):
    """
    Compute Laplacian matrix with extrinsic formula

    Inputs:
    - x: extrinsic coordinates.
    - k: number of neighbors of local stencil.
    - degr: degree of polynomials.
    - P: projection matrix.
    - tvec: tangent vector.
    - operator: 1 is Laplace-Beltrami operator, 2 is Bochner Laplacian, 3 is Hodge Laplacian and 4 is Lich Laplacian.

    Returns:
    - Matrix_Lap: Laplacian matrix.
    """

    N = x.shape[0]
    n = x.shape[1]
    d = tvec.shape[0]
    
    if d > 1:
        index = np.squeeze(generatemultiindex(degr, d)) # index is d*term
    else:
        index = range(degr + 1)
    term = index.shape[1]
    
    ### compute k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nn.fit(x)
    distance, inds = nn.kneighbors(x)
    if operator == 1:
        a1 = np.zeros(N * k)
        a2 = np.zeros(N * k)
        j3 = np.zeros(N * k)
    elif operator > 1:
        a1 = np.zeros(N * k * d ** 2)
        a2 = np.zeros(N * k * d ** 2)
        j3 = np.zeros(N * k * d ** 2)
    
    for pp in range(N):
        xx = x[inds[pp, :], :] # knn points
        x0 = x[pp, :] # center point
        pvec = tvec[:, :, pp] # pvec is d*n at center point, tvec is d*n*N
        xnorm = np.sqrt(np.sum(np.sum((xx - x0) ** 2, axis=1), axis=0) / n / k)
        yy = (xx - x0) / xnorm # k*n normalized data
        inyy = yy @ pvec.T # k*d intrinsic normalized polynomial
        
        ### Phi = [1 yy yy.^2 yy.^3 ...]
        Phi = np.ones((k, term)) # k*term
        for ss in range(term):
            for rr in range(d):
                Phi[:, ss] = Phi[:, ss] * inyy[:, rr] ** index[rr, ss]
                
        ### 1/k weight
        W = sparse.csr_matrix(np.eye(k) / k)
        W[0, 0] = 1
        
        ### GMLS
        PhiInv = np.linalg.solve(Phi.T @ W @ Phi, Phi.T @ W)
        Gi = np.zeros((k,k,n)) # 1st order derivative square matrix
        for kk in range(n):
            Phiprime = np.zeros((k, term))
            dinyy = P[inds[pp, :], :, kk] @ pvec.T / xnorm # (k*n)*(n*d)
            for ss in range(1, term): # the first term is always zero
                for rr in range(d): # derivative wrt the rr^th dimension
                    if index[rr, ss] == 0:
                        DiOne = np.zeros(k)                     
                    else:
                        DiOne = np.ones(k)
                        for tt in range(d):
                            if tt == rr:
                                # the only one derivative term in monomial
                                DiOne = DiOne * index[tt, ss] * inyy[:, tt] ** (index[tt, ss] - 1) * dinyy[:, tt]
                            else:
                                DiOne = DiOne * inyy[:, tt] ** index[tt, ss]
                    Phiprime[:, ss] += DiOne
                    
            ### compute Gi = Di*Phi^-1
            Gi[:, :, kk] = Phiprime @ PhiInv
            
        ### differential operator construction 
        if operator == 1:
            matfunlocal = np.zeros((1,k))
            for mm in range(n):
                matfunlocal += Gi[0, :, mm] @ Gi[:, :, mm]
                
            ### give values to LB Matrix
            a1[pp * k : pp * k + k] = pp * np.ones(k)
            a2[pp * k : pp * k + k] = inds[pp, :]
            j3[pp * k : pp * k + k] = matfunlocal[0, 0 : k]
        elif operator > 1:
            if operator == 2:
                Ci = np.zeros((d * k, n))
                for mm in range(n):
                    Ci[:, mm] = np.reshape(np.squeeze(tvec[:, mm, inds[pp, :]]), d * k, order='f')
                Bi = Ci @ Ci.T
                weight0 = np.zeros((d, d * k))
            
                for mm in range(n):
                    Btemp = Bi * np.kron(Gi[:, :, mm], np.ones((d, d)))
                    weight0 += Btemp[0 : d, :] @ Btemp                   
            elif operator == 3:
                Ci = np.zeros((d * k, n))
                for mm in range(n):
                    Ci[:, mm] = np.reshape(np.squeeze(tvec[:, mm, inds[pp, :]]), d * k, order='f')
                Bi = Ci @ Ci.T            
                Di = np.zeros((d * k, k))
                for mm in range(k):
                    Di[mm * d : mm * d + d, :] = tvec[:, :, inds[pp, mm]] @ np.squeeze(Gi[mm, :, :]).T
                Di2 = np.zeros((d * k, d * k))
                for mm in range(d):
                    Di2[:, mm : d * k : d] = Di
                weight0 = np.zeros((d, d * k))
                for mm in range(n):
                    Btemp = Bi * np.kron(Gi[:, :, mm], np.ones((d, d)))
                    Ei = P[inds[pp, :], :, mm] @ Ci.T
                    Ei2 = np.zeros((d * k, d * k))
                    for ss in range(d):
                        Ei2[ss : d * k : d, :] = Ei
                    Rtemp = Di2 * Ei2
                    weight0 += Btemp[0 : d, :] @ (Btemp - Rtemp)
                Fi = np.zeros((k, d * k))
                for mm in range(k):
                    Fi[:, mm * d : mm * d + d] = np.squeeze(Gi[:, mm, :]) @ tvec[:, :, inds[pp, mm]].T
                weight0 += Di[0 : d, :] @ Fi
                
            elif operator == 4:
                Ci = np.zeros((d * k, n))
                for mm in range(n):
                    Ci[:, mm] = np.reshape(np.squeeze(tvec[:, mm, inds[pp, :]]), d * k, order='f')
                Bi = Ci @ Ci.T            
                Di = np.zeros((d * k, k))
                for mm in range(k):
                    Di[mm * d : mm * d + d, :] = tvec[:, :, inds[pp, mm]] @ np.squeeze(Gi[mm, :, :]).T
                Di2 = np.zeros((d * k, d * k))
                for mm in range(d):
                    Di2[:, mm : d * k : d] = Di
                weight0 = np.zeros((d, d * k))
                for mm in range(n):
                    Btemp = Bi * np.kron(Gi[:, :, mm], np.ones((d, d)))
                    Ei = P[inds[pp, :], :, mm] @ Ci.T
                    Ei2 = np.zeros((d * k, d * k))
                    for ss in range(d):
                        Ei2[ss : d * k : d, :] = Ei
                    Rtemp = Di2 * Ei2
                    weight0 += Btemp[0 : d, :] @ (Btemp + Rtemp)                        
            for jj in range(d):
                for qq in range(d):
                    a1[pp * k * d ** 2 + jj * k * d + qq * k : pp * k * d ** 2 + jj * k * d + qq * k + k] = jj * N + pp * np.ones(k)
                    a2[pp * k * d ** 2 + jj * k * d + qq * k : pp * k * d ** 2 + jj * k * d + qq * k + k] = qq * N + inds[pp, :]
                    j3[pp * k * d ** 2 + jj * k * d + qq * k : pp * k * d ** 2 + jj * k * d + qq * k + k] \
                    = weight0[jj, qq : d * k : d]
    if operator == 1:
        Matrix_Lap = sparse.coo_matrix((j3, (a1, a2)), shape=(N, N)) # Matrix_Lich local LB sparse
    elif operator > 1:
        Matrix_Lap = sparse.coo_matrix((j3, (a1, a2)), shape=(N * d, N * d))
    return Matrix_Lap


def Laplace_v1p2_gmls_int_weightK(x, k, degr, tvec, operator):
    """
    Compute Laplacian matrix with intrinsic formula

    Inputs:
    - x: extrinsic coordinates.
    - k: number of neighbors of local stencil.
    - degr: degree of polynomials.
    - tvec: tangent vector.
    - operator: 1 is Laplace-Beltrami operator, 2 is Bochner Laplacian, 3 is Hodge Laplacian and 4 is Lich Laplacian.

    Returns:
    - Matrix_Lap: Laplacian matrix.
    """
    
    N = x.shape[0]
    n = x.shape[1]
    d = tvec.shape[0]
    
    ### generate index of polynomial
    if d > 1:
        index = np.squeeze(generatemultiindex(degr, d)) # index is d*term
    else:
        index = range(degr + 1)
    term = index.shape[1]

    ### partial index used for manifold regression
    ind_grt2 = np.where(np.sum(index, 0) >= 2) # regress only for deg>=2
    denorm = np.sum(index[:, ind_grt2], 0) # remove norm effect
    ind_b02 = (np.sum(index[:, ind_grt2], 0) == 2) & (index[0, ind_grt2] == 0)
    ind_b11 = (np.sum(index[:, ind_grt2], 0) == 2) & (index[0, ind_grt2] == 1)
    ind_b20 = (np.sum(index[:, ind_grt2], 0) == 2) & (index[0, ind_grt2] == 2)
    
    ### all index used for function components of vector fields
    denorm_all = np.sum(index, 0) # remove norm effect
    ind_a00 = (np.sum(index, 0) == 0)
    ind_a02 = (np.sum(index, 0) == 2) & (index[0, :] == 0)
    ind_a20 = (np.sum(index, 0) == 2) & (index[0, :] == 2)
    ind_a11 = (np.sum(index, 0) == 2) & (index[0, :] == 1)
    
    ### compute k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nn.fit(x)
    distance, inds = nn.kneighbors(x)
    if operator == 1:
        a1 = np.zeros(N * k)
        a2 = np.zeros(N * k)
        j3 = np.zeros(N * k)
    elif operator > 1:
        a1 = np.zeros(N * k * d ** 2)
        a2 = np.zeros(N * k * d ** 2)
        j3 = np.zeros(N * k * d ** 2)
        
    for pp in range(N):
        xx = x[inds[pp, :], :] # knn points
        x0 = x[pp, :] # center point
        pvec = tvec[:, :, pp] # pvec is d*n at center point, tvec is d*n*N
        xnorm = np.sqrt(np.sum(np.sum((xx - x0) ** 2, axis=1), axis=0) / n / k)
        yy = (xx - x0) / xnorm # k*n normalized data
        inyy = yy @ pvec.T # k*d intrinsic normalized polynomial
        
        ### Phi = [1 yy yy.^2 yy.^3 ...]
        Phi = np.ones((k, term)) # k*term
        for ss in range(term):
            for rr in range(d):
                Phi[:, ss] = Phi[:, ss] * inyy[:, rr] ** index[rr, ss]
                
        ### 1/k weight
        W = sparse.csr_matrix(np.eye(k) / k)
        W[0, 0] = 1
        
        ### GMLS
        PhiInv = np.linalg.solve(Phi.T @ W @ Phi, Phi.T @ W)
        PhiInv = PhiInv / xnorm ** np.tile(denorm_all, (k, 1)).T # remove normalization effect

        ### differential operator construction 
        if operator == 1:
            weight0 = np.zeros((1, k))
            co11 = np.zeros((1, term))
            co11[0, ind_a20] = 2
            co11[0, ind_a02] = 2
            weight0[0, 0 : k] = co11 @ PhiInv
        
            a1[pp * k : pp * k + k] = pp * np.ones(k)
            a2[pp * k : pp * k + k] = inds[pp, :]
            j3[pp * k : pp * k + k] = weight0[0, 0 : k]          
        elif operator > 1:
            ### regress for manifolds, deg>=2 of PHI inverse
            nn = np.cross(pvec[0, :], pvec[1, :]).reshape((1, n)) # 1*n
            inqq = (yy @ nn.T).reshape((k,1)) # n*x, 3rd component           
            bb = np.linalg.solve(np.squeeze(Phi[:, ind_grt2]).T \
                                 @ np.squeeze(Phi[:, ind_grt2]), np.squeeze(Phi[:,ind_grt2]).T @ inqq) # regress only for deg>=2
            bb = bb / xnorm ** (denorm.T - 1) # remove normalization effect
            bb = bb.T           
            bb02 = bb[ind_b02] # coeff b_{0,2} for manifold
            bb11 = bb[ind_b11] # coeff b_{1,1} for manifold
            bb20 = bb[ind_b20] # coeff b_{2,0} for manifold
            if operator == 2:
                weight0 = np.zeros((d, d * k))
            
                ### 1,1 entry
                co11 = np.zeros((1, term))
                co11[0, ind_a00] = 4 * bb20 ** 2 + bb11 ** 2
                co11[0, ind_a20] = 2
                co11[0, ind_a02] = 2
                weight0[0, 0 : d * k : d] = co11 @ PhiInv
            
                ### 1,2 entry
                co12 = np.zeros((1, term))
                co12[0, ind_a00] = 2 * bb20 * bb11 + 2 * bb02 * bb11
                weight0[0, 1 : d * k : d] = co12 @ PhiInv
            
                ### 2,1 entry
                co21 = np.zeros((1, term))
                co21[0, ind_a00] = 2 * bb20 * bb11 + 2 * bb02 * bb11
                weight0[1, 0 : d * k : d] = co21 @ PhiInv
            
                ### 2,2 entry
                co22 = np.zeros((1, term))
                co22[0, ind_a00] = 4 * bb20 ** 2 + bb11 ** 2
                co22[0, ind_a20] = 2
                co22[0, ind_a02] = 2
                weight0[1, 1 : d * k : d] = co22 @ PhiInv   
            elif operator == 3:
                weight0 = np.zeros((d, d * k))
            
                ### 1,1 entry
                co11 = np.zeros((1, term))
                co11[0, ind_a00] = 4 * bb20 ** 2 + 2 * bb11 ** 2 - 4 * bb02 * bb20
                co11[0, ind_a20] = 2
                co11[0, ind_a02] = 2
                weight0[0, 0 : d * k : d] = co11 @ PhiInv
            
                ### 1,2 entry
                co12 = np.zeros((1, term))
                co12[0, ind_a00] = 2 * bb20 * bb11 + 2 * bb02 * bb11
                weight0[0, 1 : d * k : d] = co12 @ PhiInv
            
                ### 2,1 entry
                co21 = np.zeros((1, term))
                co21[0, ind_a00] = 2 * bb20 * bb11 + 2 * bb02 * bb11
                weight0[1, 0 : d * k : d] = co21 @ PhiInv
            
                ### 2,2 entry
                co22 = np.zeros((1, term))
                co22[0, ind_a00] = 4 * bb20 ** 2 + 2 * bb11 ** 2 - 4 * bb02 * bb20
                co22[0, ind_a20] = 2
                co22[0, ind_a02] = 2
                weight0[1, 1 : d * k : d] = co22 @ PhiInv
            elif operator == 4:
                weight0 = np.zeros((d, d * k))
            
                ### 1,1 entry
                co11 = np.zeros((1, term))
                co11[0, ind_a00] = 8 * bb20 ** 2 + bb11 ** 2 + 4 * bb02 * bb20
                co11[0, ind_a20] = 4
                co11[0, ind_a02] = 2
                weight0[0, 0 : d * k : d] = co11 @ PhiInv
            
                ### 1,2 entry
                co12 = np.zeros((1, term))
                co12[0, ind_a00] = 4 * bb20 * bb11 + 4 * bb02 * bb11
                co12[0, ind_a11] = 1
                weight0[0, 1 : d * k : d] = co12 @ PhiInv
            
                ### 2,1 entry
                co21 = np.zeros((1, term))
                co21[0, ind_a00] = 4 * bb20 * bb11 + 4 * bb02 * bb11
                co21[0, ind_a11] = 1
                weight0[1, 0 : d * k : d] = co21 @ PhiInv
            
                ### 2,2 entry
                co22 = np.zeros((1, term))
                co22[0, ind_a00] = 8 * bb20 ** 2 + bb11 ** 2 + 4 * bb02 * bb20
                co22[0, ind_a20] = 2
                co22[0, ind_a02] = 4
                weight0[1, 1 : d * k : d] = co22 @ PhiInv
                    
            ### generation of Monge chart 
            tan_monge = np.zeros((k, n, d)) # tangent vectors in Monge chart
            inyy_unnorm = inyy * xnorm
            index2 = np.squeeze(index[:, ind_grt2])
            for rr in range(d):
                
                ### fix rr as rr th derivative
                qder = 0
                for ss in range(index2.shape[1]):
                    if index2[rr, ss] == 0:
                        DiOne = np.zeros((k, 1))
                    else:
                        DiOne = np.ones((k, 1))
                        for tt in range(d):
                            if tt == rr:
                            
                                ### the only one derivative term in monomial
                                DiOne = DiOne * index2[tt, ss] * inyy_unnorm[:, tt].reshape((k, 1)) ** (index2[tt, ss] - 1)
                            else:
                                DiOne = DiOne * inyy_unnorm[:, tt].reshape((k, 1)) ** index2[tt, ss]
                    qder += bb[0, ss] * DiOne
                tan_monge[:, :, rr] = np.tile(pvec[rr, :], (k, 1)) + qder @ nn # k*n*d
            tan_monge = tan_monge.transpose((2, 1, 0)) # d*n*k
        
            ### rotation from Monge chart to global chart
            w_all = np.zeros((d, d * k))
            for ii in range(k):
                loctemp = np.linalg.pinv(tan_monge[:, :, ii] @ tan_monge[:, :, ii].T) @ tan_monge[:, :, ii]
                w_all[0 : d, ii * d : ii * d + d] = weight0[0 : d, ii * d : ii * d + d] @ (loctemp @ tvec[:, :, inds[pp, ii]].T) # tvec is d*n*N
            for jj in range(d):
                for qq in range(d):
                    a1[pp * k * d ** 2 + jj * k * d + qq * k : pp * k * d ** 2 + jj * k * d + qq * k + k] = jj * N + pp * np.ones(k)
                    a2[pp * k * d ** 2 + jj * k * d + qq * k : pp * k * d ** 2 + jj * k * d + qq * k + k] = qq * N + inds[pp, :]
                    j3[pp * k * d ** 2 + jj * k * d + qq * k : pp * k * d ** 2 + jj * k * d + qq * k + k] \
                    = w_all[jj, qq : d * k : d]
    if operator == 1:
        Matrix_Lap = sparse.coo_matrix((j3, (a1, a2)), shape=(N, N))
    elif operator > 1:
        Matrix_Lap = sparse.coo_matrix((j3, (a1, a2)), shape=(N * d, N * d))
    return Matrix_Lap

def Laplace_v1p3_quad_prog(x, k0, degr, tvec, operator):
    """
    Compute Laplacian matrix with intrinsic formula

    Inputs:
    - x: extrinsic coordinates.
    - k: number of neighbors of local stencil.
    - degr: degree of polynomials.
    - tvec: tangent vector.
    - operator: 1 is Laplace-Beltrami operator, 2 is Bochner Laplacian, 3 is Hodge Laplacian and 4 is Lich Laplacian.

    Returns:
    - Matrix_Lap: Laplacian matrix.
    """
    
    if operator > 1:
        return None
    N = x.shape[0]
    n = x.shape[1]
    d = tvec.shape[0]    
    if d > 1:
        index = np.squeeze(generatemultiindex(degr, d)) # index is d*term
    else:
        index = range(degr + 1)
    term = index.shape[1]
    ind_a02 = (np.sum(index, 0) == 2) & (index[0, :] == 0)
    ind_a20 = (np.sum(index, 0) == 2) & (index[0, :] == 2)
    if operator == 1:
        a1 = np.zeros(N * k0)
        a2 = np.zeros(N * k0)
        j3 = np.zeros(N * k0)
    elif operator > 1:
        a1 = np.zeros(N * k0 * d ** 2)
        a2 = np.zeros(N * k0 * d ** 2)
        j3 = np.zeros(N * k0 * d ** 2)
        
    ### adaptive large enough Knn to make matrix SDD
    exitflag1 = np.zeros((N, 1)) # exitflag for 1st time
    exitflag2 = np.zeros((N, 1)) # exitflag for final time
    stdminerr = np.zeros((N, 1)) # local min error from SDD
    k2_NN_rec = np.zeros((N, 1)) + k0 # local k2-NN neighbors 
    iter_numb = np.zeros((N, 1)) # iteration number
    k2NNtotal = np.zeros((1, 1)) # k2 increment
    
    for pp in range(N):
        while True:
        
            ## iterative for many k and knn for w1<0
            k = int(k0 + iter_numb[pp, 0] * 2)
        
            ### compute k-nearest neighbors
            nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
            nn.fit(x)
            distance, inds = nn.kneighbors(x)
            xx = x[inds[pp, :], :] # knn points
            x0 = x[pp, :] # center point
            pvec = tvec[:, :, pp] # pvec is d*n at center point, tvec is d*n*N
            xnorm = np.sqrt(np.sum(np.sum((xx - x0) ** 2, axis=1), axis=0) / n / k)
            yy = (xx - x0) / xnorm # k*n normalized data
            inyy = yy @ pvec.T # k*d intrinsic normalized polynomial
        
            ### Phi = [1 yy yy.^2 yy.^3 ...]
            Phi = np.ones((k, term)) # k*term
            for ss in range(term):
                for rr in range(d):
                    Phi[:, ss] = Phi[:, ss] * inyy[:, rr] ** index[rr, ss]
            if operator == 1:
            
                ### construct target function
                P = np.eye(k+1)
                P[0, 0] = 1 / k ** 2
                P[k, k] = k ** 2
                q = np.zeros((k + 1, 1))
            
                ### inequality constraint
                G = np.zeros((k + 2, k + 1))
                G[0, 0] = 1
                G[1 : k + 1, 1:k + 1] = -np.eye(k)
                G[1 : k+1, k] = -np.ones(k)
                G[k + 1, k] = 1
                h = np.zeros((k + 2, 1))
                h[k + 1, 0] = 10 ** 5
            
                ### equality constraint
                A = np.zeros((term, k + 1))
                A[0 : term, 0 : k] = Phi.T
                b = np.zeros((term, 1))
                b[ind_a02, 0] = 2 / xnorm ** 2
                b[ind_a20, 0] = 2 / xnorm ** 2
            
                ### quadratic programming
                w_x0 = cvxopt_solve_qp(P, q, G, h, A, b)
                w_all = np.zeros((1, k))
                if w_x0 is not None:
                    exitflag = 1
                if iter_numb[pp, 0] == 0:
                    exitflag1[pp, 0] = exitflag # if 1st time success using k-NN neighbors
                if (exitflag == 1) & (w_x0[0] < -1): # (w_all(1,1) <= 0) && (w_all(2,2) <= 0)
                
                    ### give optm result to w_all
                    w_all[0, 0 : k] = w_x0[0 : k]
                
                    ### optimization result for low bound C
                    stdminerr[pp, 0] = w_x0[k]
                
                    ### final exitflag
                    exitflag2[pp, 0] = exitflag
                
                    ### final knn
                    k2_NN_rec[pp, 0] = k # the k2-NN neighbors are used
                    break # break while
                elif iter_numb[pp, 0] <= 10:
                    print("%d th point is not good, exitflag not 1, increase knn once!"%pp)
                    iter_numb[pp, 0] += 1
                else:
                    print("%d th point is boundary or very poor singular, study further why!"%pp)
                    return None
        if iter_numb[pp, 0] > 0:
                
            ### give values to Matrix Laplace-Beltrami
            a1[pp * k0 : pp * k0 + k] = pp * ones(k0)
            a2[pp * k0 : pp * k0 + k] = inds[pp, 0 : k0]
            j3[pp * k0 : pp * k0 + k] = w_all[0, 0 : k0] # j3((pp-1)*k+(1:k),1) = matfunlocal(1,1:k);
            a1[N * k0 + k2NNtotal : N * k0 + k2NNtotal + k - k0] = pp * ones(k-k0)
            a2[N * k0 + k2NNtotal : N * k0 + k2NNtotal + k - k0] = inds[pp, k0 : k]
            j3[N * k0 + k2NNtotal : N * k0 + k2NNtotal + k - k0] = w_all[0, k0 : k]
            
            ### update k2NNtotal
            k2NNtotal = k2NNtotal + (k-k0) # update j3count
        else:
            a1[pp * k : pp * k + k] = pp * np.ones(k)
            a2[pp * k : pp * k + k] = inds[pp, :]
            j3[pp * k : pp * k + k] = w_all[0, 0 : k] 
    if operator == 1:
        Matrix_Lap = sparse.coo_matrix((j3, (a1, a2)), shape=(N, N))
    elif operator > 1:
        Matrix_Lap = sparse.coo_matrix((j3, (a1, a2)), shape=(N * d, N * d))
    return Matrix_Lap
            
def cvxopt_solve_qp(P, q, G, h, A=None, b=None): 
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    args.extend([matrix(G), matrix(h)])
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    sol = solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def solve_potenial(data, nu, gm, k0, degr, method):
    """
    PDE solver.

    Inputs:
    - data: dataset.
    - nu: viscosity nu.
    - gm: coupling coefficient gamma.
    - k0: number of neighbors of local stencil.
    - degr: degree of polynomials.
    - method: 1 is extrinsic, 2 is intrinsic and 3 is optim.

    Returns:
    - numerical solution and true solution.
    """
    
    x = scipy.io.loadmat(data)
    x = x['p'] # point cloud

    ### compute the radius
    # am = np.linalg.norm(x[0, :], 2)
    am = 1

    ### x to theta
    N = x.shape[0]
    THETA = np.array([math.acos(xx[2] / am) for xx in x])
    index1 = np.where((THETA != 0) & (THETA != math.pi))
    PHI = np.zeros(N);
    PHI[index1] = x[index1, 0] / am / np.array([math.sin(xx) for xx in THETA[index1]])
    PHI[index1] = np.array([math.acos(xx) for xx in PHI[index1]])
    posneg = np.ones(N)
    temp = (x[index1, 1] / am / np.array([math.sin(xx) for xx in THETA[index1]])).flatten()
    index2 = np.where(temp < 0)
    posneg[index2] = posneg[index2] * (-1)
    PHI = PHI * posneg
    theta = np.vstack((THETA, PHI)).T
    
    ### give the force or the force potential
    RHS = -6 * x[:, 0] - 66 * x[:,0] * x[:, 1] # analytic RHS
    RHS = RHS.reshape((N, 1))

    Kg = 1 / am ** 2 # Gaussian curvature
    
    P0, tvec2 = sphere_P0Cheat_v2(theta, x, am) # Projection matrix and orthonormal tangent vector

    ### Laplace-Beltrami operator
    operator = 1
    if method == 1:
        Matrix_Lap1_Local = Laplace_v1p1_gmls_ext_weightK(x, k0, degr, P0, tvec2, operator)
    elif method == 2:
        Matrix_Lap1_Local = Laplace_v1p2_gmls_int_weightK(x, k0, degr, tvec2, operator)
    elif method == 3:
        Matrix_Lap1_Local = Laplace_v1p3_quad_prog(x, k0, degr, tvec2, operator)
    Matrix_Lap1_Local = Matrix_Lap1_Local - sparse.spdiags(np.sum(Matrix_Lap1_Local, 1).T, 0, N, N) # row sum is zero since 0 is eigenvalue of LB

    ### direct inverse for non-symmetric LB
    Psi = sparse.linalg.spsolve(-nu * Matrix_Lap1_Local + (gm - 2 * nu * Kg) * sparse.csr_matrix(np.eye(N)), RHS)

    ### Lagrange Multiplier Adjunction
    temp = np.zeros((N + 1, N + 1))
    temp[0 : N, 0 : N] = (Matrix_Lap1_Local.T @ Matrix_Lap1_Local).toarray()
    cona = np.ones(N)
    temp[0 : N, N] = cona
    temp[N, 0 : N] = cona.T
    temp = sparse.csr_matrix(temp)
    Rhs = np.zeros((N + 1, 1))
    Rhs[0 : N, 0] = Matrix_Lap1_Local.T @ Psi
    sol1 = sparse.linalg.spsolve(temp, Rhs)[0 : N]
    true_sol = x[:, 0] + x[:,0] * x[:, 1]
    return sol1, true_sol

def Gi_Save_v2_1K(x, k, degr, tvec):
    """
    PDE solver.

    Inputs:
    - x: N-by-n data set with N data points on d-dim M in R^n.
    - k: k-nearest-neighbors.
    - degr: degree of polynomials used.
    - k0: number of neighbors of local stencil.
    - P: Nxnxn little pi for i=1..n, projection matrix.
    - tvec: tvec is d*n*N, tangent vectors.

    Returns:
    - first order derivative operator
    """
    
    N = x.shape[0]
    n = x.shape[1]
    d = tvec.shape[0]
    
    if d > 1:
        index = np.squeeze(generatemultiindex(degr, d)) # index is d*term
    else:
        index = range(degr + 1)
    term = index.shape[1]
    
    ### compute k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nn.fit(x)
    distance, inds = nn.kneighbors(x)
    GiAugMent = np.zeros((k, n, N))
    for pp in range(N):
        xx = x[inds[pp, :], :] # knn points
        x0 = x[pp, :] # center point
        pvec = tvec[:, :, pp] # pvec is d*n at center point, tvec is d*n*N
        xnorm = np.sqrt(np.sum(np.sum((xx - x0) ** 2, axis=1), axis=0) / n / k)
        yy = (xx - x0) / xnorm # k*n normalized data
        inyy = yy @ pvec.T # k*d intrinsic normalized polynomial
        
        ### Phi = [1 yy yy.^2 yy.^3 ...]
        Phi = np.ones((k, term)) # k*term
        for ss in range(term):
            for rr in range(d):
                Phi[:, ss] = Phi[:, ss] * inyy[:, rr] ** index[rr, ss]
                
        ### 1/k weight
        W = sparse.csr_matrix(np.eye(k) / k)
        W[0, 0] = 1
        
        ### GMLS
        PhiInv = np.linalg.solve(Phi.T @ W @ Phi, Phi.T @ W)
        Gi = np.zeros((k,k,n)) # 1st order derivative square matrix
        for kk in range(n):
            Phiprime = np.zeros((k, term))
            II = np.zeros((k, n))
            II[:, kk] = np.ones(k)
            dinyy = II @ pvec.T / xnorm # (k*n)*(n*d)
            for ss in range(1, term): # the first term is always zero
                for rr in range(d): # derivative wrt the rr^th dimension
                    if index[rr, ss] == 0:
                        DiOne = np.zeros(k)                     
                    else:
                        DiOne = np.ones(k)
                        for tt in range(d):
                            if tt == rr:
                                # the only one derivative term in monomial
                                DiOne = DiOne * index[tt, ss] * inyy[:, tt] ** (index[tt, ss] - 1) * dinyy[:, tt]
                            else:
                                DiOne = DiOne * inyy[:, tt] ** index[tt, ss]
                    Phiprime[:, ss] += DiOne
                    
            ### compute Gi = Di*Phi^-1
            Gi[:, :, kk] = Phiprime @ PhiInv
        GiAugMent[:, :, pp] = Gi[0, :, :]
    return GiAugMent, inds

def solve_velocity(data, nu, gm, k0, degr, method):
    """
    PDE solver.

    Inputs:
    - data: dataset.
    - nu: viscosity nu.
    - gm: coupling coefficient gamma.
    - k0: number of neighbors of local stencil.
    - degr: degree of polynomials.
    - method: 1 is extrinsic, 2 is intrinsic and 3 is optim.

    Returns:
    - velocity vector u.
    """
    
    x = scipy.io.loadmat(data)
    x = x['p'] # point cloud

    ### compute the radius
    # am = np.linalg.norm(x[0, :], 2)
    am = 1

    ### x to theta
    N = x.shape[0]
    n = x.shape[1]
    THETA = np.array([math.acos(xx[2] / am) for xx in x])
    index1 = np.where((THETA != 0) & (THETA != math.pi))
    PHI = np.zeros(N);
    PHI[index1] = x[index1, 0] / am / np.array([math.sin(xx) for xx in THETA[index1]])
    PHI[index1] = np.array([math.acos(xx) for xx in PHI[index1]])
    posneg = np.ones(N)
    temp = (x[index1, 1] / am / np.array([math.sin(xx) for xx in THETA[index1]])).flatten()
    index2 = np.where(temp < 0)
    posneg[index2] = posneg[index2] * (-1)
    PHI = PHI * posneg
    theta = np.vstack((THETA, PHI)).T
    
    ### give the force or the force potential
    RHS = -6 * x[:, 0] - 66 * x[:,0] * x[:, 1] # analytic RHS
    RHS = RHS.reshape((N, 1))

    Kg = 1 / am ** 2 # Gaussian curvature
    
    P0, tvec2 = sphere_P0Cheat_v2(theta, x, am) # Projection matrix and orthonormal tangent vector
    
    ### solve potential Phi
    sol1, true_sol = solve_potenial(data, nu, gm, k0, degr, method)
    
    ### the Euclidean derivative of a function
    GiAugMent, inds = Gi_Save_v2_1K(x, k0, degr, tvec2)    
    extnormal = x/am
    uvel = np.zeros((N, n))
    for pp in range(N):
    
        ### (1*k)*(k*n)
        uvel[pp, :] = sol1[inds[pp, :]].T @ GiAugMent[:, :, pp]
        uvel[pp, :] = np.cross(uvel[pp, :], extnormal[pp, :])
    
    ### True solution of u is (0,-z,y) + (xz,-yz,-x^2+y^2)
    true_vec = np.zeros((N, n))
    true_vec[:, 0] = x[:, 0] * x[:, 2]
    true_vec[:, 1] = -x[:, 2] - x[:, 1] * x[:, 2]
    true_vec[:, 2] = x[:, 1] - x[:, 0] ** 2 + x[:, 1] ** 2
    return uvel, true_vec