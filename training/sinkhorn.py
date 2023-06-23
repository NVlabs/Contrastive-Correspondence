# ------------------------------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Taihong Xiao
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F

# version 1

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    # from IPython import embed; embed(); exit()

    return Z


# version 2

def perform_sinkhorn(C, epsilon, mu, nu, warm=False, niter=1, tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],C.shape[1])) / C.shape[1]
        a = a.to(C.device) # (b, m)

    K = torch.exp(-C/epsilon) # (b, m, n)

    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        # from IPython import embed; embed(); exit()
        b = nu/torch.einsum('bmn,bm->bn', K, a) # (b, n)
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.einsum('bmn,bn->bm', K, b)) - mu, p=1) # (b, m)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.einsum('bmn,bn->bm', K, b) # (b, m)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.einsum('bmn,bm->bn', K, a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.bmm(torch.bmm(torch.diag_embed(a), K), torch.diag_embed(b))
        # PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI,mu,nu,Err


def OT(cost, eps=0.05, exp2=0.5):
    b, m, n = cost.shape
    mu = (torch.ones((b,m))/m).to(cost.device)
    nu = (torch.ones((b,n))/n).to(cost.device)

    epsilon = eps
    cnt = 0
    while True:
        PI,a,b,err = perform_sinkhorn(cost, epsilon, mu, nu)
        #PI = sinkhorn_stabilized(mu, nu, cost, reg=epsilon, numItermax=50, method='sinkhorn_stabilized', cuda=True)
        if not torch.isnan(PI).any():
            if cnt>0:
                print(cnt)
            break
        else: # Nan encountered caused by overflow issue is sinkhorn
            epsilon *= 2.0
            #print(epsilon)
            cnt += 1

    PI = m*PI # re-scale PI 
    #exp2 = 1.0 for spair-71k, TSS
    #exp2 = 0.5 # for pf-pascal and pfwillow
    PI = torch.pow(torch.clamp(PI, min=0), exp2)
    return PI

if __name__ == '__main__':
    # aff = torch.rand(2,3,4).cuda(0)
    aff = torch.ones(2,3,4).cuda(0)
    aff = F.softmax(aff, -1)
    scores = 1 - aff
    alpha = torch.tensor(1, dtype=torch.float32).cuda(0)

    out1 = log_optimal_transport(scores, alpha, 10)

    out2 = OT(scores)
    from IPython import embed; embed(); exit()
