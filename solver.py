import numpy as np

def gen_absorb(nx,nz,nabs,a,FreeSurf=False):
    absorb = np.ones((nx,nz))
    abs_coefs = np.zeros(nabs)
    abs_coefs = np.exp(-(a**2 * (nabs-np.arange(nabs))**2))
    absorb[:nabs,:] = absorb[:nabs,:]*np.expand_dims(abs_coefs,1)
    absorb[-nabs:,:] = absorb[-nabs:,:]*np.expand_dims(abs_coefs[::-1],1)
    absorb[:,-nabs:] = absorb[:,-nabs:]*abs_coefs[::-1]
    if(FreeSurf==False):
        absorb[:,:nabs] = absorb[:,:nabs]*abs_coefs
    return absorb

def comp_deriv(p,dd):
    pdx2 = np.zeros(p.shape)
    pdz2 = np.zeros(p.shape)

    pdx2[1:-1,1:-1] = (p[2:,1:-1]-2*p[1:-1,1:-1]+p[:-2,1:-1])/(dd**2)
    pdz2[1:-1,1:-1] = (p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,:-2])/(dd**2)
    return pdx2, pdz2

def fd_ac(vp, dd, dt, srcx, srcz, wav, nabs=40, a=0.0053, FreeSurf=False):
    
    srci = int(srcx)
    srcj = int(srcz)
    nx,nz = vp.shape # infer shapes
    print(f"grid size: nx = {nx}, nz = {nz}")
    nt = wav.shape[0]
    print(f"temporal grid size: nt = {nt}")

    field2d = np.zeros((nx, nz, nt), dtype=np.float) # define variables  - field2d is output wavefield
    p = np.zeros((nx,nz), dtype=np.float) # these are pressures at current, prev and next steps
    ppast = np.zeros((nx,nz),dtype=np.float)
    pfut = np.zeros((nx,nz),dtype=np.float)
    
    vp2 = vp**2 # square of velocity for easier computation
    absorb = gen_absorb(nx,nz,nabs,a,FreeSurf=FreeSurf) # generate absorbing mask
    
    for i in range(nt): # main loop
        pdx2, pdz2 = comp_deriv(p,dd) # compute pressure derivatives
        pfut = 2 * p + vp2 * dt**2 * (pdx2 + pdz2) - ppast # compute future pressure from current and prev 
        pfut[srci,srcj] = pfut[srci,srcj] + wav[i] / (dd * dd) * dt ** 2 # inject source term at selected point
        
        p *= absorb # apply absorbing mask
        pfut *= absorb # apply absorbing mask

        field2d[:,:,i] = p  # save current pressure in output array

        ppast = p # redefine arrays moving to next step
        p = pfut
        print('\rtime step = {}'.format(i), end='')
    print()
    return field2d