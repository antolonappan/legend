import plancklens



class FILTmap(object):

    def __init__(self,libdir,sim_lib,nside,transf,maskpath,nlev_p,nlev_t,cl_len,lmin_ivf,lmax_ivf):
        self.sim_lib = sim_lib

ninv_t = [np.array([3. / nlev_t ** 2])] + maskpaths
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p = [[np.array([3. / nlev_p ** 2])] + maskpaths]
cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p)

ivfs_raw    = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len)
ftl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf) # rescaling or cuts. Here just a lmin cut
fel = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
fbl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl, fel, fbl)

