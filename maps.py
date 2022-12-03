import healpy as hp
import numpy as np
from utils import camb_clfile
import lenspyx
import os
import mpi
import toml


class CMBmap(object):
    def __init__(self,libdir,nside):
        self._libdir_ = os.path.join(libdir,f'Maps@{nside}')
        self.__def_dir__ = os.path.join(self._libdir_,'Deflection')
        self.__lensed_dir__ = os.path.join(self._libdir_,'Lensed')
        self.__glensed_dir__ = os.path.join(self._libdir_,'GaussianLensed')
        if mpi.rank == 0:
            os.makedirs(self.__def_dir__,exist_ok=True)
            os.makedirs(self.__lensed_dir__,exist_ok=True)
            os.makedirs(self.__glensed_dir__,exist_ok=True)
        self.__cl_path__ = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Data','CAMB')
        self.cl_unlensed = camb_clfile(os.path.join(self.__cl_path__,'BBSims_lenspotential.dat'))
        self.cl_lensed = camb_clfile(os.path.join(self.__cl_path__,'BBSims_lensed_dls.dat'))
        self.__seed__ = 261092
        self.nside = nside
        self.lmax = 3*nside-1

    @classmethod
    def from_config(cls,ini_file):
        config = toml.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Config',ini_file))
        return cls(config['Directory']['Main'],config['Maps']['nside'])

         
    def unlensed_alms(self,idx,dlmax=1024):
        Lmax = self.lmax + dlmax
        np.random.seed(self.__seed__+idx)
        T = hp.synalm(self.cl_unlensed['tt'],lmax=Lmax)
        E = hp.synalm(self.cl_unlensed['ee'],lmax=Lmax)
        return T,E
    
    def deflection_alms(self,idx,which='phi',dlmax=1024):
        fname = os.path.join(self.__def_dir__,f'phi_sims_{idx:04d}.fits')
        Lmax = self.lmax + dlmax
        if os.path.isfile(fname):
            P = hp.read_alm(fname)
        else:
            np.random.seed(self.__seed__+idx)
            P = hp.synalm(self.cl_unlensed['pp'],lmax=Lmax)
            hp.write_alm(fname,P)
        
        
        if which == 'phi':
            return P
        elif which == 'theta':
            return hp.almxfl(P, np.sqrt(np.arange(Lmax + 1, dtype=float) * np.arange(1, Lmax + 2)))
        else:
            raise ValueError('which must be phi or theta')

    def lensed_map(self,idx):
        fname = os.path.join(self.__lensed_dir__,f'cmb_sims_{idx:04d}.fits')
        if os.path.isfile(fname):
            return hp.read_map(fname,(0,1,2))
        else:
            T, E = self.unlensed_alms(idx)
            dlm = self.deflection_alms(idx,which='theta')
            Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], self.nside, 1, hp.Alm.getlmax(dlm.size))
            Tlen  = lenspyx.alm2lenmap(T, [Red, Imd], self.nside, facres=0, verbose=False)
            Qlen, Ulen  = lenspyx.alm2lenmap_spin([E, None], [Red, Imd], self.nside, 2, facres=0, verbose=False)
            hp.write_map(fname,[Tlen,Qlen,Ulen])
            return Tlen,Qlen,Ulen
    
    def lensed_gaussian_map(self,idx):
        fname = os.path.join(self.__glensed_dir__,f'cmb_sims_{idx:04d}.fits')
        if os.path.isfile(fname):
            return hp.read_map(fname,(0,1,2))
        else:
            np.random.seed(self.__seed__+idx)
            Cls = [self.cl_lensed['tt'],self.cl_lensed['ee'],self.cl_lensed['bb'],0*self.cl_lensed['te']]
            maps = hp.synfast(Cls,self.nside,new=True)
            hp.write_map(fname,maps)
            return maps
    

    
class SKYmap(object):

    def __init__(self,libdir,nside,depth_i,depth_p,fwhm,maskpath,fg=0):
        self.cmb_sim = CMBMap(libdir,nside)
        self._map_dir_ = os.path.join(self.libdir,f'Sky_{fg}','Maps')
        if mpi.rank == 0:
            os.makedirs(self._map_dir_,exist_ok=True)
        
        self.nside = nside
        self.lmax = 3*nside-1
        self.depth_i = depth_i
        self.depth_p = depth_p
        self.fwhm = fwhm
        self.mask = hp.read_map(maskpath)
        if hp.get_nside(self.mask) != self.nside:
            self.mask = hp.ud_grade(self.mask,self.nside)
        self.fg = fg
    
    @classmethod
    def from_config(cls,ini_file):
        config = toml.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Config',ini_file))
        mconfig = config['Maps']
        dir = config['Directory']['Main']
        nside = int(mconfig['nside'])
        noise_t = float(mconfig['noise_t'])
        noise_p = float(noise_t * np.sqrt(2))
        fwhm = float(mconfig['fwhm'])
        maskpath = mconfig['maskpath']
        fg = int(mconfig['fg'])
        return cls(dir,nside,noise_t,noise_p,fwhm,maskpath,fg)
        
    def get_map(self,idx,which='lensed'):
        fname = os.path.join(self._map_dir_,f'sky_sims_{idx:04d}.fits')
        if os.path.isfile(fname):
            return hp.read_alm(fname,(1,2,3))
        else:
            if which == 'lensed':
                TQU = self.cmb_sim.lensed_map(idx)
            elif which == 'gaussian':
                TQU = self.cmb_sim.lensed_gaussian_map(idx)
            else:
                raise ValueError('which must be lensed or gaussian')
   
            if self.fg != 0:
                raise NotImplementedError('Foregrounds not implemented yet')

            TQU = hp.smoothing(TQU,fwhm=np.radians(self.fwhm/60),lmax=self.lmax) + self.get_noise()
            alms = hp.map2alm(TQU*self.mask,lmax=self.lmax)
            hp.write_alm(fname,alms)
            return alms

    def add_fg(self,TQU):
        pass
    
    def get_noise(self):
        npix = hp.nside2npix(self.nside)
        pix_amin2 = 4. * np.pi / float(npix) * (180. * 60. / np.pi) ** 2

        sigma_pix_I = np.sqrt(self.depth_i ** 2 / pix_amin2)
        sigma_pix_P = np.sqrt(self.depth_p ** 2 / pix_amin2)

        
        noise = np.random.randn(3, npix)
        noise[0,:] *= sigma_pix_I
        noise[1,:] *= sigma_pix_P
        noise[2,:] *= sigma_pix_P
        return noise
    
    def get_sim_tlm(self,idx):
        fname = os.path.join(self._map_dir_,f'sky_sims_{idx:04d}.fits')
        return hp.read_alm(fname,1)

    def get_sim_elm(self,idx):
        fname = os.path.join(self._map_dir_,f'sky_sims_{idx:04d}.fits')
        return hp.read_alm(fname,2)

    def get_sim_blm(self,idx):
        fname = os.path.join(self._map_dir_,f'sky_sims_{idx:04d}.fits')
        return hp.read_alm(fname,3)

    def get_sim_tmap(self,idx):
        tlm = self.get_sim_tlm(idx)
        tmap = hp.alm2map(tlm,self.nside)
        del tlm
        return tmap

    def get_sim_pmap(self,idx):
        elm = self.get_sim_elm(idx)
        blm = self.get_sim_blm(idx)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2,hp.Alm.getlmax(elm.size))
        del elm,blm
        return Q ,U
