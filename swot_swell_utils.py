#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for manipulating SWOT SSH data for swell analysis

Author: Fabrice Ardhuin 
Date: First version:  04.13.2024

Date: latest version: 04.13.2024


Dependencies:
    numpy, xarray, datetime, scipy
    wave_physics_functions
"""

import numpy as np
import datetime
import xarray as xr
import glob as glob
import os

import cmocean
import cmocean.cm as cmo
lightcmap = cmocean.tools.lighten(cmo.ice, 1)
import matplotlib.colors as mcolors

import swot_ssh_utils as swot
from wave_physics_functions import wavespec_Efth_to_Ekxky, wavespec_Efth_to_first3



from numpy.ma import masked_array
from scipy import ndimage
from  spectral_analysis_functions import *
from  lib_filters_obp import *

from netCDF4 import Dataset

###################################################################
def spec_settings_for_L3(nres,version):
    dx=250
    dy=235
    indxc=259   # index of center pixel (at nadir) 
#    ISHIFT=30   # start of 40 km box for spectral analysis, in pixels, relative to track center
    ISHIFT=60   # start of 40 km box for spectral analysis, in pixels, relative to track center
    nkxr=20
    nkyr=21
    res_table=np.array([40,20,10])
    restab=res_table[0:nres]
    nX2tab=np.array([80,40,20])
    nY2tab=np.array([84,42,21])
    if (version == 'alpha'):
       samemask=1
       mtab=np.array([8,4,2])
       ntab=np.array([8,4,2])
    if (version == 'beta'):
       samemask=0
       mtab=np.array([8,2,2])
       ntab=np.array([8,2,2])
    if (version == 'gamma'):
       samemask=0
       mtab=np.array([8,2,1])
       ntab=np.array([8,2,1])
    
    
    indl=421 # alongtrack length of "chunk" of SWOT data being processed: this relatively big number is there because of the movies to get context.
    dind=84  # increment in alongtrack number of pixels   restab[0]*2 #int(abs(ddlat)*400)  # rough converstion 1 deg is 100 km and resolution is about 0.25 km
    dind=dind*2  # this is the value used by CNES 
    hemiNS=['A','N','S']
    hemiWE=['A','E','W']
    return dx,dy,indxc,ISHIFT,nkxr,nkyr,restab,nX2tab,nY2tab,mtab,ntab,indl,dind,samemask,hemiNS,hemiWE


###################################################################
def wavespec_Efth_to_kxky_SWOT(efth,modf,moddf, modang,moddth,f_xt,f_at,H,Hazc,H3,kxmax,kymax,dkx,dky,dkxr,dkyr,nkxr,nkyr,depth=3000.,doublesided=0,verbose=0,trackangle=0)  :

    dkxf=dkx/3;dkyf=dky/3;   # finer spectral resolution 
    nkx=600;nky=600;	     # not sure this is always high enough ... 

    Ekxky,kxm,kym,kx2m,ky2m=wavespec_Efth_to_Ekxky(efth,modf,moddf,modang,moddth, \
          depth=depth,dkx=dkxf,dky=dkyf,nkx=nkx,nky=nky,doublesided=doublesided,verbose=verbose,trackangle=trackangle)
          
    nxavg=round(dkxr/dkxf)   # number of spectral pixels to average
    nyavg=round(dkyr/dkyf)

    ik1=(nkxr+1)//2;ik2=ik1+nkxr
    jk1=(nkyr+1)//2;jk2=jk1+nkyr

    ishift=(1-np.mod(nkxr,2))
    jshift=(1-np.mod(nkyr,2))
    ix1=int(nkx-kxmax/dkxf)+nxavg*(ishift-1)
    iy1=int(nky-kymax/dkyf)+nyavg*(jshift-1)
    di1=-(nxavg//2); di2=di1+nxavg
    dj1=-(nyavg//2); dj2=dj1+nyavg


# Coarsening of WW3 spectrum on kx,ky grid 
    Ekxkyr=np.zeros((nkxr*2,nkyr*2))
    Ekxkyp=np.zeros((nkxr*2,nkyr*2))
# We have to deal with the non-symmetry of the spectrum : hence the np.roll 
    Ekxkyds=0.5*(Ekxky+np.fliplr(np.roll( np.flipud(np.roll(Ekxky,-1,axis=0)),-1,axis=1) ))
    Ekxkydp=np.fliplr(np.roll( np.flipud(np.roll(Ekxky,-1,axis=0)),-1,axis=1) )
# Coarsening of WW3 spectrum on kx,ky grid 
# We have to deal with the non-symmetry of the spectrum for even numbers (nkxr or nkyr) 
    for ix in range(nkxr*2): 
       for iy in range(nkyr*2): 
          Ekxkyr[ix,iy]=np.mean(Ekxkyds[ix1+ix*nxavg+di1:ix1+ix*nxavg+di2,iy1+iy*nyavg+dj1:iy1+iy*nyavg+dj2].flatten())
          Ekxkyp[ix,iy]=np.mean(Ekxkydp[ix1+ix*nxavg+di1:ix1+ix*nxavg+di2,iy1+iy*nyavg+dj1:iy1+iy*nyavg+dj2].flatten())


    Eta_WW3=Ekxkyr   # this is the WW3 spectrum on SWOT grid + double-sided
        
    Sw_obp_H  = H  * Hazc * Eta_WW3   #  without PTR ... should be removed 
    Sw_obp_H2 = H3 * Eta_WW3
# 4) Downsample in space to the target spatial frequency
    fx_alias, fy_alias, Sw_alias_H = compute_aliased_spectrum_2D(f_xt, f_at, Sw_obp_H, 1/0.250, 1/0.235, nrep=1)
    fx_alias, fy_alias, Sw_alias_H2 = compute_aliased_spectrum_2D(f_xt, f_at, Sw_obp_H2, 1/0.250, 1/0.235, nrep=1)
    
    
    # SOME CLEAN UP WILL BE GOOD BELOW ... 
    Eta_WW3_c=Ekxkyr[ik1:ik2,jk1:jk2].T           # this is the WW3 spectrum*OBP filter with alias effect 
    Eta_WW3_obp_H=Sw_alias_H[ik1:ik2,jk1:jk2].T   # without az cut-off effect
    Eta_WW3_obp_H2=Sw_alias_H2[ik1:ik2,jk1:jk2].T
# Also computes the spectrum without aliasing to check on filter + aliasing effects 
    Eta_WW3_noa_H2=Sw_obp_H2[ik1:ik2,jk1:jk2].T
    Eta_WW3_res= H3[ik1:ik2,jk1:jk2].T * Ekxkyp[ik1:ik2,jk1:jk2].T

    return Eta_WW3_obp_H2,Eta_WW3_obp_H,Eta_WW3_noa_H2,Eta_WW3_res,Eta_WW3_c,Ekxky,kxm,kym,ix1,iy1
    
###################################################################
def  SWOTspec_to_HsLm(Ekxky,kx2,ky2,swell_mask,Hhat2,trackangle)  :
    '''
    Computes parameters from SWOT ssh spectrum
    inputs :
            - Ekxky  : spectrum
            - kx2,ky2: 2D array of wavenumbers
            - swell_mask:  mask for selected area of spectrum (do not have to be binary) 
	    - Hhat2 : squared FT of the filtering function 
            - trackangle: direction of satellite track: used to shift directions back to nautical convention  
    output : 
            - Hs ... 

    '''
    nmask=np.sum(swell_mask.astype(int).flatten())
    if nmask > 0:
        E_mask=np.where( swell_mask > 0.5, np.divide(Ekxky,Hhat2),0) 
        dkx=kx2[0,1]-kx2[0,0]
        dky=ky2[1,0]-ky2[0,0]
        varmask=np.sum(E_mask.flatten())*dkx*dky*2;  # WARNING: factor 2  is only correct if the mask is only over half of the spectral domain!!
        Hs=4*np.sqrt(varmask)

# Corrects for 180 shift in direction
        shift180=0
        if np.abs(trackangle) > 90: 
          shift180=180

        kn=np.sqrt(kx2**2+ky2**2)
        m0=np.sum(E_mask.flatten())
        mQ=np.sum((E_mask.flatten())**2)
        Q18=np.sqrt(mQ/(m0**2*dkx*dky))/(2*np.pi)
        inds=np.where(kn.flatten() > 5E-4)[0]
        mm1=np.sum(E_mask.flatten()[inds]/kn.flatten()[inds])
        mmE=np.sum(E_mask.flatten()[inds]/np.sqrt(kn.flatten()[inds]))
        mp1=np.sum(np.multiply(E_mask,kn).flatten())
        if m0 > 1E-6:
           Lmm1=mm1/m0
           LE  =(mmE/m0)**2
           Lmp1=m0/mp1
           a1=np.sum(np.multiply(kx2.flatten()[inds]/kn.flatten()[inds],E_mask.flatten()[inds]))/m0
           b1=np.sum(np.multiply(ky2.flatten()[inds]/kn.flatten()[inds],E_mask.flatten()[inds]))/m0
        else:
           Lmm1=0.
           LE=0.
           Lmp1=0.
           a1=0.
           b1=0.		           
        sigth=np.sqrt(2*(1-np.sqrt(a1**2+b1**2)))*180/np.pi
        dm=np.mod(90-(-trackangle+shift180+np.arctan2(b1,a1)*180/np.pi),360) # converted to direction from, nautical
    else :
        Hs=np.nan,Lmm1=np.nan,Lmp1=np.nan,LE=np.nan,dm=np.nan,sigth=np.nan,Q18=np.nan
    return Hs,Lmm1,LE,Lmp1,dm,sigth,Q18


###################################################################
def  SWOTspec_mask_polygon(amask)  :
    '''
    Computes parameters from SWOT ssh spectrum
    inputs :
            - Eta  : spectrum
            - 
    output : 
            - vertices

    '''
    masked_data = masked_array(amask, mask=(amask > 0.5) )
    # Get coordinates of masked cells
    rows, cols = np.where(masked_data.mask)
    # Iterate through masked cells and draw polygons around them
    vertices = []
    [nkyr,nkxr]=np.shape(amask)  
# Computes polygon for showing the mask position  
    for r, c in zip(rows, cols):
    # Define coordinates of polygon vertices
    # Check neighboring cells to determine vertices
          if r == 0 or (r > 0 and not masked_data.mask[r-1, c]) :
              vertices.extend([c - 0.5, c + 0.5, r - 0.5, r - 0.5])
          if r == nkyr-1 or (r < nkyr-1 and not masked_data.mask[r+1, c]):
              vertices.extend([c - 0.5, c + 0.5, r + 0.5, r + 0.5])
          if c == 0 or (c > 0 and not masked_data.mask[r, c-1]):
              vertices.extend([c - 0.5, c - 0.5, r - 0.5, r + 0.5])
          if c == nkxr-1 or (c < nkxr-1 and not masked_data.mask[r, c+1]):
              vertices.extend([c + 0.5, c + 0.5, r - 0.5, r + 0.5])

    return vertices

###################################################################
# mybox,mybos,flbox,X,Y,sflip,signMTF,Look=SWOTarray_flip_north_up(dlat,side,ssha[j1:j2,i1:i2],flas[j1:j2,i1:i2],sig0[j1:j2,i1:i2],X,Y)
def  SWOTarray_flip_north_up(dlat,side,ssha,flas,sig0,X,Y)  :
# flips the array so that the image is rotated to have roughly the north on top and the south on bottom.
    if dlat < 0:
     sflip=0
#for descending
     if side == 'left':
       signMTF=1;
       Look=1;
       Y=-np.flipud(Y)
       mybox=(np.flipud(ssha))
       flbox=(np.flipud(flas))
       mybos=(np.flipud(sig0))
     else:
       signMTF=-1;
       Look=-1
       mybox=np.fliplr(np.flipud(ssha))
       flbox=np.fliplr(np.flipud(flas))
       mybos=np.fliplr(np.flipud(sig0))
       Y=-np.flipud(Y)
       X=-np.flipud(X)
    else:
#for ascending
      sflip=1
      if side == 'left':
        signMTF=1;
        X=-np.flipud(X)
        Look=-1
        mybox=(np.fliplr(ssha))
        flbox=(np.fliplr(flas))
        mybos=(np.fliplr(sig0))
      else:
        Look=1;
        signMTF=-1;
        mybox=ssha
        flbox=flas
        mybos=sig0
    return   mybox,mybos,flbox,X,Y,sflip,signMTF,Look

###################################################################
# modpec,inds,modelfound,timeww3,dist=swell.SWOTfind_model_spectrum(ds_ww3t,loncr,latcr,timec)
def SWOTfind_model_spectrum(ds_ww3t,loncr,latcr,timec) :
        times=str(timec)[0:13]+':00:00'
        format = '%Y-%m-%dT%H:%M:%S'
        timed  =datetime.datetime.strptime(times,format)
        timemin=int(str(timec)[14:16])
        timeww3=np.datetime64(times) #np.timedelta64(1800, 's')    
        if timemin >= 15:
            timeww3=np.datetime64(times) + np.timedelta64(1800, 's')    
        if timemin >= 45:
            timeww3=np.datetime64(times) + np.timedelta64(1, 'h')    

        loncr180 = np.where( loncr > 180, loncr-360,loncr) 

        timeww3a=timeww3+np.timedelta64(1800, 's')    
        timeww3b=timeww3-np.timedelta64(1800, 's')    
        indt= np.where(ds_ww3t.time==timeww3)[0]
        modelfound=0;dist=1;
        lonww3=0.;latww3=0.;dpt=3000.;  
        if (len(indt) >0): 
            sinlat=np.abs(np.sin(latcr*np.pi/180))
            dd= sinlat*np.abs(np.cos(ds_ww3t.longitude[indt].values*np.pi/180)-np.cos(loncr*np.pi/180)) \
               +sinlat*np.abs(np.sin(ds_ww3t.longitude[indt].values*np.pi/180)-np.sin(loncr*np.pi/180)) \
               +np.abs(ds_ww3t.latitude[indt].values-latcr)*np.pi/180
            min_index=np.argmin(dd)
            dist=dd[min_index]
            inds=indt[min_index]
            lonww3=ds_ww3t.longitude[inds].values
            latww3=ds_ww3t.latitude[inds].values
            dpt=ds_ww3t.dpt[inds].values
            #print('COUCOU lon:',loncr,latcr,timec,timeww3a,'##',len(indt),dist,ds_ww3t.longitude[inds].values,ds_ww3t.latitude[inds].values)
            modspec=ds_ww3t.efth[inds].squeeze()
            U10=ds_ww3t.wnd[inds].values
            Udir=ds_ww3t.wnddir[inds].values

            #fig,axs=plt.subplots(1,1,figsize=(7,7))
            #axs.scatter(lonc,latc,c='g',linewidth=8,label='model');
            #axs.scatter(loncr,latcr,c='b',linewidth=4,label='model');
            #axs.scatter(ds_ww3t.longitude[indt],ds_ww3t.latitude[indt],c='k',linewidth=4,label='model');
            #axs.scatter(ds_ww3t.longitude[inds],ds_ww3t.latitude[inds],c='r',linewidth=2,label='model');
            #axs.set_xlim([loncr-2,loncr+2])
            #axs.set_ylim([latcr-2,latcr+2])

        if dist < 0.02:
            modelfound=1
        else: 
            modspec=[]
            U10=[]
            Udir=[]
            inds=0
            print('Did not find model spectrum for location (lon,lat):',loncr,latcr,' at time ',timeww3, \
                  '. Min dist was:',dist,'for this model point:', lonww3,latww3)
        return modspec,inds,modelfound,timeww3,lonww3,latww3,dist,U10,Udir,dpt


###################################################################
###################################################################
def SWOTdefine_swell_mask_simple(Eta,coh,ang,medsig0,dlat,kx2,ky2,cohthr=0.3,cfac=8,mask_choice=0,\
             kxmin=-0.003,kxmax=0.003,kymin=-0.003,kymax=0.003,verbose=0):
    cohm=coh
    kx2m=kx2
    ky2m=ky2
    angm=ang
    Etam=Eta
    cohthr2=cohthr
    
    kn=np.sqrt(kx2**2+ky2**2)*1000
    
    indc=np.where((cohm > cohthr))[0]
    ncoh=len(indc)
    kp=np.where(kx2m > 0,1.,0.)
    kpy=np.where(ky2m > 0,1.,0.)
    ncoh2=0
    #if ncoh < 6:
    cohmax=np.max(cohm.flatten())
    cohthr2=cohmax/2
    indc2=np.where((cohm > cohthr2) )[0]
    ncoh2=len(indc2)
    cohOK=1
    maskset=0
    if ncoh2 > 3:
       ncoh=ncoh2
    if cohthr > 0.8/cfac and ncoh > 3 and (medsig0 < 70):
       cohs=(cohm/cohthr)*np.sign(np.cos(angm))*np.sign(ky2m*dlat)
       amask=ndimage.binary_dilation((cohs > 1 ).astype(int))
       maskset=1
    else:
       cohOK=0
       amask=Etam/10*np.sign(ky2m*dlat)
       maskset=2
    if (cohOK==1) or (mask_choice == -1):
       Emax=2/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m)
       maskset=3
    if (mask_choice == -2):
       Emax=10/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m*dlat)
       maskset=4
    if (mask_choice == -3):
       Emax=2/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m*dlat)
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       maskset=5
    if (mask_choice == -4):
       Etam=np.where(abs(kx2) <= 0.0012,Etam,0)
       Emax=2/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m*dlat)
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       maskset=6
    if (mask_choice == -5):
       Etam=np.where(kx2 >= kxmin,Etam,0)
       Etam=np.where(kx2 <= kxmax,Etam,0)
       Etam=np.where(ky2 >= kymin,Etam,0)
       Etam=np.where(ky2 <= kymax,Etam,0)
       #print('Emax:',np.max(Etam.flatten()),kxmin,kxmax,kymin,kymax)
       Emax=4/np.max(Etam.flatten())
       amask=Etam*Emax
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       maskset=7

    
    ind=np.where(amask.flatten() > 0.5)[0]
    if len(ind) >0 :
        
      indk=np.argmax(Etam.flatten()[ind])
    # defines k magnitude at spectral peak for further filtering ...   
      knm=np.sqrt(kx2m**2+ky2m**2)*1000

      knmax=knm.flatten()[ind[indk]]
      rows, cols =np.where(kn > 2*knmax  )
      for r, c in zip(rows, cols):               
        amask[r,c]=0
      rows, cols =np.where(kn < 0.6*knmax  )
      for r, c in zip(rows, cols):               
        amask[r,c]=0
      
    # Forces mask : here are a few choices ... 
       
    if mask_choice==0:
        amask=np.multiply(np.where(abs(kx2-0.5*(kxmin+kxmax)) <= 0.5*(kxmax-kxmin),1,0),   \
                          np.where(abs(ky2-0.5*(kymin+kymax)) <= 0.5*(kymax-kymin),1,0))
        maskset=10
    if mask_choice==1:
        amask=np.multiply(np.where(abs(kx2) <= 0.0006,1,0),np.where(abs(ky2-0.0015) <= 0.0005,1,0))
        maskset=11
    if mask_choice==2:
    #Forcing for Norfolk island swell on track 17
        amask=np.multiply(np.where(abs(kx2+0.001) <= 0.0003,1,0),np.where(abs(ky2-0.000) <= 0.0004,1,0))
        maskset=12

    bmask=ndimage.binary_dilation((amask > 0.5).astype(int))

    print('Swell mask option:',maskset,cohOK,mask_choice,cohthr,cohmax,ncoh,medsig0)

    return amask,bmask

###################################################################
# modpec,inds=swell.SWOTfind_model_spectrum(ds_ww3t,loncr,latcr,timec)
def SWOTdefine_swell_mask(mybox,mybos,flbox,dy,dx,nm,mm,Eta,coh,ang,dlat,mask_choice,kx2,ky2,kn,cohthr,cfac,n,m,nkxr,nkyr,\
             kxmin=-0.003,kxmax=0.003,kymin=-0.003,kymax=0.003,verbose=0):
    cohm=coh
    kx2m=kx2
    ky2m=ky2
    angm=ang
    Etam=Eta
    cohthr2=cohthr
    medsig0=np.nanmedian(mybos)

    if ((nm > n) or (mm > m)):
        # Recomputes spectra just for masking purposes
        (Etam,Etbm,angm,angstdm,cohm,crosrm,phasesm,ky2m,kx2m,dkym,dkxm,detrendam,detrendbm,nspecm)=FFT2D_two_arrays_nm_detrend_flag(mybox,10**(0.1*mybos),flbox,dy,dx,nm,mm,detrend='quadratic')

    indc=np.where((cohm > cohthr))[0]
    ncoh=len(indc)
    kp=np.where(kx2m > 0,1.,0.)
    kpy=np.where(ky2m > 0,1.,0.)
    ncoh2=0
    #if ncoh < 6:
    cohmax=np.max(cohm.flatten())
    cohthr2=cohmax/2
    indc2=np.where((cohm > cohthr2) )[0]
    ncoh2=len(indc2)
    cohOK=1
    maskset=0
    if ncoh2 > 3:
       ncoh=ncoh2
    if cohthr > 0.8/cfac and ncoh > 3 and (np.nanmedian(mybos) < 70):
       cohs=(cohm/cohthr)*np.sign(np.cos(angm))*np.sign(ky2m*dlat)
       amask=ndimage.binary_dilation((cohs > 1 ).astype(int))
       maskset=1
    else:
       cohOK=0
       amask=Etam/10*np.sign(ky2m*dlat)
       maskset=2
    if (cohOK==1) or (mask_choice == -1):
       Emax=2/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m)
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       maskset=3
    if (mask_choice == -2):
       Emax=10/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m*dlat)
       maskset=4
    if (mask_choice == -3):
       Emax=2/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m*dlat)
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       maskset=5
    if (mask_choice == -4):
       Etam=np.where(abs(kx2) <= 0.0012,Etam,0)
       Emax=2/np.max(Etam.flatten())
       amask=Etam*Emax*np.sign(-ky2m*dlat)
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       maskset=6
    if (mask_choice == -5):
       Etam=np.where(kx2 >= kxmin,Etam,0)
       Etam=np.where(kx2 <= kxmax,Etam,0)
       Etam=np.where(ky2 >= kymin,Etam,0)
       Etam=np.where(ky2 <= kymax,Etam,0)
       #print('Emax:',np.max(Etam.flatten()),kxmin,kxmax,kymin,kymax)
       Emax=4/np.max(Etam.flatten())
       amask=Etam*Emax
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       maskset=7

    ind=np.where(amask.flatten() > 0.5)[0]
    if len(ind) >0 :
      if ((nm > n) or (mm > m)):
        cfac=np.sqrt(nm*mm)
        amaskc=amask
        amask=Eta*0
        # Recopies coarse mask on fine grid 
        
        masked_data = masked_array(Etam, mask=(amaskc > 0.5) )
        # Get coordinates of masked cells
        rows, cols = np.where(masked_data.mask)
        dmc=mm//m
        dnr=nm//n
        for r, c in zip(rows, cols):
          jm1=np.max([0,r*dnr-dnr//2]);
          jm2=np.min([nkyr,r*dnr+dnr//2+1]);
          im1=np.max([0,c*dmc-dmc//2]);
          im2=np.min([nkxr,c*dmc+dmc//2+1]);
          amask[jm1:jm2,im1:im2]=1
        
      indk=np.argmax(Etam.flatten()[ind])
    # defines k magnitude at spectral peak for further filtering ...   
      knm=np.sqrt(kx2m**2+ky2m**2)*1000

      knmax=knm.flatten()[ind[indk]]
      rows, cols =np.where(kn > 2*knmax  )
      for r, c in zip(rows, cols):               
        amask[r,c]=0
      rows, cols =np.where(kn < 0.6*knmax  )
      for r, c in zip(rows, cols):               
        amask[r,c]=0
      
    # Forces mask : here are a few choices ... 
    if mask_choice==0:
        amask=np.multiply(np.where(abs(kx2-0.5*(kxmin+kxmax)) <= 0.5*(kxmax-kxmin),1,0),   \
                          np.where(abs(ky2-0.5*(kymin+kymax)) <= 0.5*(kymax-kymin),1,0))
        maskset=10
    if mask_choice==1:
        amask=np.multiply(np.where(abs(kx2) <= 0.0006,1,0),np.where(abs(ky2-0.0015) <= 0.0005,1,0))
        maskset=11
    if mask_choice==2:
    #Forcing for Norfolk island swell on track 17
        amask=np.multiply(np.where(abs(kx2+0.001) <= 0.0003,1,0),np.where(abs(ky2-0.000) <= 0.0004,1,0))
        maskset=12

    bmask=ndimage.binary_dilation((amask > 0.5).astype(int))

    if verbose > 0:
       print('Swell mask option:',maskset,cohOK,mask_choice,cohthr,cohmax,ncoh,medsig0)

    return amask,bmask


#######################" Plotting routines ########################
def draw_mask(axsfig,kx2,dkx,ky2,dky,vertices,color='k',lw=3):
    for ind in range(len(vertices) // 4):
        xy2=np.asarray(vertices[ind*4:(ind+1)*4], dtype=np.float64)
        axsfig.plot(kx2[0,0]*1000+1000*dkx*xy2[0:2],ky2[0,0]*1000+1000*dky*xy2[2:4], color=color,lw=lw)


#######################" Plotting routines ########################
def arrows_on_spectrum(axsfig,sflip,side,trackangle,color='k',lw=3,fs1=20):
    arx0=0;ary0=0;arxd=0.2;aryd=0.2;arwid=0.1;gr=[0.,1,0.]
    flip=-1+2*(1-sflip)
    sidf=1-2*(1-side)
    axsfig.arrow(arx0, ary0, -arxd*np.sin(trackangle*np.pi/180), aryd*np.cos(trackangle*np.pi/180), linewidth=4,color='k',head_width=arwid) 
    axsfig.text(arx0-arxd*1.6*np.sin(trackangle*np.pi/180),ary0+aryd*1.6*np.cos(trackangle*np.pi/180),'N',fontsize=fs1)
    axsfig.arrow(arx0, ary0, 0., aryd*flip, linewidth=4,color=gr,head_width=arwid) 
    axsfig.text(arx0+0.4*arxd,ary0+aryd*1.2*flip,'Vsat',fontsize=fs1,color=gr)
    axsfig.arrow(arx0, ary0 ,  arxd*sidf*flip, 0, linewidth=4,color=gr,head_width=arwid) 
    axsfig.text(arx0+arxd*(1.8*sidf*flip-0.8),ary0-aryd,'Look',fontsize=fs1,color=gr)

#######################" Plotting routines ########################

def plot_cur(axs,td,xt,yt,latc,globlon,globlat,U,V,lightcmap):
       if (td =='descending'):
          ind=np.where(yt >= latc)[0]
       else:
          ind=np.where(yt <= latc)[0]
       im=axs[1].pcolormesh(globlon,globlat,np.sqrt(U**2+V**2), cmap=lightcmap,rasterized=True,shading='nearest',vmin = 0, vmax =1)
       _=axs[1].set_title('Globcurrent map' )
       plt.setp(axs[1].get_yticklabels(), visible=False)
       axs[1].scatter(xt[ind],yt[ind],c='r',marker='+',s=30,linewidth=2)


def plot_spec(kx2,dkx,ky2,dky,Eta,dBE,vertices):
    fig,axs= plt.subplots(nrows=1, ncols=2,figsize=(6,3.5))
    #spec = mpl.gridspec.GridSpec(ncols=2, nrows=1,width_ratios=[6, 5])
    plt.subplots_adjust(left=0.05,bottom=0.10, top=0.92,wspace=0.1,right=0.99)
    im=axs[0].pcolormesh(kx2*1000,ky2*1000,10*np.log10((Eta)),norm = mcolors.Normalize(vmin=-10+dBE, vmax=30+dBE),rasterized=True)
    _=axs[0].set_title('SWOT spectrum' )
    draw_mask(axs[0],kx2,dkx,ky2,dky,vertices,color='w',lw=3)
    return fig,axs
    
def plot_coh(kx2,dkx,ky2,dky,coh,ang,vertices):
    fig,axs=plt.subplots(1,2,figsize=(6,3))
    plt.subplots_adjust(left=0.05,bottom=0.1, top=0.92,wspace=0.1,right=0.99)
    im=axs[0].pcolormesh(kx2*1000,ky2*1000,coh,cmap='viridis',rasterized=True,vmin = 0., vmax =1)
    draw_mask(axs[0],kx2,dkx,ky2,dky,vertices,color='w',lw=3)
    _=axs[0].set_title('coherence')
      
    im=axs[1].pcolormesh(kx2*1000,ky2*1000,ang,cmap='seismic',rasterized=True,norm = mcolors.Normalize(vmin=-180, vmax=180))
    _=axs[1].set_title('phase' )
    draw_mask(axs[1],kx2,dkx,ky2,dky,vertices,color='k',lw=3)
    plt.setp(axs[1].get_yticklabels(), visible=False)
    return fig,axs

###################################################################
def  SWOT_save_spectra(pth_results,filenopath,modelfound,cycle,tracks,side,boxindices,\
                       lonc,latc,timec,trackangle,kx2,ky2,Eta,Etb,coh,ang,crosr,amask,sig0mean,sig0std,HH,HH2,Hs_SWOT_all,Hs_SWOT,Hs_SWOT_mask,Lm_SWOT,dm_SWOT, \
                       timeww3=0,lonww3=0,latww3=0,indww3=0,distww3=0,E_WW3_obp_H=0,E_WW3_obp_H2=0,E_WW3_noa_H2=0,Hs_WW3_all=0,Hs_WW3_cut=0,\
                       Hs_WW3_mask=0,Hs=0,Tm0m1=0,Tm02=0,Qkk=0,U10=0,Udir=0,Lm_WW3=0,dm_WW3=0, verbose=0)  :
   hemiNS=['A','N','S']
   hemiWE=['A','E','W']
   lonlat=f'{abs(latc):05.2f}'+hemiNS[int(np.sign(latc))]
   if modelfound==1:
       np.savez(pth_results+'SWOT_swell_spectra_'+cycle+'_'+tracks+'_'+side+'_'+lonlat,\
                fileSWOT=filenopath,cycle=cycle,tracks=tracks,side=side,boxindices=boxindices,\
                lonc=lonc,latc=latc,timec=timec,trackangle=trackangle,\
                kx2=kx2,ky2=ky2,E_SWOT=Eta,sig0_spec=Etb,coh=coh,ang=ang,crosr=crosr,amask=amask,\
                sig0mean=sig0mean,sig0std=sig0std,HH=HH,HH2=HH2, \
                Hs_SWOT_filtered_all=Hs_SWOT_all,Hs_SWOT_filtered_mask=Hs_SWOT,Hs_SWOT_mask=Hs_SWOT_mask,\
                Lm_SWOT_filtered_mask=Lm_SWOT,dm_SWOT_filtered_mask=dm_SWOT, \
                modelfound=modelfound,timeww3=timeww3,lonww3=lonww3,latww3=latww3,indww3=indww3,distww3=distww3,\
                E_WW3_obp_H=E_WW3_obp_H,E_WW3_obp_H2=E_WW3_obp_H2,E_WW3_noa_H2=E_WW3_noa_H2,\
                Hs_WW3_all=Hs_WW3_all,Hs_WW3_cut=Hs_WW3_cut,\
                Hs_WW3_filtered_mask=Hs_WW3_mask,HsWW3=Hs,Tm0m1WW3=Tm0m1,Tm02WW3=Tm02,U10WW3=U10,UdirWW3=Udir,QkkWW3=Qkk,Lm_WW3=Lm_WW3,dm_WW3=dm_WW3) 
   else: 
        np.savez(pth_results+'SWOT_swell_spectra_'+cycle+'_'+tracks+'_'+side+'_'+lonlat, \
                fileSWOT=filenopath,cycle=cycle,tracks=tracks,side=side,boxindices=boxindices,\
                lonc=lonc,latc=latc,timec=timec,trackangle=trackangle,\
                kx2=kx2,ky2=ky2,ssh_spec=Eta,sig0_spec=Etb,coh=coh,ang=ang,amask=amask,\
                sig0mean=sig0mean,sig0std=sig0std,E_SWOT=Eta,HH=HH,HH2=HH2, \
                Hs_SWOT_filtered_all=Hs_SWOT_all,Hs_SWOT_filtered_mask=Hs_SWOT,Hs_SWOT_mask=Hs_SWOT_mask,\
                Lm_SWOT_filtered_mask=Lm_SWOT,dm_SWOT_filtered_mask=dm_SWOT, \
                modelfound=modelfound)    

###################################################################
def  SWOT_create_L3_spectra(saving_name,modelOK,restab,nkxtab,nkytab,modf=0,moddf=0,modang=0):
     SL3_nc_fid = Dataset(saving_name, 'w',format='NETCDF3_CLASSIC')
     #SL3_nc_fid.createDimension('n_box', None)
     SL3_nc_fid.createDimension('time', None)
     SL3_nc_fid.createDimension('nind', 4)
     SL3_nc_fid.createDimension('nboy_40',1)
     SL3_nc_fid.createDimension('nbox_40',2)
     SL3_nc_fid.createDimension('nboy_20',2)
     SL3_nc_fid.createDimension('nbox_20',6)
     SL3_nc_fid.createDimension('nboy_10',4)
     SL3_nc_fid.createDimension('nbox_10',14)
     sres0=sres=f'{restab[0]:02d}'
     for indres in range(len(restab)): 
        ires=restab[indres]
        sres=f'{ires:02d}'
        SL3_nc_fid.createDimension('nfy_'+sres,nkytab[indres])
        SL3_nc_fid.createDimension('nfx_'+sres,nkxtab[indres])
     if modelOK > 0:
        nf=len(modf)
        ntheta=len(modang)
        SL3_nc_fid.createDimension('nf',nf)
        SL3_nc_fid.createDimension('nphi',ntheta)

     time = SL3_nc_fid.createVariable('time', np.float64, ('time'))
     #time.units = 'days since 1990-01-01'
     time.units = 'seconds since 2000-01-01 00:00:00.0'
     time.long_name = 'time'

     boxindices= SL3_nc_fid.createVariable('boxindices_40', np.float64, ('time','nboy_40','nbox_40','nind'))
     boxindices.setncatts({'comment': u"these are along-track indices j1, j2  followed by cross-track indices i1, i2"})

     
     
     SL3_nc_track = SL3_nc_fid.createVariable('track_angle', np.float32, ('time'))
     SL3_nc_track.setncatts({'long_name': u"track_angle", \
                    'units': u"degrees", \
                    'comment': u"clockwise from north, direction to"})
                    
     SL3_nc_var = SL3_nc_fid.createVariable('Q18_40', np.float32, ('time','nboy_40','nbox_40'))
     for indres in range(len(restab)): 
        ires=restab[indres]
        sres=f'{ires:02d}'

# Integrated parameters and statistics 
        SL3_nc_var = SL3_nc_fid.createVariable('longitude_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('latitude_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('sigma0_mean_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('sigma0_std_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))
               
        SL3_nc_var = SL3_nc_fid.createVariable('H18_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('L18_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('d18_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))

        SL3_nc_var = SL3_nc_fid.createVariable('quality_frac_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('quality_flag_mask_'+sres, np.int32, ('time','nboy_'+sres,'nbox_'+sres))

# Spectral coordinates 
      
        SL3_nc_var = SL3_nc_fid.createVariable('fx2D_'+sres, np.float32, ('nfy_'+sres, 'nfx_'+sres))
        SL3_nc_var.setncatts({'long_name': u"spatial_frequency", \
                    'units': u"cycles per meter", \
                    'comment': u"kx is cross-track to right"})

        SL3_nc_var = SL3_nc_fid.createVariable('fy2D_'+sres, np.float32, ('nfy_'+sres, 'nfx_'+sres))
        SL3_nc_var.setncatts({'long_name': u"spatial_frequency", \
                    'units': u"cycles per meter", \
                    'comment': u"ky is along-track"})

        SL3_nc_filt = SL3_nc_fid.createVariable('filter_OBP_'+sres, np.float32, ('nfy_'+sres, 'nfx_'+sres))
        SL3_nc_filt.setncatts({'long_name': u"instrument_filter_OBP", 'units': u"1"})

        SL3_nc_filt = SL3_nc_fid.createVariable('filter_PTR_'+sres, np.float32, ('nfy_'+sres, 'nfx_'+sres))
        SL3_nc_filt.setncatts({'long_name': u"instrument_filter_PTR", 'units': u"1"})


 
# For higher resolution: only spectra and integrated parameters 
        SL3_nc_varE4 = SL3_nc_fid.createVariable('E_SWOT_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres,'nfy_'+sres, 'nfx_'+sres), zlib=True)
        SL3_nc_varE4.setncatts({'long_name': u"PSD of surface elevation over 40 km side box", 'units': u"m**4"})                   
        SL3_nc_varE4 = SL3_nc_fid.createVariable('mask_'+sres, np.byte, ('time','nboy_40','nbox_40','nfy_'+sres, 'nfx_'+sres))
        SL3_nc_varE4.setncatts({'long_name': u"mask_for_wind_sea_and_swell"})

     SL3_nc_varE4 = SL3_nc_fid.createVariable('coh_SWOT_40', np.float32, ('time','nboy_40','nbox_40','nfy_40', 'nfx_40'))
     SL3_nc_varE4.setncatts({'long_name': u"coherence between SSH and sigma0 over 40 km side box",   'units': u"1"})                
     SL3_nc_varE4 = SL3_nc_fid.createVariable('ang_SWOT_40', np.float32, ('time','nboy_40','nbox_40','nfy_40', 'nfx_40'))
     SL3_nc_varE4.setncatts({'long_name': u"mean phase shift between SSH and sigma0 over 40 km side box",  'units': u"1"})     
     SL3_nc_varE4 = SL3_nc_fid.createVariable('crosr_SWOT_40', np.float32, ('time','nboy_40','nbox_40','nfy_40', 'nfx_40'))
     SL3_nc_varE4.setncatts({'long_name': u"real part of cross-spectrum between SSH and sigma0 over 40 km side box",  'units': u"rad"})                


     if modelOK > 0:
     
        SL3_nc_var = SL3_nc_fid.createVariable('frequency', np.float32, ('nf'))
        SL3_nc_var = SL3_nc_fid.createVariable('df', np.float32, ('nf'))
        SL3_nc_var = SL3_nc_fid.createVariable('direction', np.float32, ('nphi'))
        SL3_nc_var = SL3_nc_fid.createVariable('efth_model', np.float32, ('time','nboy_40','nbox_40','nf','nphi'), zlib=True)
        SL3_nc_var = SL3_nc_fid.createVariable('longitude_model', np.float32, ('time','nboy_40','nbox_40'))
        SL3_nc_var = SL3_nc_fid.createVariable('latitude_model', np.float32, ('time','nboy_40','nbox_40'))
        SL3_nc_var = SL3_nc_fid.createVariable('time_model', np.float32, ('time','nboy_40','nbox_40'))
        SL3_nc_var = SL3_nc_fid.createVariable('index_model', np.int32, ('time','nboy_40','nbox_40'))
        ires=restab[0]
        sres=f'{ires:02d}'
        SL3_nc_var = SL3_nc_fid.createVariable('Q18_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('H18_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('L18_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('d18_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('Hs_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('Tm02_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('lambdac_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('Qkk_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('U10_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
        SL3_nc_var = SL3_nc_fid.createVariable('Udir_model', np.float32, ('time','nboy_'+sres,'nbox_'+sres))
 
#   if modelfound==1:
     return SL3_nc_fid

###################################################################
def  SWOT_create_L3_CNES_Light(saving_name,modelOK,restab,nkxtab,nkytab,nind,ncross,modf=0,moddf=0,modang=0):
     SL3_nc_fid = Dataset(saving_name, 'w',format='NETCDF3_CLASSIC')
     SL3_nc_fid.createDimension('n_box', nind*ncross)
     sres0=sres='' #f'{restab[0]:02d}'
     for indres in range(len(restab)): 
        ires=restab[indres]
        sres='' # '_'+f'{ires:02d}'
        SL3_nc_fid.createDimension('nfy'+sres,nkytab[indres])
        SL3_nc_fid.createDimension('nfx'+sres,nkxtab[indres])
        SL3_nc_fid.createDimension('n_along'+sres, nind)
        SL3_nc_fid.createDimension('n_cross'+sres, ncross)
     if modelOK > 0:
        nf=len(modf)
        ntheta=len(modang)
        SL3_nc_fid.createDimension('nf',nf)
        SL3_nc_fid.createDimension('nphi',ntheta)
     SL3_nc_fid.createDimension('nind', 4)  # used for indices in SSH map
     
     
     SL3_nc_track = SL3_nc_fid.createVariable('track_angle', np.float32, ('n_box'))
     SL3_nc_track.setncatts({'long_name': u"track_angle", \
                    'units': u"degrees", \
                    'comment': u"clockwise from north, direction to"})

     time = SL3_nc_fid.createVariable('time', np.float64, ('n_box'))
     #time.units = 'days since 1990-01-01'
     time.units = 'seconds since 2000-01-01 00:00:00.0'
     time.long_name = 'time'

     box_indx= SL3_nc_fid.createVariable('box_indx',  'i2', ('n_box'))
     box_indx.setncatts({'comment': u"Index locating the box cross track in the swath, increasing towards the right (satellite frame) 0-1 for 40km boxes, 0-5 for 20km boxes, 0-13 for 10km boxes"})

     box_indy= SL3_nc_fid.createVariable('box_indy',  'i2', ('n_box'))
     box_indy.setncatts({'comment': u"index locating the box along track in the swath, increasing forward (satellite frame)"})

     
     ind_box= SL3_nc_fid.createVariable('ind_box',  'i4', ('n_along','n_cross'))

     boxindices= SL3_nc_fid.createVariable('boxindices', 'i4', ('n_box','nind'))
     boxindices.setncatts({'comment': u"these are along-track indices j1, j2  followed by cross-track indices i1, i2"})

                    
     for indres in range(len(restab)): 
        ires=restab[indres]
        sres='' #f'{ires:02d}'

# Integrated parameters and statistics 
        SL3_nc_var = SL3_nc_fid.createVariable('longitude'+sres, np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('latitude'+sres, np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('sigma0_mean'+sres, np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('sigma0_std'+sres, np.float32, ('n_box'))
               
        SL3_nc_var = SL3_nc_fid.createVariable('H18'+sres, np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('L18'+sres, np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('phi18'+sres, np.float32, ('n_box'))
        SL3_nc_var.setncatts({'long_name': u"mean wave propagation direction", \
                    'units': u"degrees", \
                    'comment': u"clockwise from north, direction to"})
        SL3_nc_var = SL3_nc_fid.createVariable('Q18', np.float32, ('n_box'))
                    
        SL3_nc_var = SL3_nc_fid.createVariable('quality_frac'+sres, np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('quality_flag_mask'+sres, np.int32, ('n_box'))

# Spectral coordinates 
      
        SL3_nc_var = SL3_nc_fid.createVariable('fx2D'+sres, np.float32, ('nfy'+sres, 'nfx'+sres))
        SL3_nc_var.setncatts({'long_name': u"spatial_frequency", \
                    'units': u"cycles per meter", \
                    'comment': u"fx is cross-track to right"})

        SL3_nc_var = SL3_nc_fid.createVariable('fy2D'+sres, np.float32, ('nfy'+sres, 'nfx'+sres))
        SL3_nc_var.setncatts({'long_name': u"spatial_frequency", \
                    'units': u"cycles per meter", \
                    'comment': u"fy is along-track"})

        SL3_nc_filt = SL3_nc_fid.createVariable('filter_OBP'+sres, np.float32, ('nfy'+sres, 'nfx'+sres))
        SL3_nc_filt.setncatts({'long_name': u"instrument_filter_OBP", 'units': u"1"})

        SL3_nc_filt = SL3_nc_fid.createVariable('filter_PTR'+sres, np.float32, ('nfy'+sres, 'nfx'+sres))
        SL3_nc_filt.setncatts({'long_name': u"instrument_filter_PTR", 'units': u"1"})


 
# For higher resolution: only spectra and integrated parameters 
        SL3_nc_varE4 = SL3_nc_fid.createVariable('Efxfy_SWOT'+sres, np.float32, ('n_box','nfy'+sres, 'nfx'+sres), zlib=True)
        SL3_nc_varE4.setncatts({'long_name': u"PSD of surface elevation over 40 km side box", 'units': u"m**4"})                   
        SL3_nc_varE4 = SL3_nc_fid.createVariable('mask'+sres, np.byte, ('n_box','nfy'+sres, 'nfx'+sres))
        SL3_nc_varE4.setncatts({'long_name': u"mask_for_wind_sea_and_swell"})

     SL3_nc_varE4 = SL3_nc_fid.createVariable('coh_SWOT', np.float32, ('n_box','nfy', 'nfx'))
     SL3_nc_varE4.setncatts({'long_name': u"coherence between SSH and sigma0",   'units': u"1"})                
     SL3_nc_varE4 = SL3_nc_fid.createVariable('ang_SWOT', np.float32, ('n_box','nfy', 'nfx'))
     SL3_nc_varE4.setncatts({'long_name': u"mean phase shift between SSH and sigma0",  'units': u"1"})     
     SL3_nc_varE4 = SL3_nc_fid.createVariable('crosr_SWOT', np.float32, ('n_box','nfy', 'nfx'))
     SL3_nc_varE4.setncatts({'long_name': u"real part of cross-spectrum between SSH and sigma0",  'units': u"rad"})                


     if modelOK > 0:
     
        SL3_nc_var = SL3_nc_fid.createVariable('frequency', np.float32, ('nf'))
        SL3_nc_var = SL3_nc_fid.createVariable('df', np.float32, ('nf'))
        SL3_nc_var = SL3_nc_fid.createVariable('direction', np.float32, ('nphi'))
        SL3_nc_var = SL3_nc_fid.createVariable('efth_model', np.float32, ('n_box','nf','nphi'), zlib=True)
        SL3_nc_var = SL3_nc_fid.createVariable('longitude_model', np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('latitude_model', np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('time_model', np.float32, ('n_box'))
        SL3_nc_var = SL3_nc_fid.createVariable('index_model', np.int32, ('n_box'))
        ires=restab[0]
        sres=f'{ires:02d}'
        SL3_nc_var = SL3_nc_fid.createVariable('Q18_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('L18_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('H18_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('phi18_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('Hs_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('Tm02_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('lambdac_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('Qkk_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('U10_model', np.float32, ('n_box')) 
        SL3_nc_var = SL3_nc_fid.createVariable('Udir_model', np.float32, ('n_box')) 
 
#   if modelfound==1:
     return SL3_nc_fid

###################################################################
def SWOT_write_L3_model_old(SL3_nc,step,efth,modf,moddf,modang,lonww3,latww3,timeww3,dww3,Hs_WW3_mask,LE_WW3,dm_WW3,Q18_WW3,Hs,Tm02,lambdac):
     writeOK=0
     SL3_nc.variables['efth_model'][step,indy,indx,:,:]=efth
     if step==0:
        SL3_nc.variables['frequency'][:]=modf
        SL3_nc.variables['df'][:]=moddf
        SL3_nc.variables['direction'][:]=modang
     SL3_nc.variables['longitude_model'][step,indy,indx]=lonww3
     SL3_nc.variables['latitude_model'][step,indy,indx]=latww3
     SL3_nc.variables['time_model'][step,indy,indx]=timeww3
     SL3_nc.variables['index_model'][step,indy,indx]=indww3
     SL3_nc.variables['H18_model'][step,indy,indx]=Hs_WW3_mask
     SL3_nc.variables['L18_model'][step,indy,indx]=LE_WW3
     SL3_nc.variables['d18_model'][step,indy,indx]=dm_WW3
     SL3_nc.variables['Q18_model'][step,indy,indx]=Q18_WW3
     SL3_nc.variables['Hs_model'][step,indy,indx]=Hs
     SL3_nc.variables['Tm02_model'][step,indy,indx]=Tm02
     SL3_nc.variables['lambdac_model'][step,indy,indx]=lambdac
     return writeOK

###################################################################
def SWOT_write_L3_model(SL3_nc,step,ibox,efth,modf,moddf,modang,lonww3,latww3,timeww3,indww3,Hs_WW3_mask,LE_WW3,dm_WW3,Q18_WW3,Hs,Tm02,lambdac):
     writeOK=0
     SL3_nc.variables['efth_model'][ibox,:,:]=efth
     if step==0:
        SL3_nc.variables['frequency'][:]=modf
        SL3_nc.variables['df'][:]=moddf
        SL3_nc.variables['direction'][:]=modang
     SL3_nc.variables['longitude_model'][ibox]=lonww3
     SL3_nc.variables['latitude_model'][ibox]=latww3
     SL3_nc.variables['time_model'][ibox]=timeww3
     SL3_nc.variables['index_model'][ibox]=indww3
     SL3_nc.variables['H18_model'][ibox]=Hs_WW3_mask
     SL3_nc.variables['L18_model'][ibox]=LE_WW3
     SL3_nc.variables['phi18_model'][ibox]=(dm_WW3 + 180) % 360  # following stupid convention "to"
     SL3_nc.variables['Q18_model'][ibox]=Q18_WW3
     SL3_nc.variables['Hs_model'][ibox]=Hs
     SL3_nc.variables['Tm02_model'][ibox]=Tm02
     SL3_nc.variables['lambdac_model'][ibox]=lambdac
     return writeOK

###################################################################
def  SWOT_write_L3_old(SL3_nc,step,indx,indy,indres,sres,kx2,ky2,timec,trackangle,boxindices,coh,ang,crosr, \
                       Q18_SWOT,lonc,latc,sig0mean,sig0std,fracfla,qual_mask,Hs_SWOT_mask,\
                       Lm_SWOT,dm_SWOT,amask,HH,HH1,Etacor):
     writeOK=0
# Write variables to NetCDF file
     if (step==0 & indx==0 & indy==0):
         print('sres:',sres) 
         SL3_nc.variables['kx2_'+sres][:,:]=kx2
         SL3_nc.variables['ky2_'+sres][:,:]=ky2
         
         if (indres==0):
             if (indside==0): 
               print('time units:',SL3_time.units) # numeric values
               print('timec:',timec) # numeric values
               epoch=np.datetime64('2000-01-01T00:00:00')
               timedt=(timec-epoch)/np.timedelta64(1,'s')
               #timedt=timec.astype(datetime)
               #times = timedt #date2num(timedt, SL3_time.units)
               SL3_nc.variables['time']=timedt
               SL3_nc.variables['track_angle'][step]=trackangle      
         SL3_nc.variables['boxindices_'+sres][step,indy,indx,:]=boxindices
         SL3_nc.variables['coh_SWOT_'+sres][step,indy,indx,:,:]=coh
         SL3_nc.variables['ang_SWOT_'+sres][step,indy,indx,:,:]=np.degrees(ang)
         SL3_nc.variables['crosr_SWOT_'+sres][step,indy,indx,:,:]=crosr
         SL3_nc.variables['Q18_'+sres][step,indy,indx]=Q18_SWOT
         
         SL3_nc.variables['longitude_'+sres][step,indy,indx] = lonc 
         SL3_nc.variables['latitude_'+sres][step,indy,indx] = latc
         SL3_nc.variables['sigma0_mean_'+sres][step,indy,indx]=sig0mean      
         SL3_nc.variables['sigma0_std_'+sres][step,indy,indx]=sig0std      
         SL3_nc.variables['quality_frac_'+sres][step,indy,indx]=fracfla
         SL3_nc.variables['quality_flag_mask_'+sres][step,indy,indx]=qual_mask
         SL3_nc.variables['H18_'+sres][step,indy,indx]=Hs_SWOT_mask
         SL3_nc.variables['L18_'+sres][step,indy,indx]=Lm_SWOT
         SL3_nc.variables['d18_'+sres][step,indy,indx]=dm_SWOT
         SL3_nc.variables['mask_'+sres][step,0,indside,:,:] = amask
         SL3_nc.variables['filter_OBP_'+sres][:,:] = HH
         SL3_nc.variables['filter_PTR_'+sres][:,:] = HH1
        
         SL3_nc.variables['E_SWOT_'+sres]=Etacor
         ibox=ibox+1   
     return writeOK


###################################################################
def  SWOT_write_L3_CNES_Light(SL3_nc,ibox,step,indside,indx,indy,indres,sres,kx2,ky2,timec,trackangle, \
                       boxindices,coh,ang,crosr, \
                       Q18_SWOT,lonc,latc,sig0mean,sig0std,fracfla,qual_mask,Hs_SWOT_mask,\
                       Lm_SWOT,dm_SWOT,amask,HH,HH1,Etacor):
     writeOK=0
# Write variables to NetCDF file
     if (ibox==0):
         print('sres:',sres) 
         SL3_nc.variables['fx2D'+sres][:,:]=kx2
         SL3_nc.variables['fy2D'+sres][:,:]=ky2
         SL3_nc.variables['filter_OBP'+sres][:,:] = HH
         SL3_nc.variables['filter_PTR'+sres][:,:] = HH1
         
     SL3_nc.variables['box_indx']=indside
     SL3_nc.variables['box_indy']=step
     SL3_nc.variables['ind_box'][step,indside]=ibox 
     
     epoch=np.datetime64('2000-01-01T00:00:00')
     timedt=(timec-epoch)/np.timedelta64(1,'s')
     print('timec:',timec,timedt) # numeric values
         
     SL3_nc.variables['time'][ibox]=timedt
     SL3_nc.variables['track_angle'][ibox]=(trackangle + 180) % 360  # following stupid convention "to"    
     SL3_nc.variables['boxindices'+sres][ibox,:]=boxindices
     SL3_nc.variables['coh_SWOT'+sres][ibox,:,:]=coh
     SL3_nc.variables['ang_SWOT'+sres][ibox,:,:]=np.degrees(ang)
     SL3_nc.variables['crosr_SWOT'+sres][ibox,:,:]=crosr
     SL3_nc.variables['Q18'+sres][ibox]=Q18_SWOT
         
     SL3_nc.variables['longitude'+sres][ibox] = lonc 
     SL3_nc.variables['latitude'+sres][ibox] = latc
     SL3_nc.variables['sigma0_mean'+sres][ibox]=sig0mean      
     SL3_nc.variables['sigma0_std'+sres][ibox]=sig0std      
     SL3_nc.variables['quality_frac'+sres][ibox]=fracfla
     SL3_nc.variables['quality_flag_mask'+sres][ibox]=qual_mask
     SL3_nc.variables['H18'+sres][ibox]=Hs_SWOT_mask
     SL3_nc.variables['L18'+sres][ibox]=Lm_SWOT
     SL3_nc.variables['phi18'+sres][ibox]=(dm_SWOT + 180) % 360  # following stupid convention "to"
     SL3_nc.variables['mask'+sres][ibox,:,:] = amask
      
     SL3_nc.variables['Efxfy_SWOT'+sres][ibox,:,:]=Etacor
            
     return writeOK
     
     
     
###################################################################
def  SWOT_spectra_for_one_L3track(cycle,tracks,mask_choice,number_res,spectra_res,vtag,pth_swot,\
                                pth_WW3_trck,pth_spectra,pth_plots,modelOK=0,modeltag='',latmax=91,latmin=-91, \
                                fs1=20,dBE=25,dBE2=25,addglobcur=0,doplot=0)  :

  cohthr=0.3
  flagssha=1E10;
  ntrack=int(tracks)
  if (np.mod(ntrack,2)==1):
      td='ascending'
  else: 
      td='descending'
  ncout=''
  # Searches for L3 file ... 
  file_list = glob.glob(pth_swot+'SWOT_L3_LR_SSH_*Unsmoothed_'+cycle+'_'+tracks+'*.nc')
  if (len(file_list) > 0) : 
    file_swot=file_list[0]
    tags=file_swot.split(sep='/')
    filenopath=tags[-1]
    ncout=pth_spectra+'SWOT_L3_LR_WIND_WAVE_'+filenopath[26:65]+'_v'+vtag+'.nc' 
    
    if os.path.exists(ncout):
       print('output file already exists:',ncout)
       
  if (len(file_list) > 0) and (not  os.path.exists(ncout)) : 
    days=filenopath[34:len(filenopath)]
    print('Reading file:',file_swot,'##',days)
    ddla = xr.open_dataset(file_swot)
    #print(ddla)

    # This opens the WAVEWATCH III spectra file (computed for B. Molero).
    filetr=pth_WW3_trck+'SWOT_WW3-GLOB-30M_'+days[0:6]+'_trck.nc'
    print('file for model:',filetr) 
    ds_ww3t = xr.open_dataset(filetr)
    modang=np.mod(90-ds_ww3t.direction,360)
    moddf=ds_ww3t.frequency2.values-ds_ww3t.frequency1.values
    modf=ds_ww3t.frequency.values
    modnth=np.shape(modang)[0]
    moddth=(2*np.pi/modnth)

    # Get globcurrent 
    if addglobcur==1:
        # defines bounding box
        xt=ddla.longitude[:,20].values
        yt=ddla.latitude[:,20].values
        ind=np.where((yt > latmin-marginlat) & (yt < latmax+5))[0]
        xt=xt[ind[0]:ind[-1]:10];yt=yt[ind[0]:ind[-1]:10]
 
        ds = xr.open_dataset('../dataset-uv-nrt-hourly_20230609T0000Z_P20230726T0000.nc')
        area=[np.floor( np.nanmin([xt]) )-cmapmin,np.floor(latmin)-marginlat,  np.ceil(np.nanmax([xt]) )+cmapmax,   np.ceil(latmax)+5]
        print('YT:',area) 
        print('globlon:',ds.longitude[0].values,ds.longitude[-1].values)
        print('area:', area)

        selection = (
        (ds.longitude+360*lonshift > area[0]) &
        (ds.longitude+360*lonshift < area[2]) &
        (ds.latitude > area[1]) &
        (ds.latitude < area[3]))

        ds_glob = ds.where(selection, drop=True)
        globlon=ds_glob.longitude+360*lonshift
        globlat=ds_glob.latitude
        V=ds_glob.vo[17,0,:,:].squeeze()



    l1=latmin;l2=latmax
    if td == 'descending':
       l1=latmax;l2=latmin

    # initialize resolution and other geometry information 
    dx,dy,indxc,ISHIFT,nkxr,nkyr,restab,nX2tab,nY2tab,mtab,ntab,indl,dind,samemask,hemiNS,hemiWE=spec_settings_for_L3(number_res,spectra_res);



    step=-1;stepp=0;HsvalueOK=0;


    if (np.abs(l1) < 90): 
        ddl,indsub0,indsub1=swot.subset(ddla,[-0.5+float(l1), 0.5+float(l1)])
        ind00=(indsub0//dind)*dind
    else:
        ind00=0
    if (np.abs(l2) < 90): 
        ddl,indsub0,indsub1=swot.subset(ddla,[-0.5+float(l2), 0.5+float(l2)])
        ind99=(indsub1//dind)*dind
    else:
        nlines=ddla.dims['num_lines']
        print('DIMS:',nlines)
        ind99=nlines-indl
    
    shifty=ind00
    indsubs=np.arange(ind00,ind99,dind)
    nind=len(indsubs)
    print('creating NetCDF file:',ncout)
    SL3_nc=SWOT_create_L3_CNES_Light(ncout,modelOK,restab,nX2tab*2//mtab,nY2tab*2//ntab,nind,2,modf=modf,moddf=moddf,modang=modang)
    ibox=0
    # Start of main loop over along track positions ... 
    for indsub0 in indsubs:
       step=step+1
       subset_vars = {}
       for varname, var in ddla.data_vars.items():
           if var.dims==2:
             subset_vars[varname] = var[indsub0:indsub0+indl,:]
           else:
             subset_vars[varname] = var[indsub0:indsub0+indl]
             # Combine the subset variables into a new dataset

       ddl = xr.Dataset(subset_vars, attrs=ddla.attrs)

# gets data from SWOT L3 SSH file 
       ssha = ddl.ssha_unedited
       flag = ddl.quality_flag  # L2 : .ssh_karin_2_qual
       ssha = np.where(flag < flagssha, ssha, np.nan)
       sig0 = ddl.sigma0 #sig0_karin_2
       flas = ddl.quality_flag # sig0_karin_2_qual
       lon = ddl.longitude.values
       lat = ddl.latitude.values
       [nline,npix]=np.shape(ssha)

       # there may be a better way to get this ... but here is an angle estimates from the position of near nadir pixeks 
       dlon=lon[nline-10,indxc+ISHIFT]-lon[10,indxc+ISHIFT]
       dlat=lat[nline-10,indxc+ISHIFT]-lat[10,indxc+ISHIFT]
       midlat=0.5*(lat[nline-10,indxc+ISHIFT]+lat[10,indxc+ISHIFT])
       trackangle=-90-np.arctan2(dlat,dlon*np.cos(midlat*np.pi/180))*180/np.pi


       X=(np.arange(npix)-indxc)*dx/1000
       Y=(np.arange(nline)+indsub0-shifty)*dy/1000 # warning the along-track resolutionis not exactly 250 m, more like 235 m 
   
       for indside in range(2):    # separate the loops over left and right parts
         modelfound=0
         for indres in range(len(restab)):   # loop over different spatial resoltions 
          ires=restab[indres]
          sres='' # f'{ires:02d}'
          nX2=nX2tab[indres]  # half size of box for which spectrum is computed 
          nY2=nY2tab[indres]
          m=mtab[indres];n=ntab[indres]
          cfac=np.sqrt(n*m)
          # Defines number of spectral boxes for current resolution 
          nindx=nX2tab[0]//nX2tab[indres]+(nX2tab[0]//nX2tab[indres]-1)  # Welch-type 50 % overlap in x direction ... 
          nindy=nY2tab[0]//nY2tab[indres]
          if ires==40:
              # array of indices for left edge of each analysis window
             i1array=np.array([indxc-ISHIFT-nX2*2,ISHIFT+indxc])
          if ires==20:
             # Defines area for spectral analysis 
             i1array=np.array([indxc-ISHIFT-nX2*4,indxc-ISHIFT-nX2*3,indxc-ISHIFT-nX2*2, \
                           ISHIFT+indxc,ISHIFT+indxc+nX2,ISHIFT+indxc+nX2*2])
          if ires==10:
             # Defines area for spectral analysis 
             i1array=np.array([indxc-ISHIFT-nX2*8,indxc-ISHIFT-nX2*7,indxc-ISHIFT-nX2*6, \
                               indxc-ISHIFT-nX2*5,indxc-ISHIFT-nX2*4,indxc-ISHIFT-nX2*3, \
                               indxc-ISHIFT-nX2*2, \
                               ISHIFT+indxc,ISHIFT+indxc+nX2,ISHIFT+indxc+nX2*2, \
                               ISHIFT+indxc+nX2*3,ISHIFT+indxc+nX2*4,ISHIFT+indxc+nX2*5, \
                               ISHIFT+indxc+nX2*6])
      
          nxtile=nX2*2//m  # cross-track
          nytile=nY2*2//n  # along-track

# Loops across track and along-track (within given side) 
          for iidx in range(nindx):    
           indx=iidx+indside*nindx
           i1=i1array[indx]
           i2=i1+nX2*2
           for indy in range(nindy):
    # cross-track indices
    #alongtrack indices
          # needs extra loop here ... 
             j1=0 #nline//2-nY2*(nindy-indy) #10   # centers box on target latitude
             j2=j1+nY2*2
             latc=ddl.latitude[j1+nY2,i1+nX2].values    # WILL HAVE TO CHANGE THIS ... 
             latcr=np.round(latc*2)/2; latcs=f'{abs(latc):3.2f}'+hemiNS[int(np.sign(latc))]
             lonc=lon[j1+nY2,i1+nX2]; 
             loncr=np.round(lonc*2)/2; loncs=f'{abs(lonc):3.2f}'+hemiWE[int(np.sign(lonc))]
             lat_bounds=[-0.5+float(latc), 0.5+float(latc)];
             lonlat=latcs+loncs
             steps=f'{step:05d}'  
             sside='left'
             if indside ==1:
                 sside='right'
             filetag='SWOT_'+cycle+'_'+tracks+'_'+sside+'_'+steps+'_'+lonlat  
       
             Xmem=X;
             Ymem=Y;
# NB: with L3 data we do not need the cross-track flip, thus side is forced to "right"
# Later version will remove the flip. 
# NB: L3 sigma0 uses LINEAR units ... not dB !!
             mybox,mybos,flbox,X,Y,sflip,signMTF,Look=SWOTarray_flip_north_up(dlat, \
                                                         'right',ssha[j1:j2,i1:i2],flas[j1:j2,i1:i2],sig0[j1:j2,i1:i2],Xmem,Ymem)

             if (indres==0 & indside==0):  
                timec=ddl.time.values[j1+nY2]
                #print('track vector:',indsub0+j1,dlat,dlon*np.cos(midlat*np.pi/180),trackangle,trackangle+180,'##',dlat,Look,indsub0)
           
             nanarr = np.where(np.isnan(mybox), 1.0,0)  # Need to count these in for  flags ... 
             infarr = np.where(np.isinf(mybox), 1.0,0)  # Need to count these in for  flags ... 
             flaarr = np.where(flbox > 1, 1.0,0)  # Need to count these in for  flags ... 
             fracbad=np.sum((nanarr+infarr).flatten())/((j2-j1)*(i2-i1))
             fracfla=np.sum(flaarr.flatten())/((j2-j1)*(i2-i1))
           
             #print('BAD:',fracbad,fracfla)  
    # Computes spectrum from SWOT SSH data
    # Note: this uses tiles: we may use these higher resolution estimates to avoid duplication 
             (Eta,Etb,ang,angstd,coh,crosr,phases,ky2,kx2,dky,dkx,detrenda,detrendb,nspec)=FFT2D_two_arrays_nm_detrend_flag(mybox,mybos,flbox, \
                                                                                                     dy,dx,n,m,detrend='quadratic') 

         
             if (iidx==0 & indy==0 & ((1-samemask)*step)==0):   
                kxmax=-2*kx2[0,0]
                kymax=-2*ky2[0,0]
                nkxr=nxtile      # twice the SWOT range to allow aliasing computation 
                nkyr=nytile
                dkxr=kxmax/nkxr
                dkyr=kymax/(nkyr-1)  # only true in nkyr is odd ?? 

                kxr=np.linspace(-nkxr*dkxr,(nkxr-1)*dkxr,nkxr*2)
                kyr=np.linspace(-nkyr*dkyr,(nkyr-1)*dkyr,nkyr*2)
                fx_wreg=kxr*1000
                fy_wreg=kyr*1000
                kxr2, kyr2 = np.meshgrid(kxr,kyr,indexing='ij') 
                kn=np.sqrt(kx2**2+ky2**2)*1000
 
                ik1=(nxtile+1)//2;ik2=ik1+nxtile
                jk1=(nytile+1)//2;jk2=jk1+nytile
   
# Defines the spectral response H associated with SWOT on board processing and PTR 
                x_xt, w_xt, f_xt, H_xt = get_obp_filter(L_filt = 0.980, f_axis = fx_wreg, plot_flag = False, kernel="parzen")
                x_at, w_at, f_at, H_at = get_obp_filter(L_filt = 1, f_axis = fy_wreg, plot_flag = False, kernel="bharris")
                x_at, w_at, f_obp, H_ptr = get_obp_filter(L_filt = 3, sampling_in = 0.0125,f_axis = fy_wreg, plot_flag = False, kernel="alejandro_azptr")
    
                H = np.repeat(np.array([H_xt]), len(H_at), axis=0).T * np.repeat(np.array([H_at]), len(H_xt), axis=0)
                Hptr = np.repeat(np.array([H_ptr]), len(H_xt), axis=0)
                H2=H*Hptr                     # note that when model data is also used, H3 is defined below to include az cut-off 
                H3=H2

# these filters now have smae dimension as the spectra 
                HH =H[ik1:ik2,jk1:jk2].T
                HH1=Hptr[ik1:ik2,jk1:jk2].T
                HH2=H2[ik1:ik2,jk1:jk2].T
                HH3=H3[ik1:ik2,jk1:jk2].T

         
             if (modelOK > 0 ):
               if (indres == 0):
# Looks for matching wavemodel spectrum 
                  modspec,indww3,modelfound,timeww3,lonww3,latww3,distww3,U10,Udir,dpt=SWOTfind_model_spectrum(ds_ww3t,lonc,latc,timec)
# Computes kx,ky spectrum from WW3 on fine grid: using even number of k's makes the spectrum non-symmetric 
# warning: this is repeated if indres > 0 because spectral resolution may differ 
               if (modelfound>0 & iidx==0 & indy==0):
                   efth=modspec.values;
                   [Ef,th1m,sth1m,Hs,Tm0m1,Tm02,Qf,Qkk] = wavespec_Efth_to_first3(efth,modf,moddf, modang.values,moddth) 
                   sigu=(Hs/4)*2*np.pi/Tm02
                   lambdac=1/((7310/875.0e3)/sigu/(np.pi))   # az cut-off wavenumber in cpm
                   Hazc = np.exp(-(kyr2*lambdac)**2)        # this is the effect of velocity bunching for sigma0 ... what about the phase? 
                   H3=H*Hptr*Hazc
                   HH3=H3[ik1:ik2,jk1:jk2].T

# converts the model spectrum to the SWOT (kx,ky) geometry
                   Eta_WW3_obp_H2,Eta_WW3_obp_H,Eta_WW3_noa_H2,Eta_WW3_res,Eta_WW3_c,Ekxky,kxm,kym,ix1,iy1= \
                              wavespec_Efth_to_kxky_SWOT(efth,modf,moddf, modang,moddth,f_xt,f_at,H,Hazc,H3, \
                                            kxmax,kymax,dkx,dky,dkxr,dkyr,nxtile,nytile,doublesided=0,verbose=0,trackangle=(trackangle+sflip*180)*np.pi/180)
        


# Defines swell mask  : uses function SWOTdefine_swell_mask
             if (iidx==0 & indy==0 & (samemask*indres)==0):
                mask_adapt=mask_choice
                qual_mask=0
                if (modelfound==0):
                   mask_adapt=0
                   qual_mask=1
                spec_for_mask=Eta  #/HH3  # if model is not found: uses SWOT for mask ... 
                if mask_adapt == -3 :
                   spec_for_mask=Eta_WW3_noa_H2/HH2
                if mask_adapt == -4 :
                   spec_for_mask=Eta_WW3_noa_H2
                if mask_adapt == -5 :
                   spec_for_mask=Eta_WW3_res
                amask,bmask=SWOTdefine_swell_mask_simple(spec_for_mask,coh,ang,np.nanmedian(mybos),dlat,kx2,ky2,cohthr,cfac,mask_adapt)

                vertices=SWOTspec_mask_polygon(amask) 
                
# Computes model parameters ... 
             if (modelfound > 0):
               if( indres == 0): 
                Eta_WW3_obp_H2_mask=np.where( bmask > 0.5, Eta_WW3_obp_H2,0) 
                var2=np.sum(Eta_WW3_obp_H2.flatten())*dkxr*dkyr; 
                var3=np.sum(Eta_WW3_c.flatten())*dkxr*dkyr;
                Hs_WW3_all=4*np.sqrt(var2)
                Hs_WW3_cut=4*np.sqrt(var3)
                # NB: I was using bmask for this computation before Sept. 2024
                Hs_WW3_mask,Lm_WW3,LE_WW3,Lmnew,dm_WW3,Q18_WW3,spr=SWOTspec_to_HsLm(Eta_WW3_obp_H2,kx2,ky2,amask,HH3,trackangle)
                write_OK=SWOT_write_L3_model(SL3_nc,step,ibox,efth,modf,moddf,modang,lonww3,latww3,timeww3,\
                                               indww3,Hs_WW3_mask,LE_WW3,dm_WW3,Q18_WW3,Hs,Tm02,lambdac)
            
####################################################################################
                if ((step < doplot) & (indres==0)):
                   fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(6,3.5))
                   spec = mpl.gridspec.GridSpec(ncols=2, nrows=1,width_ratios=[6, 5])
                   plt.subplots_adjust(left=0.05,bottom=0.1, top=0.92,wspace=0.1,right=0.99)
    
# Plotting WW3 spectrum, same resolution as SWOT spectrum 
                   im=ax[0].pcolormesh(kx2*1000,ky2*1000,10*np.log10(Eta_WW3_obp_H2),cmap='viridis',rasterized=True,vmin = -10+dBE, vmax=30+dBE)
                   _=ax[0].set_title('model spectrum  (dB)')
                   draw_mask(ax[0],kx2,dkx,ky2,dky,vertices,color='w',lw=3) 
                
# Plotting WW3 spectrum, same resolution as SWOT spectrum but 2 x spectral range and no ambiguity in direction 
                   im=ax[1].pcolormesh(-kxm[ix1:ix1+nxtile*6]*1000,-kym[iy1:iy1+nxtile*6]*1000,10*np.log10(Ekxky[ix1:ix1+nxtile*6,iy1:iy1+nxtile*6]).T, \
                                       cmap='viridis',rasterized=True,vmin=-10+dBE2, vmax=30+dBE2)
                   _=ax[1].set_title('model unfiltered')
                   plt.setp(ax[1].get_yticklabels(), visible=False)
                   plt.show()
                   fig.savefig(pth_plots+filetag+'WW3.png',dpi=100)    

########################### Plots SWOT spectrum E_S(kx,ky)
             if (step < doplot):
                fig,axs= plot_spec(kx2,dkx,ky2,dky,Eta,dBE,vertices)
                if addglobcur==1:
                   plot_cur(axs,td,xt,yt,latc,globlon,globlat,U,V,lightcmap)
                   axs[1].set_xlim([area[0],area[2]])
                   axs[1].set_ylim([area[1],area[3]])

                fig.savefig(pth_plots+filetag+'spec.png',dpi=100) 
###############
                fig,axs=plot_coh(kx2,dkx,ky2,dky,coh,np.degrees(ang),vertices)
                fig.savefig(pth_plots+filetag+'coh.png',dpi=100)    
        
#########  Stats and integrated parameters
             Eta_SWOT_mask=np.where( amask > 0.5, Eta,0)    
    
             varm=np.sum(Eta.flatten())*dkx*dky;
             var0=np.sum(Eta_SWOT_mask.flatten())*dkx*dky*2;
             Hs_SWOT_all=4*np.sqrt(varm)
             Hs_SWOT=4*np.sqrt(var0)
             Hs_SWOT_mask,Lm_SWOT,LE_SWOT,Lmnew,dm_SWOT,spr,Q18_SWOT=SWOTspec_to_HsLm(Eta,kx2,ky2,amask,HH2,trackangle)
             HsvalueOK=1

             Xbox=X[i1:i2];Ybox=Y[j1:j2]
        
             boxindices=[indsub0+j1,indsub0+j2,i1,i2]
             sig0mean=np.nanmedian(mybos)
             sig0std=np.nanstd(mybos)

             Etacor=Eta/HH3
             writeOK=SWOT_write_L3_CNES_Light(SL3_nc,ibox,step,indside,indx,indy,indres,sres,kx2,ky2,timec,trackangle,boxindices,coh,ang,crosr, \
                       Q18_SWOT,lonc,latc,sig0mean,sig0std,fracfla,qual_mask,Hs_SWOT_mask,\
                       LE_SWOT,dm_SWOT,amask,HH,HH1,Etacor)

             print('Writing to file for latitude ',latc,' in range [',latmin,latmax,'] , indices:', \
                   step,indside,ires,'##',indy,iidx,', size:',np.shape(Eta))
             ibox=ibox+1
        
# This processing is just for display purposes: larger piece of SWOT data used in animations 
         if ( (stepp < doplot) & (indside == 0)):
            if (modelfound>0 ):
                if indside==0:
                   I1=indxc-ISHIFT//2-200;I2=I1+200;J1=0;J2=420;
                if indside==1:
                   I1=indxc+ISHIFT//2;I2=I1+200;J1=0;J2=420;
                SSHA,SIG0,FLAS,X,Y,sflip,signMTF,Look=SWOTarray_flip_north_up(dlat,'right',ssha[J1:J2,I1:I2], \
                                                                                    flas[J1:J2,I1:I2],sig0[J1:J2,I1:I2],Xmem,Ymem)

                (Eta,Etb,ang,angstd,coh,crosr,phases,ky2,kx2,dky,dkx,detrenda,detrendb,nspec)=FFT2D_two_arrays_nm_detrend_flag(SSHA,10**(0.1*SIG0),FLAS,dy,dx,10,5,detrend='quadratic') 
                sig0max=np.nanmax(sig0)
                sig0min=np.nanmin(sig0)
                sig0mean=np.nanmedian(sig0)
                sig0std=np.nanstd(sig0)
                YP=Y #-Y[J1]
                fig,axs=plt.subplots(1,2,figsize=(12,10))#,sharey=True,sharex=True)
                plt.subplots_adjust(left=0.1,bottom=0.07, top=0.96,wspace=0.05,right=0.99)
 
                if Look==-1:
                   axs=np.roll(axs,1)
                   plt.setp(axs[0].get_yticklabels(), visible=False)
                   _=axs[1].set_ylabel('along-track (km)',fontsize=fs1)
                else:
                   plt.setp(axs[1].get_yticklabels(), visible=False)
                   _=axs[0].set_ylabel('along-track (km)',fontsize=fs1)
  
                im=axs[0].pcolormesh(X[I1:I2],YP[J1:J2],detrenda,rasterized=True, cmap=lightcmap,vmin=-0.1,vmax=0.1)
  
                arx0=X[(i1+i2)//2]-7.5*(1-indside);ary0=YP[40];arxd=5;aryd=5;arwid=1;gr=[0.,1,0.]
                if addarrows==1:
                   axs[0].arrow(arx0, ary0, arxd*np.sign(dlat)*np.sin(trackangle*np.pi/180), -np.sign(dlat)*aryd*np.cos(trackangle*np.pi/180), linewidth=4,color='k',head_width=arwid) 
                   axs[0].text(arx0+arxd*1.4*np.sign(dlat)*np.sin(trackangle*np.pi/180),ary0-np.sign(dlat)*aryd*1.4*np.cos(trackangle*np.pi/180),'N',fontsize=fs1)
                   axs[0].arrow(arx0, ary0, 0., 5*np.sign(dlat), linewidth=4,color=gr,head_width=arwid) 
                   axs[0].text(arx0+0.4*arxd,ary0+aryd*1.2*np.sign(dlat),'Vsat',fontsize=fs1,color=gr)
                   axs[0].arrow(arx0, ary0 ,  arxd*Look, 0, linewidth=4,color=gr,head_width=arwid) 
                   axs[0].text(arx0+arxd*(1.5*Look-0.5),ary0-2,'Look',fontsize=fs1,color=gr)
        
                   _=axs[0].set_xlabel('cross-track (km)', fontsize=fs1)
                   _=axs[0].set_title('sea level, track '+tracks+' '+cycle+', '+latcs+' '+loncs)
                   axs[0].set_xlim((X[I1],X[I2]))
                   axs[0].set_ylim((YP[J1],YP[J2]))
  # Now changes the labels to the distance alongtrack 
                   Yticks=axs[0].get_yticks()
                   newlabs=[ f"{int(np.abs(value)):04d}" for value in Yticks ]
                   axs[0].set_yticklabels(newlabs)
    
      
                   im=axs[1].pcolormesh(X[I1:I2],YP[J1:J2],SIG0,cmap='Greys_r',rasterized=True,vmax=sig0mean+2.5*sig0std,vmin=sig0mean-2.5*sig0std)
                   axs[1].set_xlim((X[I1],X[I2]))
                   axs[1].set_ylim((YP[J1],YP[J2]))
                   _=axs[1].set_xlabel('cross-track (km)', fontsize=fs1)
                   _=axs[1].set_title(r' $\sigma_0$, median='+f'{abs(sig0mean):4.1f}'+'dB, std='+f'{abs(sig0std):4.1f}'+'dB')
                   axs[1].set_yticklabels(np.abs(Yticks).astype(str))

                if (HsvalueOK ==1):
                   axs[1].add_patch(Rectangle((arx0-4.5*arxd,ary0-2.0*aryd),arxd*10,aryd*3.4,facecolor="white",alpha=0.5) )
                   resultat1=r'total $H_{s}$ model='+f'{Hs:4.2f}'+' m, $Q_{kk}$ model ='+f'{Qkk:4.0f}'+' m'
                   axs[1].text(arx0-4*arxd,ary0+0.8*aryd, resultat1,fontsize=16,color='k')
                   resultat1=r'$H_{18}$ SWOT='+f'{Hs_SWOT_mask:4.2f}'+' m, $H_{18}$ model = '+f'{Hs_WW3_mask:4.2f}'+' m'
                   axs[1].text(arx0-4*arxd,ary0, resultat1,fontsize=16,color='k')
                   resultat2=r'$L_{18}$ SWOT='+f'{Lm_SWOT:4.0f}'+' m, $L_{18}$ model = '+f'{Lm_WW3:4.0f}'+' m'
                   axs[1].text(arx0-4*arxd,ary0-0.8*aryd, resultat2,fontsize=16,color='k')
                   resultat3=r'$\theta_{18}$ SWOT='+f'{dm_SWOT:3.0f}'+r' deg, $\theta_{18}$ model= '+f'{dm_WW3:3.0f}'+' deg'
                   axs[1].text(arx0-4*arxd,ary0-1.6*aryd, resultat3,fontsize=16,color='k')
    #  print('Hs from SWOT :',Hs_SWOT, Hs_SWOT_all, Hs_SWOT_mask,' and WW3:',Hs_WW3,Hs_WW3_all,Hs_WW3_mask,'##',Hs_WW3_cut )
    #  print('Lm,dm from SWOT:',Lm_SWOT,dm_SWOT,' and WW3:',Lm_WW3,dm_WW3,shiftdir,ncoh )

    
                if savefile=='pdf' :    
                   fig.savefig(pth_plots+filetag+'map.pdf') #',dpi=100)
                else :
                   fig.savefig(pth_plots+filetag+'map.png',dpi=100)
                stepp=stepp+1
    SL3_nc.close()  # close the L3 spectra file
  else: 
    file_swot=''
  return file_swot


###################################################################
def  SWOT_denoise_isotropic(Ekxky,kx2,ky2,ndir=0,verbose=0)  :
    '''
    Converts E(f,theta) spectrum from buoy or model to E(kx,ky) spectrum similar to image spectrum
    2023/11/14: preliminary version, assumes dfreq is symmetric (not eaxctly true with WW3 output and waverider data) 
    inputs :
            - Ekxky : spectrum
            - kx2,ky2: 2D arrays of wavenumbers in cycles / m  
    output : 
            - Ekxky_nonoise: denoised spectrum
            - 
    '''
    dkx=kx2[0,1]-kx2[0,0]
    dky=ky2[1,0]-ky2[0,0]
    kmax=abs(kx2[0,0])
    if (ndir==0):
        ndir=2*(1+np.ceil(2*np.pi*(kmax/dkx))//2).astype(int)
    
    #print('Number of directions:',kmax,dkx,kmax/dkx,ndir) 
    theta1=np.arange(0,ndir,1)*2*np.pi/ndir
    #print('dirs:',theta1*180/np.pi) 
    kn=np.sqrt(kx2**2+ky2**2)
    theta=np.arctan2(ky2,kx2)
    theta=np.where(theta < 0,theta+2*np.pi,theta)
    dk=dkx
    kn1=np.arange(0,kmax*2,dk)
    nk=len(kn1)
    theta2,kn2 = np.meshgrid(theta1,kn1,indexing='ij')   #should we transpose kx2 and ky2 ???

    fig1=0
    if (fig1==1):
        fig,axs=plt.subplots(1,2,figsize=(10.8,8))
        fs1=16
        im=axs[0].pcolormesh(kx2*1000,ky2*1000,kn,rasterized=True)
        _=plt.colorbar(im,ax=axs[0],label='k (cpk)', location='bottom',shrink=0.8)
        _=axs[0].set_xlabel('$k_x$ (cycles / km)', fontsize=fs1)
        _=axs[0].set_ylabel('$k_y$ (cycles / km)', fontsize=fs1)
        im=axs[1].pcolormesh(kx2*1000,ky2*1000,theta*180/np.pi,rasterized=True)
        _=plt.colorbar(im,ax=axs[1],label='theta (deg)', location='bottom',shrink=0.8)
        #im=axs[1].pcolormesh(theta1,kn1,theta2*180/np.pi,rasterized=True)
        #_=plt.colorbar(im,ax=axs[1],label='theta (deg)', location='bottom',shrink=0.8)
  
    Jac=1 # kn
    Ekxkymin=Ekxky
    Ekth=np.zeros((ndir,nk))+np.max(Ekxky)
    #Ekth = griddata((theta.flatten(),kn.flatten()), (Ekxky*Jac).flatten(), (theta2,kn2), method='nearest')
    [ny,nx]=np.shape(Ekxky) 
    for ix in range(nx):
        for iy in range(ny):
            ik=np.around(kn[iy,ix]/dk).astype(int)
            it=np.mod(np.around(theta[iy,ix]/dk),ndir).astype(int)
            #print(ix,iy,ik,it,kn[iy,ix]/dk)
            Ekth[it,ik]=Ekxky[iy,ix]
    Emin=np.min(Ekth,axis=0)
    Ekxky_nonoise=np.zeros((ny,nx))
    for ix in range(nx):
        for iy in range(ny):
            ik=np.around(kn[iy,ix]/dk).astype(int)
            #print(ix,iy,ik,it,kn[iy,ix]/dk)
            Ekxky_nonoise[iy,ix]=Ekxky[iy,ix]-Emin[ik]
    
# make sure energy is exactly conserved (assuming kmax is consistent with fmax
    if verbose==1: 
        Hs2=4*np.sqrt(np.sum(np.sum(Ekth))*dkx*np.pi*2/ndir)
        Hs1=4*np.sqrt(np.sum(np.sum(Ekxky))*dkx*dky)
        print('Hs1,Hs2:',Hs1,Hs2)

    return Ekth,kn1,theta1,Ekxky_nonoise

    
###################################################################    
def dist_sphere(lo1,lo2,la1,la2):
    '''
    Computes spherical distance in radians 
    inputs :
            - lo1,lo2,la1,la2 : longitudes and latidudes (in degrees) 
    output : 
            - alpha : spherical distance from (lo1,la1) to (lo2,la2), in radians
            - beta1 : heading of great circle at (lo1,la1) that goes to (lo2,la2), in radians

    '''
    dtor=np.pi/180
    alpha=np.arccos(np.sin(la2*dtor)*np.sin(la1*dtor)+np.cos(la2*dtor)*np.cos(la1*dtor)*np.cos((lo2-lo1)*dtor))
    
    denom=(np.sin(alpha)*np.cos(la1*dtor))
    if (abs(denom) > 0.):
        ratio=(np.sin(la2*dtor)-np.cos(alpha)*np.sin(la1*dtor))/denom
        if (np.abs(ratio) > 1):
            ratio=np.sign(ratio)
        # Law of cosines to get cos(beta1)
        beta1 =np.arccos(ratio)
        # Law of sines to get sin(beta1)
        if (alpha > 0): 
            sinbeta=np.sin((lo2-lo1)*dtor)*np.cos(la2*dtor)/np.sin(alpha) 
            if (sinbeta < 0): 
                beta1=-beta1
    else:
        beta1=0.
    #beta2 =np.arccos((np.sin(la1*dtor)-np.cos(alpha)*np.sin(la2*dtor))/(np.sin(alpha)*np.cos(la2*dtor)))
    #beta2=np.arctan2(np.sin(lo2-lo1)*np.cos(la2),np.cos(la1)*np.sin(la2)-np.sin(la1)*np.cos(la2)*np.cos(lo2-lo1))
    #beta3=np.arctan2(np.sin(lo1-lo2)*np.cos(la1),np.cos(la2)*np.sin(la1)-np.sin(la2)*np.cos(la1)*np.cos(lo1-lo2))

    return alpha,beta1 #,beta2    
    
###################################################################
def SWOTdefine_swell_mask_storm(kx2,ky2,trackangle,lo1,la1,lo2,la2,tds,tola=2E6,tolr=0.25,thrcos=0.97,distshift=0,timeshift=0):
    '''
    Define mask based on storm position 
    inputs :
            - lo1,la1 : storm longitude and latitude 
            - lo2,la2 : observation longitude and lat. 
            - tola: tolerance on distance (in meters: 2E6 is 2000 km) 
            - tolr: relative tolerance on distance 
            - thrcos: threshold for cosine of direction (equivalent to a tolerance in angles) 
            - distshift : shift in anglular distance (radians) 
            - timeshift : shift in time (days)  
            
    output : amask; bmask. bmask is dilated compared to amask. 
    '''
    alpha,beta=dist_sphere(lo2,lo1,la2,la1)
    alpha=alpha+distshift
    dtor=np.pi/180
    dalpha=alpha*4E7/(2*np.pi)
    Cgt=dalpha/(tds+timeshift*86400)
    kt=9.81/(2*Cgt)**2/(2*np.pi)
    kt2=kt*np.max([dalpha/(dalpha-tola),1+tolr])
    kt1=kt*np.min([dalpha/(dalpha+tola),1-tolr])

    kn=np.sqrt(kx2**2+ky2**2)
    km=np.where((kn <kt2) & (kn > kt1),1,0)

    the=np.arctan2(kx2,ky2)/dtor
    amask=np.where((km*np.cos(beta-trackangle*dtor-the*dtor) > thrcos),1,0)

    bmask=ndimage.binary_dilation((amask > 0.5).astype(int))

    return amask,bmask
    
    ######################  Defines L,H parametric models based on updated JONSWAP shape + propagation 
def  Lmodel_eval(xdata,tds,incognita)  :
    '''
    Define wavelengths model
    inputs :
            - xdata : distances 
            - incognita : (2,) vector with 
                                [0] = distance shift (in radians, multiply by RE for distance),  
                                [1] = time shift (in days)
            
    output : - wavelengths
    '''
    fj=9.81*(tds+incognita[1]*86400)/((xdata+incognita[0])*2*40000*1000)
    Lj=9.81/(2*np.pi*fj**2)
    return fj,Lj


def  Lmodel(incognita,data)  :
    ydata =data[0]
    xdata =data[1]
    costfun=data[2]
    tds=data[3]
    fj,fff = Lmodel_eval(xdata,tds,incognita)
    thr=1E-5
   
    if costfun=='LS':
       cy= (   ((ydata - fff) **2)).sum()
    else:
       ratio = np.divide(ydata+thr,fff+thr) 
       cy= ( ratio - np.log(ratio)).sum()
    
    return cy

###############################################
def  Hmodel_eval(xdata,fj,gammaPM,incognita)  :
    '''
    Define swell heights model
    inputs :
            - xdata : distances 
            - incognita : (2,) vector with [0] = peak frequency, [1] = energy level
            
    output : - wavelengths
    '''
    fp=incognita[0]
    pow=17
    tac=5
    gamma=0 # energy dissipation ... 
    facPM=np.where(fj > fp,np.exp(-1.25*(fj/fp)**-4),np.exp(-1.25)*(fj/fp)**(5+pow*np.tanh(tac*(fp-fj)/fp)))
    #gammaPM=2 #incognita[2] #1.1
    H=incognita[1]*np.sqrt(fj**-5*facPM*gammaPM**(np.exp(-(fj-fp)**2/(2*(0.07*fp)**2))- gamma*xdata ))\
                                                      /np.sqrt(xdata*np.sin(xdata))
    return H


def  Hmodel(incognita,data)  :
    ydata =data[0]
    xdata =data[1]
    fj    =data[2]
    gammaPM=data[3]
    costpow=data[4]
    costfun=data[5]
    fff = Hmodel_eval(xdata,fj,gammaPM,incognita)
    thr=1E-5
    if costfun=='LS':
       cy= (   ((ydata*xdata**costpow - fff*xdata**costpow) **2)).sum()
    else:
       ratio = np.divide(ydata+thr,fff+thr) 
       cy= ( ratio - np.log(ratio)).sum()
    
    return cy
