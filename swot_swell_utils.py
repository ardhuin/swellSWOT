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


from wave_physics_functions import wavespec_Efth_to_Ekxky

import matplotlib.colors as mcolors


from numpy.ma import masked_array
from scipy import ndimage
from  spectral_analysis_functions import *
from  lib_filters_obp import *

from netCDF4 import Dataset

###################################################################
def spec_settings_for_L3(nres,version):
    dx=250
    dy=235
    indxc=259
    ISHIFT=30   # start of 40 km box for spectral analysis, in pixels, relative to track center
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
    if len(Ekxky > 0):
        E_mask=np.where( swell_mask > 0.5, np.divide(Ekxky,Hhat2),0) 
        dkx=kx2[0,1]-kx2[0,0]
        dky=ky2[1,0]-ky2[0,0]
        varmask=np.sum(E_mask.flatten())*dkx*dky*2;  # WARNING: factor 2  is only correct if the mask is only over half of the spectral domain!!
        Hs_SWOT=4*np.sqrt(varmask)

        kn=np.sqrt(kx2**2+ky2**2)
        m0=np.sum(E_mask.flatten())
        mQ=np.sum((E_mask.flatten())**2)
        Q18=np.sqrt(mQ/(m0**2*dkx*dky))/(2*np.pi)
        inds=np.where(kn.flatten() > 5E-4)[0]
        mm1=np.sum(E_mask.flatten()[inds]/kn.flatten()[inds])
        mp1=np.sum(np.multiply(E_mask,kn).flatten())
        Lmp1_SWOT=m0/mp1
        Lmm1_SWOT=mm1/m0

# Corrects for 180 shift in direction
        shift180=0
        if np.abs(trackangle) > 90: 
          shift180=180
        cosm_SWOT=np.mean(np.multiply(kx2,E_mask).flatten())
        sinm_SWOT=np.mean(np.multiply(ky2,E_mask).flatten())
        dm_SWOT=np.mod(90-(-trackangle+shift180+np.arctan2(sinm_SWOT,cosm_SWOT)*180/np.pi),360) # converted to direction from, nautical
    else :
        Hs_SWOT=NaN,Lmm1_SWOT=NaN,Lmp1_SWOT=NaN,dm_SWOT=NaN
    return Hs_SWOT,Lmm1_SWOT,Lmp1_SWOT,dm_SWOT,Q18


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
def  SWOT_create_L3_spectra(saving_name,filenopath,modelOK,restab,nkxtab,nkytab,modf=0,moddf=0,modang=0):
     SL3_nc_fid = Dataset(saving_name, 'w',format='NETCDF3_CLASSIC')
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
        SL3_nc_fid.createDimension('nky_'+sres,nkytab[indres])
        SL3_nc_fid.createDimension('nkx_'+sres,nkxtab[indres])
     if modelOK > 0:
        nf=len(modf)
        ntheta=len(modang)
        SL3_nc_fid.createDimension('nf',nf)
        SL3_nc_fid.createDimension('ntheta',ntheta)

     time = SL3_nc_fid.createVariable('time', np.float64, ('time'))
     #time.units = 'days since 1990-01-01'
     time.units = 'seconds since 2000-01-01 00:00:00.0'
     time.long_name = 'time'

     boxindices= SL3_nc_fid.createVariable('boxindices_40', np.float64, ('time','nboy_40','nbox_40','nind'))
     boxindices.setncatts({'comment': u"these are along-track indices j1, j2  followed by cross-track indices i1, i2"})

     
     
     SL3_nc_track = SL3_nc_fid.createVariable('trackangle', np.float32, ('time'))
     SL3_nc_track.setncatts({'long_name': u"track_angle", \
                    'units': u"degrees", \
                    'comment': u"clockwise_blabla"})
                    
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
      
        SL3_nc_var = SL3_nc_fid.createVariable('kx2_'+sres, np.float32, ('nky_'+sres, 'nkx_'+sres))
        SL3_nc_var.setncatts({'long_name': u"spatial_frequency", \
                    'units': u"cycles per meter", \
                    'comment': u"kx is cross-track to right"})

        SL3_nc_var = SL3_nc_fid.createVariable('ky2_'+sres, np.float32, ('nky_'+sres, 'nkx_'+sres))
        SL3_nc_var.setncatts({'long_name': u"spatial_frequency", \
                    'units': u"cycles per meter", \
                    'comment': u"ky is along-track"})

        SL3_nc_filt = SL3_nc_fid.createVariable('filter_OBP_'+sres, np.float32, ('nky_'+sres, 'nkx_'+sres))
        SL3_nc_filt.setncatts({'long_name': u"instrument_filter_OBP", 'units': u"1"})

        SL3_nc_filt = SL3_nc_fid.createVariable('filter_PTR_'+sres, np.float32, ('nky_'+sres, 'nkx_'+sres))
        SL3_nc_filt.setncatts({'long_name': u"instrument_filter_PTR", 'units': u"1"})


 
# For higher resolution: only spectra and integrated parameters 
        SL3_nc_varE4 = SL3_nc_fid.createVariable('E_SWOT_'+sres, np.float32, ('time','nboy_'+sres,'nbox_'+sres,'nky_'+sres, 'nkx_'+sres), zlib=True)
        SL3_nc_varE4.setncatts({'long_name': u"PSD of surface elevation over 40 km side box", 'units': u"m**4"})                   
        SL3_nc_varE4 = SL3_nc_fid.createVariable('mask_'+sres, np.byte, ('time','nboy_40','nbox_40','nky_'+sres, 'nkx_'+sres))
        SL3_nc_varE4.setncatts({'long_name': u"mask_for_wind_sea_and_swell"})

     SL3_nc_varE4 = SL3_nc_fid.createVariable('coh_SWOT_40', np.float32, ('time','nboy_40','nbox_40','nky_40', 'nkx_40'))
     SL3_nc_varE4.setncatts({'long_name': u"cohenrence between SSH and sigma0 over 40 km side box",   'units': u"1"})                
     SL3_nc_varE4 = SL3_nc_fid.createVariable('ang_SWOT_40', np.float32, ('time','nboy_40','nbox_40','nky_40', 'nkx_40'))
     SL3_nc_varE4.setncatts({'long_name': u"mean phase shift between SSH and sigma0 over 40 km side box",  'units': u"1"})     
     SL3_nc_varE4 = SL3_nc_fid.createVariable('crosr_SWOT_40', np.float32, ('time','nboy_40','nbox_40','nky_40', 'nkx_40'))
     SL3_nc_varE4.setncatts({'long_name': u"real part of cross-spectrum between SSH and sigma0 over 40 km side box",  'units': u"rad"})                


     if modelOK > 0:
        SL3_nc_var = SL3_nc_fid.createVariable('frequency', np.float32, ('nf'))
        SL3_nc_var = SL3_nc_fid.createVariable('df', np.float32, ('nf'))
        SL3_nc_var = SL3_nc_fid.createVariable('direction', np.float32, ('ntheta'))
        SL3_nc_var = SL3_nc_fid.createVariable('efth_model', np.float32, ('time','nboy_40','nbox_40','nf','ntheta'), zlib=True)
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
