#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for manipulating SWOT SSH data for swell analysis

Author: Fabrice Ardhuin 
Date: First version:  04.13.2024

Date: latest version: 04.13.2024


Dependencies:
    numpy, xarray, datetime, scipy
"""

import numpy as np
import datetime
import xarray as xr
from numpy.ma import masked_array

from netCDF4 import Dataset

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
        varmask=np.sum(E_mask.flatten())*dkx*dky*2;  # WARNING: this is only correct if the mask is only over half of the spectral domain!!
        Hs_SWOT=4*np.sqrt(varmask)

        kn=np.sqrt(kx2**2+ky2**2)
        m0=np.sum(E_mask.flatten())
        inds=np.where(kn.flatten() > 5E-4)[0]
        mm1=np.sum(E_mask.flatten()[inds]/kn.flatten()[inds])
        mp1=np.sum(np.multiply(E_mask,kn).flatten())
        Lmp1_SWOT=m0/mp1
        Lmm1_SWOT=mm1/m0

# NEED TO CHECK FOR 180° BUG ... 
        cosm_SWOT=np.mean(np.multiply(kx2,E_mask).flatten())
        sinm_SWOT=np.mean(np.multiply(ky2,E_mask).flatten())
        dm_SWOT=np.mod(90-(-trackangle+np.arctan2(sinm_SWOT,cosm_SWOT)*180/np.pi),360) # converted to direction from, nautical
    else :
        Hs_SWOT=NaN,Lmm1_SWOT=NaN,Lmp1_SWOT=NaN,dm_SWOT=NaN
    return Hs_SWOT,Lmm1_SWOT,Lmp1_SWOT,dm_SWOT


###################################################################
def  SWOTspec_mask_polygon(Eta,amask)  :
    '''
    Computes parameters from SWOT ssh spectrum
    inputs :
            - Eta  : spectrum
            - 
    output : 
            - vertices

    '''
    masked_data = masked_array(Eta, mask=(amask > 0.5) )
    # Get coordinates of masked cells
    rows, cols = np.where(masked_data.mask)
    # Iterate through masked cells and draw polygons around them
    vertices = []
    [nkyr,nkxr]=np.shape(Eta)  
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
        lonww3=0.;latww3=0.;    
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
            #print('COUCOU lon:',loncr,latcr,timec,timeww3a,'##',len(indt),dist,ds_ww3t.longitude[inds].values,ds_ww3t.latitude[inds].values)
            modspec=ds_ww3t.efth[inds].squeeze()
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
            inds=0
            print('Did not find model spectrum for location (lon,lat):',loncr,latcr,' at time ',timeww3)
        return modspec,inds,modelfound,timeww3,lonww3,latww3,dist

###################################################################
# modpec,inds=swell.SWOTfind_model_spectrum(ds_ww3t,loncr,latcr,timec)
def SWOTdefine_swell_mask(Eta,coh,mask_choice,kx2,ky2,cohthr) :
    indc=np.where((coh > cohthr))[0]
    ncoh=len(indc)
    
    kp=np.where(kx2 > 0,1.,0.)
    kpy=np.where(ky2 > 0,1.,0.)
    ncoh2=0
    #if ncoh < 6:
    cohmax=np.max(coh.flatten())
    cohthr=cohmax/2
    indc2=np.where((coh > cohthr) )[0]
    ncoh2=len(indc2)
    cohOK=1
    if ncoh2 > 3:
       ncoh=ncoh2
    if cohthr > 0.8/cfac and ncoh > 3 and (np.nanmedian(mybos) < 40):
       cohs=(coh/cohthr)*np.sign(np.cos(ang))*np.sign(ky2*dlat)
       amask=ndimage.binary_dilation((cohs > 1 ).astype(int))
    else:
       cohOK=0
       amask=Eta/10*np.sign(ky2*dlat)
    if (cohOK==1) or (mask_choice < 0):
       amask=Eta/10*np.sign(ky2*dlat)
       Emax=2/np.max(Eta.flatten())
       amask=Eta*Emax*np.sign(-ky2)
    
    ind=np.where(amask.flatten() > 0.5)[0]
    if len(ind) >0 :
          indk=np.argmax(Eta.flatten()[ind])
          # defines k magnitude at spectral peak for further filtering ...   
          knm=kn.flatten()[ind[indk]]
          print('KNM:',knm,Eta.flatten()[ind[indk]],np.max(Eta),ncoh,ncoh2,'coh:',cohthr)
      
    rows, cols =np.where(kn > 2*knm  )
    for r, c in zip(rows, cols):               
          amask[r,c]=0
    rows, cols =np.where(kn < 0.6*knm  )
    for r, c in zip(rows, cols):               
          amask[r,c]=0
      
    # Forces mask : here are a few choices ... 
       
    if mask_choice==1:
        amask=np.multiply(np.where(abs(kx2) <= 0.0006,1,0),np.where(abs(ky2-0.0015) <= 0.0003,1,0))
      #amask=np.multiply(np.where(abs(kx2-0.000) < 0.0012,1,0),np.where(abs(ky2-0.0015) < 0.0003,1,0))
    if mask_choice==2:
    #Forcing for Norfolk island swell on track 17
        amask=np.multiply(np.where(abs(kx2+0.001) <= 0.0003,1,0),np.where(abs(ky2-0.000) <= 0.0004,1,0))

        
    bmask=ndimage.binary_dilation((amask > 0.5).astype(int))
    return amask,bmask

###################################################################

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