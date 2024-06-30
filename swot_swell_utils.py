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
from scipy import ndimage
from  spectral_analysis_functions import *

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

# Corrects for 180 shift in direction
        shift180=0
        if np.abs(trackangle) > 90: 
          shift180=180
        cosm_SWOT=np.mean(np.multiply(kx2,E_mask).flatten())
        sinm_SWOT=np.mean(np.multiply(ky2,E_mask).flatten())
        dm_SWOT=np.mod(90-(-trackangle+shift180+np.arctan2(sinm_SWOT,cosm_SWOT)*180/np.pi),360) # converted to direction from, nautical
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
            print('Did not find model spectrum for location (lon,lat):',loncr,latcr,' at time ',timeww3)
        return modspec,inds,modelfound,timeww3,lonww3,latww3,dist,U10,Udir


###################################################################
###################################################################
def SWOTdefine_swell_mask_simple(Eta,coh,ang,medsig0,dlat,kx2,ky2,cohthr=0.3,cfac=8,mask_choice=0) :
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
             kxmin=-0.003,kxmax=0.003,kymin=-0.003,kymax=0.003):
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
       print('Emax:',np.max(Etam.flatten()),kxmin,kxmax,kymin,kymax)
       Emax=4/np.max(Etam.flatten())
       amask=Etam*Emax
       amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
       #amask=ndimage.binary_dilation((amask > 0.5).astype(int)) 
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
def  SWOT_save_spectra(pth_results,filenopath,modelfound,cycle,tracks,side,boxindices,\
                       lonc,latc,timec,trackangle,kx2,ky2,Eta,Etb,coh,ang,amask,sig0mean,sig0std,HH,HH2,Hs_SWOT_all,Hs_SWOT,Hs_SWOT_mask,Lm_SWOT,dm_SWOT, \
                       timeww3=0,lonww3=0,latww3=0,indww3=0,distww3=0,E_WW3_obp_H=0,E_WW3_obp_H2=0,E_WW3_noa_H2=0,Hs_WW3_all=0,Hs_WW3_cut=0,\
                       Hs_WW3_mask=0,Hs=0,Tm0m1=0,Tm02=0,Qkk=0,U10=0,Udir=0,Lm_WW3=0,dm_WW3=0, verbose=0)  :
   hemiNS=['A','N','S']
   hemiWE=['A','E','W']
   lonlat=f'{abs(latc):05.2f}'+hemiNS[int(np.sign(latc))]
   if modelfound==1:
       np.savez(pth_results+'SWOT_swell_spectra_'+cycle+'_'+tracks+'_'+side+'_'+lonlat,\
                fileSWOT=filenopath,cycle=cycle,tracks=tracks,side=side,boxindices=boxindices,\
                lonc=lonc,latc=latc,timec=timec,trackangle=trackangle,\
                kx2=kx2,ky2=ky2,E_SWOT=Eta,sig0_spec=Etb,coh=coh,ang=ang,amask=amask,\
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
