import os
import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage

import importlib
import Sentinel2_Composites_allmonths
importlib.reload(Sentinel2_Composites_allmonths)
from Sentinel2_Composites_allmonths import median_def

#Call def
#concatenate all monthly medians Apr - Oct (7months)
medians = np.concatenate([median_def(i) for i in range (4,11)])
rawmedians=medians[:] #unchange copy 
#mask all cloudy pixels as an nan array
#create a np.ones mask
mask=np.ones((10980,10980))
for i in range (0,35,5):
  #print(i)
  mask[np.isnan(medians[i,:,:])]=np.nan
print("cloudy:",np.count_nonzero(np.isnan(mask)))
  
#mask all 35 metrics using the mask above
for i in range (0,35):
  print("loop to mask all cloudy pixel over 7 months: Step ",i)
  medians[i,:,:][np.isnan(mask)]= np.nan
#compare with mask above
print("cloudy:",np.count_nonzero(np.isnan(medians[2,:,:])))

# Derive traning mask 10m for all pixels nan vs 1-selected pixel
train_mask=np.empty((10980,10980))*np.nan
for i in range (0,3660,20): #outer loop 30m, filter every 20 pixels
  for j in range (0,3660,20):
    data10m = medians[:,i*3:i*3+3,j*3:j*3+3] #subset 10m pixel falls in 30m pixels
    print("count the number of non-nan pixels in 9 pixels: ",np.count_nonzero(~np.isnan(data10m[1,:,:])))
    if np.count_nonzero(~np.isnan(data10m[1,:,:])) >=5:
      data10m_mean=np.nanmean(data10m,axis=(1,2)) #an 1D array numpy
      data10m_mean=np.tile(data10m_mean[:,np.newaxis,np.newaxis], (1,3,3))
      print(data10m_mean[1,:,:])
      #find location of min distance from 10m pixels to mean
      x=np.nanargmin(np.sum(np.absolute((data10m-data10m_mean)),axis=0))
      print("location of pixel with min distance: ",x)
      train_mask[i*3:i*3+3,j*3:j*3+3][(x//3,x-(3*(x//3)))]=1
#print(train_mask.shape)
#np.set_printoptions(threshold=np.inf)
#print(train_mask[1,:])
#print(np.count_nonzero(~np.isnan(train_mask)))

#Working with CDL
CDL=rasterio.open('/gpfs/scratch/khtran/Data/2020/CDL10m/SiouxFalls/unzip/T14TPPCDL2019.tif').read(1).astype('float32')
CDL=np.where((CDL==122)|(CDL==123)|(CDL==124),121,CDL) #combine all 4 developed into developed land
#Filtering CDL every 20 pixels + No edge + homogeneous
maskCDL=np.empty((3660,3660))*np.nan
for i in range (0,3660,20):
    for j in range (0,3660,20):
        print(i,j)
        CDL_9cells=CDL[i-1:i+2,j-1:j+2]
        #print(CDL_9cells)
        #print(CDL[i,j])
        if (CDL_9cells.size==9) and (CDL_9cells==CDL[i,j]).all(): # size ==9 removes edge pixels  #bug version 1 ##################################
            maskCDL[i,j]=1
            #print(maskCDL[i,j])

CDL[np.isnan(maskCDL)]=np.nan
#keep majority classes
#keep developed + open water + >1% in 30mCDL
CDL=np.where((CDL!=121)&(CDL!=111)&(CDL!=61)&(CDL!=1)&(CDL!=5)&(CDL!=176)&(CDL!=195)&(CDL!=36)&(CDL!=37)&(CDL!=141),np.nan,CDL)
#Final train mask for both CDL and medians metrics
CDL_resample=np.kron(CDL, np.ones((3,3))) #resample 1 to 3x3 : to 10980x10980
#Creating final training mask - use train_mask above to keep min distance pixels
train_mask[np.isnan(CDL_resample)]=np.nan
#Stack CDL_resample and median 36x10980x10980
CDL_resample=CDL_resample[np.newaxis,:, :] #resample to [1, 10980, 10980]
Alldata_stacked=np.concatenate((CDL_resample,medians))

#Use final train_mask to keep only min distance + CDL filtered pixels 
for i in range (0,36):
  print("loop to all data 36 metrics to select only training pixels ",i)
  Alldata_stacked[i,:,:][np.isnan(train_mask)]=np.nan
  
#print("shape of all-data metric:",Alldata_stacked.shape)
#reshape Alldata_stacked into dataframe 2D [10980*10980, 36]
Alldata=np.reshape(Alldata_stacked,[36,-1]).transpose()
#print("shape of traing data:",Alldata.shape)  
#remove all nan pixels on Alldata
trainingdata= Alldata[~np.isnan(Alldata)].reshape(-1,36)
df_train=pd.DataFrame(trainingdata, columns =['CDL','AprB2','AprB3','AprB4','AprB8','AprNDVI','MayB2','MayB3','MayB4','MayB8','MayNDVI','JunB2','JunB3','JunB4','JunB8','JunNDVI','JulB2','JulB3','JulB4','JulB8','JulNDVI','AugB2','AugB3','AugB4','AugB8','AugNDVI','SepB2','SepB3','SepB4','SepB8','SepNDVI','OctB2','OctB3','OctB4','OctB8','OctNDVI'])
#Print("Prepare dataframe")
print(df_train)
df_train.to_csv("/gpfs/scratch/khtran/Data/2020/CDL10m/SiouxFalls/out/v4/Trainingdata_SF_20.csv",index=False)

#np.count_nonzero(~np.isnan(CDL))

#from PIL import Image
#im=Image.fromarray(train_mask)
#im.save('/gpfs/scratch/khtran/Data/SouthDakota/Crop_MLDL/out/train_mask.tif')

#path_coorsys= '/gpfs/scratch/khtran/Data/SouthDakota/Crop_MLDL/unzip/S2A_MSIL2A_20190403T172011_N0211_R012_T14TPP_20190403T213749.SAFE/GRANULE/L2A_T14TPP_A019743_20190403T173034/IMG_DATA/R10m/'
#im_coorsys=rasterio.open(path_coorsys+'T14TPP_20190403T172011_B02_10m.jp2')
#export true color image

#write image
#truecolor = rasterio.open('/gpfs/scratch/khtran/Data/SouthDakota/Crop_MLDL/out/Aprmedian_truecolor.tiff','w',driver='GTiff',
#                         width=im_coorsys.width, height=im_coorsys.height,
#                         count=3,
#                         crs=im_coorsys.crs,
#                         transform=im_coorsys.transform,
#                         dtype=im_coorsys.dtypes[0]
#                         )
#truecolor.write(rawmedians[0,:,:].astype('uint16'),3) #blue
#truecolor.write(rawmedians[1,:,:].astype('uint16'),2) #green
#truecolor.write(rawmedians[2,:,:].astype('uint16'),1) #red
#truecolor.close()

#show image
#from rasterio import plot
#import matplotlib.pyplot as plt
#%matplotlib inline
#plot.show(band4)
##type of raster byte
#band4.dtypes[0]
##raster sytem of reference
#band4.crs
##raster transform parameters
#band4.transform