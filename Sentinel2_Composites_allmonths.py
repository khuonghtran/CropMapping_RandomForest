import os
import rasterio
import numpy as np
import scipy.ndimage
from PIL import Image as image

dir_name="/gpfs/scratch/khtran/2020/CDL10m/SiouxFalls/unzip/"
def median_def(month):
	Stacks=[]
	for root,dirs,files in os.walk(dir_name):
		for file in sorted(files):
			#print(file)
			if file.endswith('.jp2') and "SCL_20m" in file and file[11:13]==str(month).zfill(2):
				#acquistion date
				acqdate = file[7:15]
				filename_SCL = os.path.join(root, file)
				#print(filename_SCL)
				SCL=rasterio.open(filename_SCL).read(1).astype('float32')
				
				#resample to 10m - nearest method
				SCL_resample=scipy.ndimage.zoom(SCL, 2, order=0)
				#keep 2,4,5,6,7 - bug version 1######################################################
				SCL_resample[(SCL_resample==0.)|(SCL_resample==1.)|(SCL_resample==3.)|(SCL_resample==8.)|(SCL_resample==9.)|(SCL_resample==10.)|(SCL_resample==11.)]=np.nan #assign nan as cloudy pixels
				
				#print(SCL_resample)


				#find other bands with same acquistion date
				for root,dirs,files in os.walk(dir_name):
					for file1 in sorted(files):
						#print(file1)
						if acqdate in file1 and "B02_10m" in file1 and file1.endswith('.jp2'):
							filename_B2 = os.path.join(root, file1)
							#print(filename_B2)
							B2=rasterio.open(filename_B2).read(1).astype('float32')
							B2[np.isnan(SCL_resample)]=np.nan
							#print(B2)

						if acqdate in file1 and "B03_10m" in file1 and file1.endswith('.jp2'):
							filename_B3 = os.path.join(root, file1)
							#print(filename_B3)
							B3=rasterio.open(filename_B3).read(1).astype('float32')
							B3[np.isnan(SCL_resample)]=np.nan
							#print(B3)
	 
						if acqdate in file1 and "B04_10m" in file1 and file1.endswith('.jp2'):
							filename_B4 = os.path.join(root, file1)
							#print(filename_B4)
							B4=rasterio.open(filename_B4).read(1).astype('float32')
							B4[np.isnan(SCL_resample)]=np.nan
							#print(B4)
						
						if acqdate in file1 and "B08_10m" in file1 and file1.endswith('.jp2'):
							filename_B8 = os.path.join(root, file1)
							#print(filename_B8)
							B8=rasterio.open(filename_B8).read(1).astype('float32')
							B8[np.isnan(SCL_resample)]=np.nan
							#print(B8)
							

							#NDVI calculation
							ndvi=np.where((B4+B8)==0.,np.nan,(B8-B4)/(B8+B4))
							ndvi[np.isnan(SCL_resample)]=np.nan

							#stack image
							imagestacked=np.stack((B2,B3,B4,B8,ndvi), axis=0)
							Stacks.append(imagestacked)

	#Calculate monthly median
	B2 = [Stacks[x][0,:,:] for x in range (0,len(Stacks))]
	B2_median = np.nanmedian(B2, axis=0)
	B3 = [Stacks[x][1,:,:] for x in range (0,len(Stacks))]
	B3_median = np.nanmedian(B3, axis=0)
	B4 = [Stacks[x][2,:,:] for x in range (0,len(Stacks))]
	B4_median = np.nanmedian(B4, axis=0)
	B8 = [Stacks[x][3,:,:] for x in range (0,len(Stacks))]
	B8_median = np.nanmedian(B8, axis=0)
	ndvi = [Stacks[x][4,:,:] for x in range (0,len(Stacks))]
	ndvi_median = np.nanmedian(ndvi, axis=0)
	#Stack monthly median
	median = np.stack((B2_median,B3_median,B4_median,B8_median,ndvi_median),axis=0)
	return median

#concatenate all monthly medians Apr - Oct (7months)
#medians = np.concatenate([median_def(i) for i in range (4,11)])
