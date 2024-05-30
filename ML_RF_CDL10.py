import os
import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt
import importlib
import Sentinel2_Composites_allmonths
importlib.reload(Sentinel2_Composites_allmonths)
from Sentinel2_Composites_allmonths import median_def

#Call def
#concatenate all monthly medians Apr - Oct (7months) as raw data
medians = np.concatenate([median_def(i) for i in range (4,11)])
rawmedians=medians[:] #unchange copy 

#create cloud mask >4
B2=np.stack(medians[i,:,:] for i in range(0,35,5))
B2_medians=B2.reshape(7,-1) #shape 7,10980^2
count=sum(np.isnan(B2_medians)) #shape 1,
cloud=count.reshape(10980,-1)
#im=Image.fromarray(cloud.astype('uint8')
#im.save('/gpfs/scratch/khtran/Data/2020/CDL10m/SiouxFalls/out/v5/cloud.tif')
cloudmask=np.ones((10980,10980))
cloudmask[cloud>4]=np.nan

#interpolation
interps=[]
for x in range(0,5):
	Bx=np.stack(medians[i,:,:] for i in range(x,35,5))
	Bx_medians=Bx.reshape(7,-1)
	Bx_df=pd.DataFrame(Bx_medians)
	Bx_medians_interp=Bx_df.interpolate(method='linear', limit_direction='both', axis=0)
	Bx_medians_interp_arr=Bx_medians_interp.to_numpy() #shape 7,-1
	Bx_interp=Bx_medians_interp_arr.reshape(7,10980,-1) #shape 7 10980 10980
	interps.append(Bx_interp)
	print(x,"done")
#Reshape to original structure
data_interps=np.stack([[interps[0][j,:,:],interps[1][j,:,:],interps[2][j,:,:],interps[3][j,:,:],interps[4][j,:,:]] for j in range(0,7)])
alldata=data_interps.reshape((35,-1)).transpose() #shape 10980^2,35
alldata_df=pd.DataFrame(alldata)
#count nan
#alldata_df.isna().sum().sum()
#replace npnan by fillvalue -999.0 for 4 pixels has 7 months missing data
inputdata=alldata_df.replace(np.nan,-999.0) ##interpolated ###############

#Read csv training data
trainingdata=pd.read_csv('/gpfs/scratch/khtran/2020/CDL10m/SiouxFalls/out/v5/Trainingdata_SF_20.csv')
data=trainingdata.iloc[:,1:]
label=trainingdata.iloc[:,0]
# # plot average of each band
# bandnames=['Blue','Green','Red','NIR','NDVI']
# nbands= len(bandnames)
# labels = [61, 1, 5, 36, 37]
# cropnames=['Fallow/Idle Cropland', 'Corn', 'Soybeans','Alfalfa','Other Hay/Non-Alfalfa']
# nfeatures=data.values.shape[1]
# for bandid in range(len(bandnames)):
	# plt.clf()
	# for i in range(len(labels)):
		# colors = ['olive', 'gold', 'darkgreen', 'm','lime']
		# linestyles =  ['-', '--', ':', '-.','-']
		# markers =['o','^','s','x','+']
		# ax = [k1+4 for k1 in range(0,int(nfeatures/nbands))]
		# x_class = data[label==labels[i]]
		# aymean =  np.nanmean(x_class,axis=0)
		# #bandid = 6 #ndvi
		# ayband = [aymean[j] for j in range(bandid,nfeatures,nbands)]
		# z=plt.plot(ax,ayband, marker=markers[i],linestyle=linestyles[i],color=colors[i],label=cropnames[i])
	# z=plt.xlabel('Month')
	# if bandid<4:
		# z=plt.ylabel('Surface reflectance')
	# else:
		# z=plt.ylabel('NDVI')
	# z=plt.legend()
	# z=plt.title('Median '+bandnames[bandid],fontsize = 12,fontweight='bold')
	# figdir = '/gpfs/scratch/khtran/2020/CDL10m/SiouxFalls/out/v5/'
	# z=plt.savefig(figdir+bandnames[bandid]+'_Timeseries_'+'SF.png')
#Split the training data
data_train, data_test, label_train, label_test = train_test_split(data,label,test_size=0.2,random_state=42)

#Setting classifier, train model
clf=RandomForestClassifier(n_estimators=500)
clf.fit(data_train,label_train)

#Accuracy checking
label_pred=clf.predict(data_test)
accuracy_score(label_pred,label_test)
#Confusion matrix
confusion_matrix(label_test,label_pred)
#Plot confusion matrix

matrix=plot_confusion_matrix(clf, data_test, label_test,cmap=plt.cm.Reds)
matrix.ax_set_title('Confusion Matrix', color='white')
plt.xlable('Predicted Lable', color='white')
plt.ylable('Actual Lable', color='white')
plt.gcf().axes[0].tick_params(color='white')
plt.gcf().axes[1].tick_params(color='white')
plt.gcf().set_size_inches(10,15)
plt.show()

#Image classification
CDL_pred=clf.predict(inputdata) 
CDL_classified=CDL_pred.reshape((10980,-1)).astype('uint8')
CDL_classified[np.isnan(cloudmask)]=0
#print(CDL_classified.shape)

#Write CDL_ML result
#Read coordinate system
path_coorsys= '/gpfs/scratch/khtran/Data/2020/CDL10m/SiouxFalls/unzip/S2A_MSIL2A_20190403T172011_N0211_R012_T14TPP_20190403T213749.SAFE/GRANULE/L2A_T14TPP_A019743_20190403T173034/IMG_DATA/R10m/'
im_coorsys=rasterio.open(path_coorsys+'T14TPP_20190403T172011_B02_10m.jp2')

CDL10m = rasterio.open('/gpfs/scratch/khtran/Data/2020/CDL10m/SiouxFalls/out/v5/CDL10m_SF_2019.tiff','w',driver='GTiff',
                         width=im_coorsys.width, height=im_coorsys.height,
                         count=1,
                         crs=im_coorsys.crs,
                         transform=im_coorsys.transform,
                         dtype='uint8'
                         )
CDL10m.write(CDL_classified,1)
CDL10m.close()