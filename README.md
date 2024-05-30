Code for the publication "10 m crop type mapping using Sentinel-2 reflectance and 30 m cropland data layer product"
International Journal of Applied Earth Observation and Geoinformation Volume 107, March 2022, 102692

This code is Copyright Mr. Khuong Tran, South Dakota State University. 
Use of this code for commercial purposes is not permitted. 
Contact khuong.tran@jacks.sdstate.edu for more information and updates of this code

Version 1.0
1 Mar 2020

----------------------------------------------------------------------------------------
**The code present in directory, including:**
1) download_unzip.py
-  This code was developed using the python SentinelAPI package for batch-downloading a large set of Sentinel-2 images
- And then unzip Sentinel-2 images for further processing
- Notes: please change your Scihub account and local directories
  
2) Sentinel2_Composites_allmonths.py 
- This is a function I developed to do monthly composite Sentinel-2 images using median method
- It is called in the Training_data.py

3) Training_data.py
- This code is very important!!!
- Basically, we wanted to create a set of training dataset from monthly composite Sentinel-2 images (10m pixels) and operational Cropland Data Layer, CDL (30m pixels)
- We developed a "centroid" method that can extract the best pixel of 10m within a 30m CDL pixel.
- The training data was then extracted as tubular (excel) data for training machine learning model

4) ML_RF_CDL10.py
- This code is for the training and testing Random Forest Classification for crop type mapping
- Notes: There is a section of linear interpolation to fill gaps in the input feature of Sentinel-2 time series

  
