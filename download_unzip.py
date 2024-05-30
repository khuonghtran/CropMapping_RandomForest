#import os
#from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
##Login
#user='yourusername'
#password='yourpassword'
#api = SentinelAPI(user,password, 'https://scihub.copernicus.eu/dhus')
#footprint = geojson_to_wkt(read_geojson(r"/gpfs/scratch/khtran/Data/SouthDakota/Crop_MLDL/WillowCreek.json"))
#print(footprint)
#products_2019=api.query(footprint,
#                        date=('20190401','20190501'),
#                        platformname='Sentinel-2',
#                        Producttype='S2MSI2A',
#                        cloudcoverpercentage=(0,100))
#print(len(products_2019))
#out="/gpfs/scratch/khtran/Data/SouthDakota/Crop_MLDL/zip/"
#os.chdir(out) #change directory to out
#for i in products_2019:
#    api.get_product_odata(i)
#    api.download(i)

#unzip	
import os, dozipfile
dir_name="/gpfs/scratch/khtran/Data/SouthDakota/Crop_MLDL/zip/"
outpath="/gpfs/scratch/khtran/Data/SouthDakota/Crop_MLDL/unzip/"
extension=".zip"
os.chdir(dir_name) #change directory to dir with files
for item in os.listdir(dir_name):
    if item.endswith(extension):
        file_name=os.path.abspath(item) #get full path of files
        zip_ref=zipfile.ZipFile(file_name)
        print(file_name)
        zip_ref.extractall(outpath)
        zip_ref.close()