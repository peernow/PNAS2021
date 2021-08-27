#!/usr/bin/env python
### uses python/anaconda/2019.10/3.7
### example file to carry out the cloud controlling factor Ridge regressions for shortwave (SW) data
### learns Ridge regression coefficients from historical reanalysis/historical+RCP (until 2019) CMIP data
### pre-processed input files can be found on figshare (see readme.txt)
### author: Peer Nowack
domain_size = 'triple'
spectrum = 'SW'
### for file naming
area_code = spectrum + '_' + domain_size

### import necessary Python packages
import numpy as np
import scipy
import netCDF4
from sklearn.preprocessing import StandardScaler                                                                                                                 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import joblib
from sklearn.metrics import r2_score
import glob

### paths to files - adapt
path_to_files = './'

### list identifiers for the datasets used to train each Ridge regression 
### (four reanalyses and 52 CMIP5/CMIP6 models)
model_list = ['CFSR', 'ERA5', 'JRA-55', 'MERRA2', 'ACCESS-CM2_historical-ssp_r1', 
              'ACCESS-ESM1-5_historical-ssp_r1', 'ACCESS1-0_historical-rcp45_r1',
              'ACCESS1-3_historical-rcp45_r1', 'BCC-CSM2-MR_historical-ssp_r1', 
              'BCC-ESM1_historical-ssp_r1','BNU-ESM_historical-rcp45_r1','CCSM4_historical-rcp45_r1',
              'CESM2-WACCM_historical-ssp_r1','CESM2_historical-ssp_r1','CNRM-CM5_historical-rcp45_r1',
              'CNRM-CM6-1_historical-ssp_r1','CNRM-ESM2-1_historical-ssp_r1',
              'CSIRO-Mk3-6-0_historical-rcp45_r1','CanESM2_historical-rcp45_r1', 'CanESM5_historical-ssp_r1',
              'EC-Earth3-Veg_historical-ssp_r1','FGOALS-f3-L_historical-ssp_r1','FGOALS-g3_historical-ssp_r1',
              'GFDL-CM3_historical-rcp45_r1','GFDL-CM4_historical-ssp_r1','GFDL-ESM2G_historical-rcp45_r1',
              'GFDL-ESM2M_historical-rcp45_r1','GISS-E2-1-G_historical-ssp_r1','GISS-E2-H_historical-rcp45_r1',
              'GISS-E2-R_historical-rcp45_r1','HadGEM2-ES_historical-rcp45_r1',
              'HadGEM3-GC31-LL_historical-ssp_r1','INM-CM4-8_historical-ssp_r1','INM-CM5-0_historical-ssp_r1',
              'IPSL-CM5A-LR_historical-rcp45_r1','IPSL-CM5A-MR_historical-rcp45_r1',
              'IPSL-CM5B-LR_historical-rcp45_r1','IPSL-CM6A-LR_historical-ssp_r1',
              'MIROC-ES2L_historical-ssp_r1','MIROC-ESM_historical-rcp45_r1','MIROC5_historical-rcp45_r1',
              'MIROC6_historical-ssp_r1','MPI-ESM-LR_historical-rcp45_r1','MPI-ESM-MR_historical-rcp45_r1',
              'MPI-ESM1-2-HR_historical-ssp_r1','MPI-ESM1-2-LR_historical-ssp_r1',
              'MRI-CGCM3_historical-rcp45_r1','MRI-ESM2-0_historical-ssp_r1','NESM3_historical-ssp_r1',
              'NorESM1-M_historical-rcp45_r1','NorESM2-LM_historical-ssp_r1','NorESM2-MM_historical-ssp_r1',
              'UKESM1-0-LL_historical-ssp_r1','bcc-csm1-1-m_historical-rcp45_r1',
              'bcc-csm1-1_historical-rcp45_r1','inmcm4_historical-rcp45_r1']

### all data has been interpolated to a common 5x5 degree grid

nr_lat = 36
nr_lon = 72
nr_models = len(model_list)

### define number of controlling factor variables 
### (needed to define array sizes below) 
nr_planes = 5

### read longitude and latitude coordinates from one example file

lon = netCDF4.Dataset(path_to_files+'data/CMIP/inputs/hur700_Amon_ACCESS1-0_historical_r1i1p1.nc')['lon'][:]
lat = netCDF4.Dataset(path_to_files+'data/CMIP/inputs/hur700_Amon_ACCESS1-0_historical_r1i1p1.nc')['lat'][:]

### define list of regularization parameters that each Ridge regression is cross-validated over
alpha_i=[0,1e-12,3e-12,1e-11,3e-11,1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,0.0001,
         0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,10000,30000,100000,300000,1000000,
         3000000,1e7,3e7,1e8,3e8,1e9,3e9,1e10,3e10,1e11,3e11,1e12,3e12]
### y-data is zero-meaned; so don't fit the intercept

parameters = {
    'alpha': alpha_i,
    'fit_intercept': [False],
    'max_iter':[1000],
    'random_state':[100]
             }
### prepare for data storage
dict_regr_results = {}
### prepare numpy array that can hold all learned coefficients
### (for each reanalysis/climate model dataset)
### prepare for easy 'circular' read out of the data across longitudes for plotting
coeffs = np.empty((nr_models,nr_lat,nr_lon,nr_planes,nr_lat,nr_lon*3))
number_samples = np.empty((nr_models,nr_lat,nr_lon))
# import warnings
# warnings.filterwarnings('ignore')
### define number of consistent timesteps to select (relies on consistent data pre-processing)
nt=235

### loop over all models, latitudes, and longitudes
### to train individual Ridge regressions
for modeli in range(0,nr_models):
    alpha_list = []
    dict_regr_results[model_list[modeli]] = []
    print(model_list[modeli])
    for lati in range(0,nr_lat):
        ### data needed to mask out latitudes with unreliable satellite data 
        ### (see Methods section in main paper)
        rsdt_pi = netCDF4.Dataset(path_to_files+'data/solar_zenith/solar_zenith_angle_histrcp.nc')['sza'][:nt,lati]
        for loni in range(0,nr_lon):
            lon_sel = loni+nr_lon
            ### defines number of grid points/size of box around the target grid point considered in each regression
            lond = 10
            latd = 5
            ### take into account slightly different data formats after pre-processing when reading in data
            if model_list[modeli] in ['CFSR']:
                hur700_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CFSR/inputs/hur700_*"+"*.nc")[0])['hur'][:nt,:,:]
                wap500_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CFSR/inputs/wap500_*"+"*.nc")[0])['wap'][:nt,:,:]
                utrh_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CFSR/inputs/utrh_*"+"*.nc")[0])['utrh'][:nt,:,:]
                ts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CFSR/inputs/ts_*"+"*.nc")[0])['ts'][:nt,:,:]
                eislts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CFSR/inputs/eislts_*"+"*.nc")[0])['eislts'][:nt,:,:]
                Y_pi = netCDF4.Dataset(path_to_files+"data/obs_clouds/albcld_200003-201909.nc")['albcld'][:nt,lati,loni]
            elif model_list[modeli] in ['ERA5']:
                hur700_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/ERA5/inputs/hur700_*"+"*.nc")[0])['hur'][:nt,:,:]
                wap500_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/ERA5/inputs/wap500_*"+"*.nc")[0])['wap'][:nt,:,:]
                utrh_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/ERA5/inputs/utrh_*"+"*.nc")[0])['utrh'][:nt,:,:]
                ts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/ERA5/inputs/ts_*"+"*.nc")[0])['ts'][:nt,:,:]
                eislts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/ERA5/inputs/eislts_*"+"*.nc")[0])['eislts'][:nt,:,:]
                Y_pi = netCDF4.Dataset(path_to_files+"data/obs_clouds/albcld_200003-201909.nc")['albcld'][:nt,lati,loni]
            elif model_list[modeli] in ['MERRA2']:
                hur700_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/MERRA2/inputs/hur700_*"+"*.nc")[0])['hur'][:nt,:,:]
                wap500_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/MERRA2/inputs/wap500_*"+"*.nc")[0])['wap'][:nt,:,:]
                utrh_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/MERRA2/inputs/utrh_*"+"*.nc")[0])['utrh'][:nt,:,:]
                ts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/MERRA2/inputs/ts_*"+"*.nc")[0])['ts'][:nt,:,:]
                eislts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/MERRA2/inputs/eislts_*"+"*.nc")[0])['eislts'][:nt,:,:]
                Y_pi = netCDF4.Dataset(path_to_files+"data/obs_clouds/albcld_200003-201909.nc")['albcld'][:nt,lati,loni]
            elif model_list[modeli] in ['JRA-55']:
                hur700_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/JRA55/inputs/hur700_*"+"*.nc")[0])['hur'][:nt,0,:,:]
                wap500_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/JRA55/inputs/wap500_*"+"*.nc")[0])['wap'][:nt,:,:]
                utrh_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/JRA55/inputs/utrh_*"+"*.nc")[0])['utrh'][:nt,:,:]
                ts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/JRA55/inputs/ts_*"+"*.nc")[0])['ts'][:nt,:,:]
                eislts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/JRA55/inputs/eislts_*"+"*.nc")[0])['eislts'][:nt,:,:]
                Y_pi = netCDF4.Dataset(path_to_files+"data/obs_clouds/albcld_200003-201909.nc")['albcld'][:nt,lati,loni]
            else:
                hur700_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CMIP/inputs/hur700_Amon*"+model_list[modeli]+"*.nc")[0])['hur'][:nt,0,:,:]
                wap500_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CMIP/inputs/wap500_Amon*"+model_list[modeli]+"*.nc")[0])['wap'][:nt,0,:,:]
                utrh_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CMIP/inputs/utrh_Amon*"+model_list[modeli]+"*.nc")[0])['utrh'][:nt,:,:]
                ts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CMIP/inputs/ts_Amon*"+model_list[modeli]+"*.nc")[0])['ts'][:nt,:,:]
                eislts_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CMIP/inputs/eislts_Amon*"+model_list[modeli]+"*.nc")[0])['eislts'][:nt,:,:]
                Y_pi = netCDF4.Dataset(glob.glob(path_to_files+"data/CMIP/albcld_LW/albcld_Amon_*"+model_list[modeli]+"*.nc")[0])['albcld'][:nt,lati,loni]
            ### stack data so that the predictor boxes can be easily selected and the resulting coefficients indexed for further analysis
            hur700_pi_3 = np.dstack((hur700_pi,hur700_pi,hur700_pi))
            utrh_pi_3 = np.dstack((utrh_pi,utrh_pi,utrh_pi))
            ts_pi_3 = np.dstack((ts_pi,ts_pi,ts_pi))
            wap500_pi_3 = np.dstack((wap500_pi,wap500_pi,wap500_pi))
            eislts_pi_3 = np.dstack((eislts_pi,eislts_pi,eislts_pi))
            nt_min_pi = min(nt,len(ts_pi[:,0,0]),len(eislts_pi[:,0,0]),len(hur700_pi[:,0,0]),len(wap500_pi[:,0,0]),len(utrh_pi[:,0,0]),len(rsdt_pi[:]),len(Y_pi[:]))
            ### predictor number of latitude points eventually becomes smaller as one approaches the poles
            nrlat_local = hur700_pi_3[:nt_min_pi,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1].shape[1]
            nrlon_local = hur700_pi_3[:nt_min_pi,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1].shape[2]
#             print(nrlat_local,nrlon_local)
            nr_features = nrlat_local*nrlon_local*nr_planes
            ### collect predictor variables, then flatten array for input to Ridge regression object
            X_pi = np.stack((ts_pi_3[:nt_min_pi,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1],
                              eislts_pi_3[:nt_min_pi,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1],
                              hur700_pi_3[:nt_min_pi,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1],
                              wap500_pi_3[:nt_min_pi,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1],
                              utrh_pi_3[:nt_min_pi,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1]),axis=1)
            X_pi = X_pi.reshape((nt_min_pi,nr_features))
            ### define cross-validation method
            cv_obj = KFold(n_splits=5,shuffle=False)
            ### define CV/regression object
            regr_obj = GridSearchCV(Ridge(),parameters,cv=cv_obj,n_jobs=-1,refit=True)
            ### mask data according to solar zenith angle (see main paper)
            mask_train = np.argwhere(rsdt_pi[:] < 80.0)
            number_samples[modeli,lati,loni] = len(Y_pi[mask_train])
            ### Scale each predictor to zero mean und unit standard deviation
            ### (necessary for Ridge)
            scaler_model = StandardScaler().fit(X_pi[mask_train,:][:,0,:])
            X_pi_train_norm = scaler_model.transform(X_pi[mask_train,:][:,0,:])
            ### fit for specific target grid box and dataset
            regr_obj.fit(X_pi_train_norm[:,:],Y_pi[mask_train][:,0])
            ### store regression results
            dict_regr_results[model_list[modeli]].append(regr_obj)
            ### store best alpha
            alpha_list.append(regr_obj.best_estimator_.alpha)
            ### store Ridge coefficients
            coeffs[modeli,lati,loni,:,max(0,lati-latd):min(lati+latd+1,nr_lat),lon_sel-lond:lon_sel+lond+1] = regr_obj.best_estimator_.coef_.reshape(nr_planes,nrlat_local,nrlon_local)
    print(model_list[modeli],min(alpha_list),max(alpha_list))

### directory to store output
path_batch = './output/'
### save results
joblib.dump(dict_regr_results, path_batch+area_code+'.sav')
joblib.dump(coeffs, path_batch+'coeffs_'+area_code+'.sav')

### save results as netcdf4 files as well
file_to_write = netCDF4.Dataset(path_batch+'coeffs'+area_code+'.nc','w',dtype=np.float32,format='NETCDF4_CLASSIC')
model_dim = file_to_write.createDimension('model_nr',nr_models)
lat_out_dim = file_to_write.createDimension('lat_out',36)
lon_out_dim = file_to_write.createDimension('lon_out',72)
nr_var_dim = file_to_write.createDimension('var',nr_planes)
lat_in_dim = file_to_write.createDimension('lat_in',36)
lon_in_dim = file_to_write.createDimension('lon_in',216)
coeffs_var = file_to_write.createVariable('coeffs',np.float32,('model_nr','lat_out','lon_out','var', 'lat_in','lon_in'))
coeffs_var[:] = coeffs
file_to_write.close()

file_to_write = netCDF4.Dataset(path_batch+'number_samples'+area_code+'.nc','w',dtype=np.float32,format='NETCDF4_CLASSIC')
model_dim = file_to_write.createDimension('model_nr',nr_models)
        
lat_dim = file_to_write.createDimension('lat',36)
lon_dim = file_to_write.createDimension('lon',72)
Number_samples = file_to_write.createVariable('number_samples',np.float32,('model_nr','lat','lon'))
Number_samples[:] = number_samples
        
file_to_write.close()




