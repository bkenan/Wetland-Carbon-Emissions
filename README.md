# Duke Data+ research project

This repo was created for Summer 2022 Data+ research project. The motivation is to minimize global warming in USA by applying machine learning. Based on the current climate analysis, future wetland carbon emissions will be predicted in a warming climate, so we are building the models to measure the potential expected values. This research will better assess wetland carbon emissions over the entire Southeast and provide critical information on future carbon budgets on regional scales.
We comprehensively quantify the wetland carbon emissions in the entire Southeast USA using machine learning techniques and various climate datasets—including remote sensing data, climate observations, and hydrological model (PIHM-Wetland) outputs. The idea is to apply end-to-end data science techniques to establish the relationship between hydroclimatological variables and wetland carbon emissions at observational sites. Spatial distributed carbon emissions from the entire SE US wetland ecosystems is created finally.  


## Data Sources

The datasets used in this research project have been downloaded from the following sources:

[AmeriFlux](https://ameriflux.lbl.gov)

The locations can be seen on the map:

<img width="372" alt="map" src="https://user-images.githubusercontent.com/53462948/178688240-772000f0-b457-4e1e-b076-8b64b44dc242.png">


## Data insights

The main dataset "summary_wetland_gapfilled" contains half-hourly data for different climate features that can help to predict NEE. We turned this data into daily data by keeping the days that had at least 40 valid observations. The additional datasets for WTD based on the locations have also been incorporated. 

Features: Incoming shortwave radiation,  Air temperature,  Wind speed,  Soil temperature,  Vapor pressure deficit,  Carbon dioxide concentration in atmosphere,  Soil water content,  Water table depth  
Target: NEE (Net exchange of ecosystem) for carbon dioxide  

Number of missing variables initially:

<img width="263" alt="image" src="https://user-images.githubusercontent.com/53462948/178687316-9358cce3-2d97-43e5-bff4-a2970f3d465b.png">

