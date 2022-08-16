# Duke Data+ research project

This repo was created for Summer 2022 Data+ research project on Climate change. The motivation is to minimize global warming in USA by applying machine learning approaches. Based on the current climate analysis, future wetland carbon emissions will be predicted in a warming climate, so we are building the models to measure the potential expected values. Previous researches found that the emissions from wetland ecosystems provide a potentially positive feedback to global climate warming. However, human factors have contributed to rapid decline of wetlands throughout the world making the prompt actions necessary regarding the climate change.
Previous traditional estimates of wetland carbon emissions were often obtained in a way that did not allow the researchers to fully understand the dynamics between environmental factors resulting in large uncertainties in the predictons. Machine learning is particularly promising in its ability to deduce
relationships between factors. This research wil better assess wetland carbon emissions over the entire Southeast and provide critical information on future carbon budgets on regional scales.
We comprehensively quantify the wetland carbon emissions in the entire Southeast USA using machine learning techniques and various climate datasets—including remote sensing data, climate observations, and hydrological model (PIHM-Wetland) outputs. The idea is to apply end-to-end data science techniques to establish the relationship between climate related variables and wetland carbon emissions at observational sites. The measurements of the carbon emissions from the Southeast US wetland ecosystems is created finally.  


## Data Sources

The datasets used in this research project have been mainly downloaded from the following sources:

[AmeriFlux](https://ameriflux.lbl.gov)

Also Groundwater depth measurements from SE
States and USGS Water Data were used in the data collection process.

The locations can be seen on the map:

<img width="372" alt="map" src="https://user-images.githubusercontent.com/53462948/178688240-772000f0-b457-4e1e-b076-8b64b44dc242.png">


## Data insights

The main dataset contains half-hourly data for different climate features that can help to predict NEE. We turned this data into daily data by keeping the days that had at least 40 valid observations. The additional datasets for Water table depth (WTD) based on the locations have also been incorporated. However, since we had a lot of important WTD data missing, we had to look for additional data sources and collected some more data from the above mentioned additional sources. We also combined additional features, the precipitations and sin/cos day of the year from the previously done works in this area.

Number of total processed observations: 19,243
Number of missing variables initially:

<img width="424" alt="missing values" src="https://user-images.githubusercontent.com/53462948/179900983-db755ab1-e773-4478-9051-5955eb2f6369.png">


## Data pipeline

![pipeline](https://user-images.githubusercontent.com/53462948/184851369-1b5458a7-0e69-4ad5-a479-5f70b4af6755.png)


- **Features**: Incoming shortwave radiation,  Air temperature,  Wind speed,  Soil temperature,  Vapor pressure deficit, Soil water content,  Water table depth
- **Target**: NEE (Net exchange of ecosystem) for carbon dioxide  


