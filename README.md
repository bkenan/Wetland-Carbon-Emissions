# Duke Data+ research project

This repo was created for Summer 2022 Data+ research project on Climate change. The motivation is to minimize global warming in USA by applying machine learning approaches. Based on the current climate analysis, future wetland carbon emissions will be predicted in a warming climate, so we are building the models to measure the potential expected values. Previous researches found that the emissions from wetland ecosystems provide a potentially positive feedback to global climate warming. However, human factors have contributed to rapid decline of wetlands throughout the world making the prompt actions necessary regarding the climate change.
Previous traditional estimates of wetland carbon emissions were often obtained in a way that did not allow the researchers to fully understand the dynamics between environmental factors resulting in large uncertainties in the predictons. Machine learning is particularly promising in its ability to deduce
relationships between factors. This research wil better assess wetland carbon emissions over the entire Southeast and provide critical information on future carbon budgets on regional scales.
We comprehensively quantify the wetland carbon emissions in the entire Southeast USA using machine learning techniques and various climate datasets—including remote sensing data, climate observations, and hydrological model (PIHM-Wetland) outputs. The objective is to apply end-to-end data science techniques to establish the relationship between climate related variables and wetland carbon emissions at observational sites, and hence build a dynamic tool for thge measurements of the carbon emissions from the Southeast US wetland ecosystems that can provide the future research work with the required estimations and data.  


## Data Sources

The datasets used in this research project have been mainly downloaded from the following sources:

[AmeriFlux](https://ameriflux.lbl.gov)

Also Groundwater depth measurements from SE
States and USGS Water Data were used in the data collection process.

The locations can be seen on the map:

<img width="800" height="450" alt="map" src="https://user-images.githubusercontent.com/53462948/178688240-772000f0-b457-4e1e-b076-8b64b44dc242.png">


## Data insights

The main dataset contains half-hourly data for different climate features that can help to predict NEE. We turned this data into daily data by keeping the days that had at least 40 valid observations. The additional datasets for Water table depth (WTD) based on the locations have also been incorporated. However, since we had a lot of important WTD data missing, we had to look for additional data sources and collected some more data from the above mentioned additional sources. We also combined additional features, the precipitations and sin/cos day of the year from the previously done works in this area.

Number of total processed observations: 19,243


## Data pipeline

![pipeline](https://user-images.githubusercontent.com/53462948/184851369-1b5458a7-0e69-4ad5-a479-5f70b4af6755.png)

It turns out that the final features and target variables are the following after going through the pipeline.

- **Features**: Incoming shortwave radiation,  air temperature,  wind speed,  soil temperature,  vapor pressure deficit, soil water content,  water table depth
- **Target**: NEE (Net exchange of ecosystem) for carbon dioxide  

The top 3 useful features are the incoming shortwave radiation, locations and soil water content based on the feature importance analysis.

## Machine learning pipeline

Overall workflow for the modeling:

![ml](https://user-images.githubusercontent.com/53462948/184878026-89018867-f872-465f-adca-07e4fb37a8fd.png)

Final model's architecture which is a meta-learning algorithm. The individual estimators provide outputs that are used as inputs by the final estimator to provide the outputs:

![ensemble](https://user-images.githubusercontent.com/53462948/184873641-740d6819-0e80-4371-8ad9-531091a77d5f.png)

## Functionality

We have used HTML, CSS and JavaScript in the front-end for making this tool user-friendly. This is the home page:

<img width="1470" alt="home" src="https://user-images.githubusercontent.com/53462948/184875363-9980ff3e-f744-4764-82ee-37901963f0e6.png">

In a nutshell, users can upload their csv/excel files containing the required climate data for the specific sites with the dates accordingly, and get the complementary data including the NEE immediately. Moreover, the real-time map is also provided in the separate window allowing the scientists to visualise the NEE change in the regions.

Screenshot from a portion of the map in action:
 
<img width="1312" alt="map" src="https://user-images.githubusercontent.com/53462948/184878406-9868cbd5-299c-42c4-9944-5eaef8b58a6a.png">


## Demo

<img src="https://user-images.githubusercontent.com/53462948/184876672-a4467b16-64c9-43c1-8f1d-15f5d62dfe64.gif" width="800" height="450"/>

