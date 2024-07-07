# swellSWOT
Analysis tools for looking at long swells in SWOT "unsmoothed" data. Some example illustrations can be found in these videos: 
https://www.youtube.com/playlist?list=PLm1sPhvTQhOxMjglmUjAyB28bSM45C2eQ 

Main contributors: 
Fabrice Ardhuin and Marine De Carlo

These tools have been used for the following papers: 
http://dx.doi.org/10.13140/RG.2.2.32296.17925   Phase-resolved swells across ocean basins in SWOT altimetry data: revealing centimeter-scale wave heights including coastal reflection

The most simple Toolbox is "swot_swell_fig1.ipynb", it takes a sample of SWOT "unsmoothed" data and computes a wave spectrum from it. 
If you have co-located wave model output, it will also
compare model and SWOT data.


For some of the other notebooks, you may need to download these data sets: 

https://www.seanoe.org/data/00885/99739/  Spotter buoy data for storm "Rosemary"

https://www.seanoe.org/data/00886/99783/  (this is our model hindcast only for June 2023, the full year 2023 and
more is available on datarmor) 

(or if you are working on our datarmor cluster, here they are: ) 
/home/datawork-WW3/PROJECT/SWOT

## Installation 
git clone --quiet https://github.com/ardhuin/swellSWOT
