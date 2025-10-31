# swellSWOT
Analysis tools for looking at long swells in SWOT "unsmoothed" data and working with SWOT-derived swell spectra. Some example illustrations can be found in these videos: 
https://www.youtube.com/playlist?list=PLm1sPhvTQhOxMjglmUjAyB28bSM45C2eQ 

Main contributors: 
Fabrice Ardhuin, Marine De Carlo, Taina Postec, Beatriz Molero, Adrien Nigou 

These tools have been used for the following papers: 
http://dx.doi.org/10.13140/RG.2.2.32296.17925   Phase-resolved swells across ocean basins in SWOT altimetry data: revealing centimeter-scale wave heights including coastal reflection
https://www.pnas.org/doi/10.1073/pnas.2513381122  Sizing the largest ocean waves using the SWOT mission



The most simple Toolbox (using surface elevation data) is "swot_swell_fig1.ipynb", it takes a sample of SWOT "unsmoothed" data and computes a wave spectrum from it. 
If you have co-located wave model output, it will also
compare model and SWOT data.

## Working with SWOT spectra you can: 
- download the spectra from AVISO (August 2023 to June 2024): https://www.aviso.altimetry.fr/en/data/products/windwave-products/swot-karin-level-3-wind-wave-product.html
- compute your own spectra from SWOT L2 or L3 surface elevation maps: 
use the notebook swot_compute_spectra_light_multi_tracks.ipynb
Note that this produces very similar files ... but not exactly the same. More later here about differences. 

### basic plotting of SWOT spectra (and SSH map if you have it)
- plot_one_spectrum.ipynb : this notebook just plots one spectrum 
- L3_fit_one_track_LandH.ipynb : plots parameters for an entire track but also includes 
an interactive plot of single spectra (using a widget slider)  

### spectral partitionning using watershed method
- plot_watershed.ipynb : this works with version 2 of the CNES L3 product, but also with version 2.1_light (with X-spectra phase included): here are examples here [https://drive.google.com/drive/folders/1Eu_KRgLw6uHMKNFgUUJ6n6w5gY_CMNVB?usp=drive_link](https://drive.google.com/drive/folders/1SSUNcgist3ZEP2cJi6p_AKEkf426LwFM?usp=drive_link)

For some of the other notebooks, you may need to download these data sets: 

https://www.seanoe.org/data/00885/99739/  Spotter buoy data for storm "Rosemary"

https://www.seanoe.org/data/00886/99783/  (this is our model hindcast only for June 2023, the full year 2023 and
more is available on datarmor) 

(or if you are working on our datarmor cluster, here they are: ) 
/home/datawork-WW3/PROJECT/SWOT

## Installation 
git clone --quiet https://github.com/ardhuin/swellSWOT
