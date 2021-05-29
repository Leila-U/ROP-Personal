#
#####################################################################
#name: build_regions_dataset.R

#author: Damian M. Maddalena <damian@maddalena.org>
#
#description: aggregate the data need for bioclimactic region calculations
#
#notes: 
#
#######################################################################



###############
#load libraries
###############
#library(FedData)
library(rgdal)
library(raster)

##############
#set switches
##############

#polys
read.polys <- 1

#elevation data
get.elev <- 0

#bioclim
read.bio <- 1

#clip brick
slice.brick <- 1

#####################
#set global variables
#####################

#tmp location
tout <- '/mnt/backforty/'

#get og raster outputdir
otd <- dirname(rasterTmpFile()) 

#set the tmp output dir for raster operations
rasterOptions(tmpdir=otd)

################
#build workspace
################

#define variables for workspace locations
rootdir <- getwd()
outdir <- file.path(rootdir,"output")
mapdir <- file.path(outdir,'maps')
figdir <- file.path(outdir,'figures')
tabdir <- file.path(outdir,'tables')
spacedir <- file.path(outdir,'geospatial')
datadir <- file.path(rootdir,'data')
rawdir <- file.path(datadir,'raw')
rawtabdir <- file.path(rawdir,'tables')
rawgeodir <- file.path(rawdir,'geospatial')
prodir <- file.path(datadir,'processed')
protabdir <- file.path(datadir,'processed')

#create workspace dirs if they do not exist

#list of workspaces
workspacelist <- c(rootdir,outdir,figdir,tabdir,mapdir,spacedir,rawdir, rawgeodir,rawtabdir,prodir,protabdir)
#loop through each location in the list, creating them if they do no exist
for (w in workspacelist){
   dir.create(w, showWarnings = FALSE, recursive=TRUE)
}


if (read.polys == 1){
   print('Reading study area poly.')
   study.polys <- 'washington_wgs84'
   study.area <- readOGR(rawgeodir,study.polys)
}

if (get.elev == 1){
   print('Downloading NED data.')
   study.elev <- get_ned(template=study.area,label='tri_state',res=1,raw.dir=rawgeodir,extraction.dir=rawgeodir)
}

#open the bioclim data variables
if (read.bio == 1){
   print('Reading bioclimatic data and creating raster stack')    
   raster.list <- list.files(path=rawgeodir, pattern ="wc2.0_bio_30s_[0-9][0-9].tif", full.names=TRUE)
   b.stack <- stack(raster.list)
}

if (slice.brick == 1){
   print('Extracting study area from raster stack.')
   sb.stack <- mask(b.stack,study.area)
   sb.stack <- crop(sb.stack,extent(study.area))
   sb.df <- as.data.frame(b.stack, xy=TRUE,na.rm=TRUE)

}
