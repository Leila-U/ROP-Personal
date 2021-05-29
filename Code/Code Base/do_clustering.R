#######################################################################
#name: do_clustering.R

#author: Damian M. Maddalena <damian@maddalena.org>
#
#description: clustering analysis for ecoregions
#
#notes: 
#
#######################################################################

##############
#set switches
##############


read.cluster <- 1

#normalize data frame
be.normal <- 1

#run clustering
make.nut.clusters <- 1
#this sets the k value list to process
cluster.range <- seq(5,30,5)
#cluster.range <- seq(5,25,5)

#####################
#set global variables
#####################

#no scientific notation
options(scipen=999)
wgs84 <-"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs "

#ifc colors
map.colors <- c("#89C5DA", "#DA5724", "#74D944", "#CE50CA", "#3F4921", "#C0717C", "#CBD588", "#5F7FC7", "#673770", "#D3D93E", "#38333E", "#508578", "#D7C1B1", "#689030", "#AD6F3B", "#CD9BCD", "#D14285", "#6DDE88", "#652926", "#7FDCC0", "#C84248", "#8569D5", "#5E738F", "#D1A33D", "#8A7C64", "#599861")

###############
#load libraries
###############
library(rgdal)
library(rgeos)
library(raster)
library(ggplot2)
library(broom)
library(dplyr)

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

#read in the clustering dataset
if (read.cluster == 1){
   c.df <- read.table(file.path(prodir,'clustering_dataset.txt'),header=TRUE,sep='|',fill=TRUE)
   
   #subset data only, no coordinates
   cd.df <- c.df[c(3:21)]
}

#normalize
if (be.normal == 1){
   #this page was used for code to convert centers from clustering output back to real data values
   ncd.df <- scale(cd.df,center=TRUE,scale=TRUE)
   scale.list <- list(scale = attr(ncd.df, "scaled:scale"), center = attr(ncd.df, "scaled:center"))
   
   ncd.df <- as.data.frame(ncd.df)
}

if (make.nut.clusters == 1){
   for (i in cluster.range){
      print(paste('Generating',i,'ecoregions.'))
      nut.clusters <- kmeans(ncd.df,i)
      cl.df <- data.frame(c.df[1:2],nut.clusters$cluster)
      #add names to data frame
      names(cl.df) <- c('lon','lat','cluster')
      #cl.df$cluster <- as.factor(paste('Region',cl.df$cluster))
      cl.df$cluster <- as.factor(cl.df$cluster)
     
      #aggregate the cluster means for each variable
      cl.means.df <- aggregate(cd.df, by=list(cluster=nut.clusters$cluster),mean)
      cl.means.df <- cl.means.df[,-1] <-round(cl.means.df[,-1],1)
 
      #write out the cluster table
      write.table(cl.df,file.path(tabdir,paste('clusters_k',i,'.csv',sep="")),sep="|",col.names=TRUE,row.names=FALSE)

      #write out the cluster means table
      reg.names <- as.data.frame(row.names(cl.means.df))
      names(reg.names) <- c('cluster')
      cl.means.df <- cbind(reg.names,cl.means.df)
      write.table(cl.means.df,file.path(tabdir,paste('clusters_means_k',i,'.csv',sep="")),sep="|",col.names=TRUE,row.names=FALSE)

      p <- ggplot(cl.df, aes(x=lon,y=lat)) + geom_raster(aes(fill=cluster)) + coord_fixed(1.3)
      #p <- p + geom_polygon(data=study.states.df,aes(x=long,y=lat,group=group),fill=NA,color='black') 
      #p <- p + geom_point(data = ws.df, aes(x=long, y=lat),size=1.5,color='black',pch=17)
      #p <- p + geom_point(data = f.cent.df, aes(x=x, y=y),size=1.5,color='yellow')
      p <- p + ggtitle(paste('Ecoregion Delineation: k=',i,sep="")) + labs(x='Longitude', y='Latituide')+ theme(plot.title = element_text(hjust = 0.5))
      #p <- p + theme(legend.position="none") + theme(legend.title = element_blank()) 
      p <- p + theme(legend.position="right") + theme(legend.title = element_blank()) 
      #p <- p + scale_fill_manual(values=map.colors)

      ggsave(file.path(mapdir,paste('map_ecoregions_k',i,'.png',sep="")),device='png',plot=p,width=11,height=8.5)
  }
}

