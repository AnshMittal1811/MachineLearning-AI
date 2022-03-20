# This is an example of how to load and display the functional (fMRI) and
# structural (sMRI) ICA maps, provided for the 2014 MLSP Competition, using
# R 3.1.0.
#
# It also includes an example of how to compute the correlation between the
# fMRI and sMRI maps, like was done in reference [2] below.
#
# This script requires the oro.nifti package to be installed.
# 
# The maps provided for the competition are a subset of the components
# discussed in the following publications:
# 
# 28 aggregate resting-state fMRI ICA maps:
# [1] E. Allen, et al, "A baseline for the multivariate comparison of
# resting state networks," Frontiers in Systems Neuroscience, vol. 5, p.
# 12, 2011.
# 
# 32 mean gray-matter concentration sMRI ICA maps:
# [2] Segall JM, et al (2012) Correspondence between structure and function
# in the human brain at rest. Front. Neuroinform. 6:10. doi: 10.3389/fninf.2012.00010 

##install.packages('oro.nifti') # You might have to run this line if oro.nifti is not installed
library(oro.nifti)

## Define function show_maps()
# show_maps() is a simple support function provided along with this script
# for the 2014 MLSP Competition.
show_maps <- function(D,ci){
  jet.colors <- colorRampPalette(c("white","cyan","black","red","yellow"))
  #"#00007F", "blue", "#007FFF", "cyan",
                                   #"black", "yellow", "#FF7F00", "red", "#7F0000"))
  idx = which(abs(D[,,,ci]) == max(abs(D[,,,ci])), arr.ind=TRUE)
  # Save the current graphical parameters defaults:
  old.par = par()
  # Change graphical parameters
  par(mfrow=c(1,3),oma=rep(0,4),mar=rep(1,4))
  x = 1:dim(D)[1]
  y = 1:dim(D)[2]
  z = 1:dim(D)[3]
  image(y,z,as.matrix(D[idx[1],,,ci]),zlim=max(abs(D[,,,ci]))*c(-1,1),col=jet.colors(256),
        asp=1,axes=F,ann=F)
  image(x,z,as.matrix(D[,idx[2],,ci]),zlim=max(abs(D[,,,ci]))*c(-1,1),col=jet.colors(256),
        asp=1,axes=F,ann=F,xlim=c(max(x),min(x)))
  image(x,y,as.matrix(D[,,idx[3],ci]),zlim=max(abs(D[,,,ci]))*c(-1,1),col=jet.colors(256),
        asp=1,axes=F,ann=F,xlim=c(max(x),min(x)))
  # Set graphical parameters back to the default:
  suppressWarnings(par(old.par))
}

## Load fMRI maps
# Load 28 rs-fMRI ICA maps: one 3D volume per map, all in a single 4D matrix
# Assumes 'rs_fMRI_ica_maps.nii' file is in the current working directory
# To change the current working directory use setwd()
Df = readNIfTI(file.path(getwd(),'rs_fMRI_ica_maps.nii'))

# Load fMRI component numbers (as defined in reference [1] above)
fMRI_comp_ind = as.vector(read.csv(file="comp_ind_fMRI.csv",head=TRUE,sep=","))

# Select component to display:
#fMRI_ci = 6 # numbers 1..28
fMRI_ci = which(fMRI_comp_ind == 24) # numbers as defined in [1] above

# Show cross-section views of selected map, around max value voxel
show_maps(Df,fMRI_ci)



## Load sMRI maps
# Load 32 sMRI ICA maps: one 3D volume per map, all in a single 4D matrix
# Assumes 'gm_sMRI_ica_maps.nii' file is in the current folder
# To change the current working directory use setwd()
Ds = readNIfTI(file.path(getwd(),'gm_sMRI_ica_maps.nii'))

# Load sMRI component numbers (as defined in reference [2] above)
sMRI_comp_ind = as.vector(read.csv(file="comp_ind_sMRI.csv",head=TRUE,sep=","))

# Select component to display:
#sMRI_ci = 9 # numbers 1..32
sMRI_ci = which(sMRI_comp_ind == 10) # numbers as defined in [2] above

# Show cross-section views of selected map, around max value voxel
show_maps(Ds,sMRI_ci)

## Correlation between fMRI and sMRI maps

# A simple way to find relationship between fMRI and sMRI features is by
# measuring linear correlation between the provided ICA maps from each
# modality.

# Reshape ICA maps to columns of a table:
Dftab = as.matrix(Df)
sz = dim(Df);
dim(Dftab) = c(prod(sz[1:3]),sz[4])
Dstab = as.vector(Ds)
sz = dim(Ds);
dim(Dstab) = c(prod(sz[1:3]),sz[4])

# Create a mask to remove lines from the table which correspond to
# out-of-brain locations.
# This is useful for the correlation computation.

# Out-of-brain locations are those with zero-valued values:
msk = rowSums(abs(Dftab)) != 0

# Correlation between fMRI and sMRI maps:
cor_fs = cor(Dftab[msk,],Dstab[msk,]) # A 28x32 correlation matrix

# View correlations in an image:
sz = dim(cor_fs)
jet.colors <- colorRampPalette(c("white","cyan","black","red","yellow"))
image(1:sz[2], 1:sz[1], t(cor_fs), zlim=.5*c(-1,1), col=jet.colors(256),
      asp=1, xlab='sMRI map', ylab= 'fMRI map', axes=F)
axis(3, at=1:sz[2], labels=sMRI_comp_ind[,1], cex.axis=.4)
axis(2, at=1:sz[1], labels=fMRI_comp_ind[,1], cex.axis=.4, las=2)