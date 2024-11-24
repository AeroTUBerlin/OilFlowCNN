#####################################################################################
# This implementation is based on the algorithm described in:                       #
# Jerry Westerweel and Fulvio Scarano, "Universal outlier detection for PIV data",  #
# Experiments in Fluids, 2005                                                       #
# DOI: https://doi.org/10.1007/s00348-005-0016-6                                    #
#####################################################################################

import numpy as np

def detection_outlier(U,V, thr,radius,eps):    
    J, I = U.shape

    def Normfluct(AngleComp):
        MedianRes = np.zeros([J,I])
        normfluct = np.zeros([J,I])

        AngleComp = np.pad(AngleComp, pad_width=radius, mode='constant', constant_values=(np.nan))
        
        for ii, i in enumerate(np.arange( radius, I-radius+2)):
            for jj, j in enumerate(np.arange(radius, J-radius+2)):
                Neigh = AngleComp[j-radius:j+radius+1, 
                                  i-radius:i+radius+1]

                NeighCol = Neigh.flatten()[:,None]
                NeighCol2 = np.vstack([NeighCol[:(2*radius+1)*radius+radius],
                                       NeighCol[(2*radius+1)*radius+radius+1:]])
                
                NeighCol2 = NeighCol2[~np.isnan(NeighCol2)]

                Median = np.median(NeighCol2)
                Fluct= AngleComp[j,i] - Median
                
                Res = NeighCol2 - Median
                MedianRes = np.median(np.abs(Res))
                normfluct[jj,ii] = np.abs(Fluct/(MedianRes+eps))
        return normfluct
    
    return np.sqrt(Normfluct(U)**2 + Normfluct(V)**2) > thr 