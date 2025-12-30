import numpy as np

k = np.zeros(shape=(256,256,256))

center = (50,128,128)

for i in range(k.shape[0]):
    for j in range(k.shape[1]):
        for t in range(k.shape[2]):
            if(np.sqrt( pow(i - center[0],2) +  pow(j - center[1],2) + pow(t - center[2],2))) < 5:
                k[i,j,t] = 10
np.save("/home/lonqi/work/CT_Recon_UI/asset/model/ball/vol_gt.npy",arr=k)