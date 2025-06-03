#########################################################################################
# File: main.py                                                                         #
# Author: Jonas Schulte-Sasse                                                           #
# Created: 2024-11-24                                                                   #
# Description: Flow direction prediction of an oil flow visualization with a CNN.       #
#########################################################################################

#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from outlier_algorithm import detection_outlier

#%%
def load_img(image, img_res, m_input):
    # Model's define pixel resolution.
    model_res = 23 # px/mm

    # Image to predict:
    img = tf.io.read_file(image)
    img = tf.image.decode_image(img, channels=1)

    # Resize image to match model resolution.
    img = tf.image.resize(img, 
                          (np.array(img.shape[:2]) * (model_res/img_res)).astype(int), 
                          method='lanczos5', antialias=False) 
    
    # Number of patches within image's height and width.
    p_h, p_w = np.floor_divide(img.shape[:2], m_input)

    # Get only full paches.
    img = img[:p_h*m_input, :p_w*m_input]
    
    # Normalize the image pixel values.
    img /= 255.

    # Get pachtes from image.
    pat = patch_img(img, m_input, p_w, p_h)

    return img, pat, p_w, p_h

def patch_img(img, p_s, p_w, p_h):
    # Tranform the image into a n-array of patches.
    patches = np.reshape(img, (p_h, p_s, p_w, p_s, -1))
    patches = np.transpose(patches, (0,2,1,3,4))
    patches = np.reshape(patches, (p_h*p_w, p_s, p_s, -1))
    return patches

def correction_outlier(y, radius, thr, eps, patience=10, max_iter=50):
    neigbor_mask = np.arange(-1, 1+1)[:,None].repeat(3,axis=1)

    ind_nocenter = np.arange(neigbor_mask.size)
    ind_nocenter = ind_nocenter[ind_nocenter != neigbor_mask.size//2]

    y = np.fmod(y, 2*np.pi)

    quota, k, Thr_i, eps_i = [0,1], 0, thr, eps
    while True:
        U,V = np.cos(y), np.sin(y)
        outliers = detection_outlier(U, V, Thr_i, radius, eps_i)

        quota[0] = outliers.sum()/outliers.size  

        if quota[0]<quota[1]:
            p = 0
        else:
            p += 1
            Thr_i *= 1.1

        print(f'{k}: quota: {1-quota[0]:%}, {p}')
        if (quota[0] <= 0.0001) or (p >= patience) or (k == max_iter): 
            break

        index = np.vstack(np.where(outliers==True)).T

        ind_neigh = np.vstack([neigbor_mask.flatten(),
                               neigbor_mask.T.flatten()]) + np.transpose(index[:,None],(0,2,1))

        corners = np.sum(index==0,axis=1) + np.sum(index==(U.shape[0]-1),axis=1)
        corners = corners*neigbor_mask.shape[0] - (corners==2)

        lx = np.zeros([len(corners)])
        for l in range(len(corners)):
            lx[l] = np.any(np.all(ind_neigh[l].T[:,None]==index,axis=-1),axis=-1).sum()
        x = np.argsort(neigbor_mask.size - corners - lx)[::-1]

        cond_neigh = (((neigbor_mask.size - corners - lx)/((neigbor_mask.size-1) - corners)) > .5)
        index = index[x][cond_neigh[x]]
        
        y_pad = np.pad(y, pad_width=1, mode='constant', constant_values=(np.nan))
        
        for row,col in index:
            neigh = np.vstack([(row+neigbor_mask).flatten(),
                               (col+neigbor_mask.T).flatten()]).T[ind_nocenter]

            mask = np.any(np.all(neigh[:, None]==index, axis=-1), axis=-1)
            realNeigh = y_pad[*(neigh[~mask] + 1).T]
            realNeigh = realNeigh[~np.isnan(realNeigh)]
            
            y[row,col] = np.mean(realNeigh)

        quota[1] = quota[0]
        k += 1
    
    return y, outliers

#%%
# Load TensorFlow model and get input size.
model = tf.keras.models.load_model(r'.\OilFlowCNN.keras', compile=False) # or r'.\OilFlowCNN.h5'
model_input =  model.input_shape[1]

# Image and its resolution variable.
img_path = 'image.png'
img_res = 10            # px/mm

# Get image, patches and number of patches in width and height.
img, patches, n_w, n_h = load_img(img_path, img_res, model_input)

# Prediction of flow directions and resize values in radians from normalized predicted values. 
# True values need to be rotated by 180°.
y_p = model.predict(patches, batch_size=3) * np.deg2rad(357) - np.pi

# Transformation to a 2D-array and values bigger than 2*np.pi ...
y_p = np.fmod(np.reshape(y_p, [n_h, n_w]), 2*np.pi)

# Allocation of variables to see the correction steps.
y_c1, y_c2 = np.zeros([*y_p.shape]), np.zeros([*y_p.shape])
#%%
# Definition for the outlier correction algorithm.
r = 2               # Nieghbour radius.
thr = 1             # Threshold value.
eps = .5            # Noise value in array.

# Detection of outliers.
U, V = np.cos(y_p), np.sin(y_p)
outliers = detection_outlier(U, V, thr, r, eps)

# Correction algorithm:
# Step 1: rotation by 180°.
y_c1[:,:] = y_p[:,:]
q, p = 0,0
while True:
    U_r, V_r = np.cos(y_c1), np.sin(y_c1)
    outliers_r = detection_outlier(U_r, V_r, thr, r, eps)
    y_c1[outliers_r] = np.fmod(y_c1[outliers_r] - np.pi, 2*np.pi) 

    if p == 5: 
        break
    elif outliers_r.sum() >= q: 
        p += 1
    
    q = outliers_r.sum()

# Step 2: correction of outliers with average value from neighbours.
y_c2[:,:] = y_c1[:,:]
y_c2, outliers_c = correction_outlier(y_c2, r,thr,eps, patience=10, max_iter=200)

#%%
# Variables needed to plot each arrow at the center of each patch.
Y, X = (np.indices(U.shape) + 1/2)*model_input

# Predicted values from model.
fig, ax = plt.subplots(1,1, dpi=600)
ax.imshow(img)
ax.quiver(X,Y, U/4, -V/4, outliers, angles='xy', pivot='mid', scale_units='xy',
                        cmap='bwr_r', edgecolor='k', scale=0.0015,
                        linewidth=.5, headwidth=3, headlength=3.5, 
                        headaxislength=3.1, width=0.0035)
ax.axis('off')
fig.tight_layout()

# Predicted values after correction step 1.
U_c1, V_c1 = np.cos(y_c1), np.sin(y_c1)
fig, ax = plt.subplots(1,1, dpi=600)
ax.imshow(img)
ax.quiver(X,Y, U_c1/4, -V_c1/4, outliers_r, angles='xy', pivot='mid', scale_units='xy',
                        cmap='bwr_r', edgecolor='k', scale=0.0015,
                        linewidth=.5, headwidth=3, headlength=3.5, 
                        headaxislength=3.1, width=0.0035)
ax.axis('off')
fig.tight_layout()

# Predicted values after correction step 2.
U_c2, V_c2 = np.cos(y_c2), np.sin(y_c2)
fig, ax = plt.subplots(1,1, dpi=600)
ax.imshow(img)
ax.quiver(X,Y, U_c2/4, -V_c2/4, outliers_c, angles='xy', pivot='mid', scale_units='xy',
                        cmap='bwr_r', edgecolor='k', scale=0.0015,
                        linewidth=.5, headwidth=3, headlength=3.5, 
                        headaxislength=3.1, width=0.0035)
ax.axis('off')
fig.tight_layout()