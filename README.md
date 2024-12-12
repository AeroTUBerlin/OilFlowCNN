# Oil-flow Convolutional Neural Network
This site supports the paper published [paper], showcasing a Convolutional Neural Network (CNN) trained to predict the flow direction from an oil flow visualizations. 

## Overview
The [code](https://github.com/aero24xx/OilFlowCNN/blob/main/main.py) provided in this repository shows an example to predict the flow direction.

## Example visualization:
The image shows an oil-flow visualization over a backward facing ramp in gray scales.

![Oil flow visualization](https://github.com/aero24xx/OilFlowCNN/blob/main/image.png "Backward facing ramp")

### Output prediction:
The following image shows the predicted direction (arrows) obtained by the CNN. The color of each arrows depends on the outlier algorithm output; red is not calculated as an outlier, blue is an outlier.  

![Output precition](https://github.com/aero24xx/OilFlowCNN/blob/main/output_0.png "Backward facing ramp")

### Step 1 - corrected values:
The first step to correct the field is made by rotating each outlier by 180° and calculating its neighbourhood to check if the change has corrected the outlier.

![Oil flow visualization](https://github.com/aero24xx/OilFlowCNN/blob/main/output_1.png "Backward facing ramp")

### Step 2 - corrected values:
The second step takes into account the neighbourhood of each outlier to correct the direction. The new direction is an average of its neighbours.
![Oil flow visualization](https://github.com/aero24xx/OilFlowCNN/blob/main/output_2.png "Backward facing ramp")

## References
This project uses an implementation of the algorithm described in:

Jerry Westerweel and Fulvio Scarano, "Universal outlier detection for PIV data", Experiments in Fluids, 2005
DOI: [https://doi.org/10.1007/s00348-005-0016-6 ](https://doi.org/10.1007/s00348-005-0016-6 )