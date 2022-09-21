# Data Preprocessing

### Data 
The dataset used by this project is collected using an [App for Google Tango Device](https://drive.google.com/file/d/1xJHZ_O-uDSJdESJhZ3Kpy86kWaGX9K2g/view) and optionally, app for Android Devices from [RoNIN](https://github.com/Sachini/ronin) or [Fusion-DHL](https://github.com/Sachini/Fusion-DHL), and pre_processed to the data format specified [here](https://www.dropbox.com/s/8m0x0yvhtbt86q1/README.txt?dl=0) 
The code for pre-processing raw data from apps are not included in this repository.

You can download the NILoc dataset from our [project website](https://sachini.github.io/niloc) or [HERE](https://www.dropbox.com/scl/fo/uux0twqk7gsgwdpljkahd/h?dl=0&rlkey=0g8qi66jsl14ffbx6r7nfn3rx).

The data pre-processing is two-fold:
1. [Real data](#1-real-data) : pre-process by performing distance based sampling of relative velocity from IMU
2. [Synthetic data](#2-synthetic-data) \[Optional]: generate synthetic trajectory data and perform similar pre-processing

## 1. Real data
This outlines the data pre-procecssing for real world dataset where we have IMU signals, and ground-truth trajectories (aligned across sequences to a common coordinate frame).

We use a data-collection app (while skipping the pre-calibration steps) and pre-processing procedure similar to [RoNIN](https://github.com/Sachini/ronin). Inertial tracking trajectory is computed using RoNIN ResNet checkpoint and added to the data file as `computed/ronin` 
(This can be in any *arbitary* gravity aligned reference coordinate frame). We use Tango Are Description (ADF) file to align ground-truth pose to a common coordinate frame, and the trajectory position in horizontal plane (2D) is saved as `computed/aligned_pos`.

Our dataset can be downloaded from [Dropbox](https://www.dropbox.com/scl/fo/uux0twqk7gsgwdpljkahd/h?dl=0&rlkey=0g8qi66jsl14ffbx6r7nfn3rx).

### 1.1 Generate occupancy map

If a floorplan is unavailble,  we generate an occupancy map from ground-truth trajectories
```
python real_data/map_creation.py <data folder containing hdf5 files> --map_dpi <resolution - pixels per meter>
```
The result `floorplan.png` will be saved in the data folder.

### 1.2 Flood-fill

Flood-filled map is used in synthetic data generation (Supplementary Sec 1.1 optimizing with floorplan)
```
python real_data/flood_fill.py <Path to occupancy map or floorplan image>
```

### 1.3 Distance based sampling
We perform distance-based sampling on input trajectories, selecting one sample for each `1/<resolution>` meters travelled.

```
python preprocess/real_data/distance_sample.py --data_dir <data folder containing hdf5 files> --map_dpi <resolution - pixels per meter> --out_dir <folder to save results>
```
The program outputs a txt file per trajectory containing following columns:
 - time (seconds)
 - trajectory from ronin x, y (pixels) 
 - ground-truth trajectory x,y (pixels)

## 2. Synthetic data

We generate synthetic data using A* algorithm and smooth the results using B-spline approximations. Complete steps 1.1 and 1.2 above, to get the occupancy maps.

The configurations for data generation is in `config/synthetic_data.yaml`
```
python preprocess/gen_synthic_data.py
```
The program outputs a txt file similar to 1.3. where timestamp is the frame number and trajectory from ronin is the smoothed synthetic trajectory.

## 3. Configuration setup

Configuration includes dataset and floorplan image paths for each building X:
    1. set path to relevant `floorplan.png` in `niloc/config/grid/<X>.yaml`
    2. The train/ validation/ test split for real IMU data are provided with the datasets. Set data folder and file list path in `niloc/config/dataset/<X>.yaml`  
    3. [Optional] You can divide generated synthetic data imu train/ validation sets and merge with provided IMU data lists for pretraining. Create a new configuration file `niloc/config/dataset/<X>_syn.yaml` for pretraining.   