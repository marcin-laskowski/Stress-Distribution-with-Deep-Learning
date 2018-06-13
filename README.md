# Stress Distribution with Deep Learning
A deep learning approach to estimate stress distribution: a fast and accurate surrogate of finite-element analysis


### Note
The files contain code and data associated with the paper titled "A Deep Learning Approach to Estimate Stress Distribution: A Fast and Accurate Surrogate of Finite Element Analysis". The paper is authored by Liang Liang, Minliang Liu, Caitlin Martin, and Wei Sun, and published at Journal of The Royal Society Interface, 2018.

The following repository contains further investigation of the method used in the mentioned above paper.

Copyright (c) 2018 by Biologically Inspired Computer Vision Group at Imperial College London. All rights reserved.


### Files
1. Data: ShapeData.mat, StressData.mat
2. Code of DL-model: DLStress.py, im2patch.m, UnsupervisedLearning.m,
3. Code for visualization: ReadMeshFromVTKFile.m, ReadPolygonMeshFromVTKFile.m, WritePolygonMeshAsVTKFile.m, Visualization.m
4. Template meshes for visualization: TemplateMesh3D.vtk, TemplateMesh2D.vtk


### System Requirement
- OS: Windows (64bit) 7 or 10
- Hardware: Intel quad-core CPU, 32G RAM


### Software Requirement
- **Anaconda**: https://www.anaconda.com/download/; select the python 3.6 version
- **Keras 2.0.4**: https://github.com/fchollet/keras. Keras can be install from Anaconda Cloud: https://anaconda.org/anaconda/keras
- **Tensorflow 1.1.0**: https://www.tensorflow.org/. Tensorflow CPU version can be installed from Anaconda Cloud: https://anaconda.org/conda-forge/tensorflow
- **Matlab** (at least 2016b): https://www.mathworks.com/products/matlab.html
- **MatConvNet**: http://www.vlfeat.org/matconvnet/; version 1.0-beta24 (backward compatibility not guaranteed)
- **Paraview**: https://www.paraview.org/download/; https://www.paraview.org/paraview-guide/
- **Spyder**: https://spyder-ide.github.io
- **Python 3.5**


### Installation guide
1. Install Matlab
2. Install MatConvNet
3. Install Anaconda 3.6
4. Make downgrade to python version 3.5 using following formula
```
conda install python=3.5
```
5. Install Tensorflow in Anaconda
```
conda create -n tensorflow pip python=3.5
activate tensorflow
pip install --ignore-installed --upgrade tensorflow
```
6. Install keras and then change the keras_backend to Tensorflow (in the path: /home/user/anaconda2/envs/tfpy36/etc/conda/activate.d/keras_activate.sh)
```
conda install -c conda-forge keras
```
7. Install Spyder in Anaconda: https://anaconda.org/conda-forge/spyder
8. Setup Matlab engine for python
```
c:\matlab\R2017a\extern\engines\python
python setup.py install
```

### Running procedure
1. Activate the anaconda environment in a cmd window, and type spyder. Then you should see something like
this. Spyder is a Python IDE. The current directory of Spyder is shown on top right. Open DLStress.py in Spyder, and run the code. You need to change the current directory of Spyder so that it contains DLStress.py. Change the path of MatConvnet in UnsupervisedLearning.m
2. Create folder 'result'
3. Once you save the result to mat files, open Visualization.m, and then convert the result to vtk files.
4. Open the vtk files in Paraview. You will see the ground-truth and predicted stress fields on 2D/3D meshes.
