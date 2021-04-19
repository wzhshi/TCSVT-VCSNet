This is the test code for our TCSVT paper "Video Compressed Sensing Using a Convolutional Neural Network".

@article{Shi2021Video,
  author={Wuzhen, Shi and Shaohui, Liu and Feng, Jiang and Debin, Zhao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Video Compressed Sensing Using a Convolutional Neural Network}, 
  year={2021},
  volume={31},
  number={2},
  pages={425-438},
  doi={10.1109/TCSVT.2020.2978703}}


To use this code, do the following:
1, Install the MatconvNet https://www.vlfeat.org/matconvnet/
2, Copy the folder "myLayers" to the MatConvNet installation directory.
3, add the code "addpath(fullfile(root, 'myLayers')) ;" in the file "vl_setupnn.m".
4, Copy these functions in the folder "Dagnn" into the folder "+dagnn" of the MatConvNet.
5, Run the file test_VCSNet.m