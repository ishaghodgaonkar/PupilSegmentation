# PupilSegmentation

This code attempts to segment pupils from an image using semantic segmentation using U-Net[1], and uses DeepVOG [2] to construct ellipses. 
Diameter measurements are the average of the major and minor axis. 
A demo can be run on the test set provided by [3] by running 'python3 eval.py' or on the video provided by [3] running 'python3 test.py'. 

Dependencies:
NumPy
torch
torchvision
segmentation_models_pytorch
matplotlib
PIL
cv2

References

[1] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241. Springer, 2015

[2] Yuk-Hoi Yiu, Moustafa Aboulatta, Theresa Raiser, Leoni Ophey, Virginia L. Flanagin, Peter zu Eulenburg, Seyed-Ahmad Ahmadi. DeepVOG: Open-source pupil segmentation and gaze estimation in neuroscience using deep learning. Journal of Neuroscience Methods, Volume 324, 2019, 108307, ISSN 0165-0270.
DeepVOG GitHub Repo: https://github.com/pydsgz/DeepVOG

[3] Pupil Segmentation Dataset located at: https://github.com/Gyoorey/PupilDataset
