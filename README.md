# DASClustering
Learnable unsupervised neural network using curvelet transform for features extraction.

This code is inspired by a scattering network written by Leonard Seydoux (https://github.com/leonard-seydoux/scatnet/tree/master/scatnet). It accompanys the paper 'Clustering distributed acoustic sensing signals via curvelet transform and unsupervised deep learning' written by Bolin Li, Sjoerd de Ridder and Andy Nowacki. The package was built upon Tensorflow 2.15.
Please note that the reader needs to copy the files 'fdct_wrapping.m', 'fdct_wrapping_window.m', and 'ifdct_wrapping.m' from https://curvelet.org/software.php into the 'curvenet' directory for the code working correctly. 
