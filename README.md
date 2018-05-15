# PatchMatch-Implementation
This is my implementation of the patchmatch algorithm. The patchmatch algorithm takes 2 arguments, a source and a target image. The purpose of this implementation is to recreate the target image by using pixels in the source image. In order to do this, I created a Nearest-Neighbour Field (NNF) to store the mappings. To find the best matching pixel, the algorithm iterates through every pixel and performs propagation twice and a random search phase.

The part of this project that I contributed to was the all of the source code in main.cpp. The cmake and additional resources were provided by my tutor/lecturer.
