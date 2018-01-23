# Remarks

Please note that I haven't worked on this project in quite a while - I will try pushing fixes at some point but for now the code does not run as is in Xcode 9 / iOS 11. Modifications should not be too difficult, refer to the issues for problems that were solved already and feel free to open a PR anytime!

# iOS-Metal-Haarcascade - Updated

This is a small example I've put together to demonstrate object detection via Haar Classifiers on iOS using the Metal API. This is the updated version, for the old version please check out the respective branch. I've tried to optimize a lot of steps since the last version.

With this version, the **full cascade detection pipeline resides on the GPU**, including the grouping of rectangles. Part of the code and a lot of ideas are based on the OpenCV Cuda Shaders, so make sure to check out the cuda in the cudalegacy folder, which containers the shader files for haar detection.

I've converted the OpenCV HaarCascade File haarcascade_frontalface_default to JSON just because its easier to parse than XML, so theoretically other files should be supported as well. Currently, only one haar feature per classifier is supported, so if you try to convert other xml files and parse them, you'll receive an exception. 

The example I've used is for face detection, hence the naming scheme in the project. I'm using triple buffering and a MTLView in which I then render. The three main kernels are

* Pixel Parallel Processing from Texture
* Pixel Parallel Processing from Buffer
* Stage Parallel Processing from Buffer

In the pixel parallel processing, each thread in the compute kernel basically computes multiple stages for its position, while in the stage parallel processing, multiple threads are used for computing the results of a single stage. The variance map is computed within the first pixel-parallel computing stage. 

I've also managed to put the full grouping stage onto the GPU using a number of kernels, and it seems to perform really well (< 1 ms for a maximum of 1500 found rectangles). I'm using stream compaction instead of atomic operations as in the last version, and this really boosted performance. 

To sum up: the full detection pipeline is done on a single MTLCommandBuffer instance, without the need to copy results back to the CPU (i.e. for determining the number of found rectangles for grouping). This is achieved using `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`, where a buffer can be used as input argument to dispatch the number of threadblocks per grid - and it turns out calling that with buffer containing zero costs no performance. 

I've created a few wrapper classes for `MetalComputePipeline`, which basically stores instructions in advance and thus makes the code a little bit cleaner.

Note that I'm currently using the original method by Viola and Jones for grouping, i.e. overlapping rectangles are considered neighbors and are thus averaged for the final rectangle - which means that if two different faces are close together, the current version will probably no be able to differentiate due to overlapping regions. 

Performance in Release mode is pretty stable at 30 FPS on an iPhone 7, with the current settings (720x1280 input, processing at half the initial resolution, minimum rectangle size 360, maximum rectangle size 720, scale factor 1.2). I've also written some pretty messy unit tests, but they do their job :)

Some of the code is still a mess and a lot of parts are specifically written for the included json file (like how the number of stages are split, etc.), so beware when using this for other haarcascade classifier files.

Even though tests showed correct detection of a single face/two faces, I provide no guarantees for general correct functionality. Feel free to use this in your own projects. 
