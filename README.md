# cuckoostash
Implementation of Cuckoo Hashing with Stash for CPU/GPU

[![Build Status](https://travis-ci.com/mattpaletta/cuckoostash.svg?branch=master)](https://travis-ci.com/mattpaletta/cuckoostash)

### Original sources
* [Original Paper](https://www.cs.bgu.ac.il/~asharf/Projects/RealTimeParallelHashingontheGPU.pdf)
* [Dissertation](http://idav.ucdavis.edu/~dfalcant//downloads/dissertation.pdf)

### My Implementation
This implementation is taken from the [dissertation](http://idav.ucdavis.edu/~dfalcant//downloads/dissertation.pdf). (Section 6.1)
The default values used are as close as possible to the paper.

The original algorithm described in the paper is implemented in CUDA. My CUDA implementation has not been tested.  I have a parallel CPU implementation (in numpy), and am working on an OpenCL version (CUDA kernels translated to OpenCL and executed in OpenCL).  For more information on this process, see: [mattpaletta/pycoriander](https://github.com/mattpaletta/pycoriander) for my python bindings, or the original at: [coriander](https://github.com/hughperkins/coriander.git)
