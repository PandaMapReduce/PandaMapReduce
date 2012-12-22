Source Code Repository for Panda Framework on GPUs/CPUs

==============================

6/2012 - 12/2012
Hui Li
Department of Computer Science
Indiana University
Bloomington, Indiana
==============================

=== About ===

Heterogeneous parallel system with multi processors and accelerators are becoming ubiquitous due to better costperformance
and energy-efficiency. These heterogeneous processor architectures have different instruction sets and are optimized
for either task-latency or throughput purposes. Challenges occur in regard to programmability and performance when coprocessing SPMD computations on heterogeneous architectures simultaneously. In order to meet these challenges, we designed and implemented a runtime system with MapReduce interface that used for co-processing SPMD job on GPUs and CPUs simultaneously. We are proposing a hybrid MapReduce programming interface for the developer and leverage the two-level scheduling approach in order to efficiently schedule tasks with heterogeneous granularity on the GPUs and CPUs. Experimental results of Cmeans clustering, matrix multiplication and word count indicate that using all CPU cores increase the GPU performance by 11.5%, 5.1%, and 41.9% respectively. 

=== Software Installation ===

1) Install NVIDIA CUDA Driver (available from http://www.nvidia.com/object/cuda_get.html)
2) Install NVIDIA CUDA Toolkit (available from http://www.nvidia.com/object/cuda_get.html)
3) Install NVIDIA CUDA SDK (available from http://www.nvidia.com/object/cuda_get.html)
4) Checkout source code from Github repository into your Linux machine


=== Compiling the Panda Code (Linux) ===

    $ cd ~/Panda_WordCount
    $ make

=== Compiling the CUDA Code (Windows) ==

We will provide the Windows vesions soon. Windows users should be able to use the included ".sln" Visual Studio solution file. Ensure that the $(CUDA_INC_PATH) and $(CUDA_LIB_PATH) environment variables are set properly. It was developed and tested on Windows 7 with Visual Studio 2008 Professional.

======== Running the Code ==============

Usage: 
    ./panda_cmeans

=============Version Info ==============

Latest workable version 0.32

  1) run on multiple gpus
  2) run on gpus and cpus simultaneously
  3) region-based memory management
  4) local combiner
  5) iterative mapreduce support

Applications:

  1) word count
  2) matrix multiplication
  3) cmeans


Developing version 0.4

  1) gpu_host_map() function
  2) multiple gpu context management


============== License ===============

Copyright (c) 2012, Indiana University

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: 1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
