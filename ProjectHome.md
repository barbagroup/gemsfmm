## GemsFMM — GPU implementation of the treecode/fast multipole method (Laplace kernel) ##

**Summary** This is a GPU implementation of the treecode/FMM for the Laplace kernel.
It includes a CPU implementation with/without SSE, and two different translation operators for the FMM.  See more details on the Wiki.

### Description ###
This code was produced to accompany a chapter contribution to the book _GPU Computing Gems_,  published in 2011.  To produce the software, we started with existing code to perform the FMM on GPU that had all the bells and whistles, but was not very friendly to read, and then proceeded to ruthlessly edit it to the minimum expression possible.  In the process, variable names were changed and code prettified, to make it more human-readable.  Moreover, many techniques which may be considered standard by the expert in FMM algorithms, were removed in the interest of expressing the methods in the simplest possible way. We hope that the result is software which is useful both for applications and pedagogically.

|We distribute this code under the MIT License, giving potential users the greatest freedom possible. We do, however, request fellow scientists that if they use our codes in research, they kindly include us in the acknowledgement of their papers.  We do not request gratuitous citations;  only cite our publications if you deem it warranted.|
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Related projects ###

  * **[PyFMM](http://code.google.com/p/pyfmm/)**: a Python implementation of the FMM for the calculation of the velocity field induced by _N_ vortex particles.
  * **[PetFMM](http://barbagroup.bu.edu/Barba_group/PetFMM.html)**: Open-source, parallel C++ implementation of the Fast Multipole Method. PetFMM can be obtained directly from its [repository](http://bitbucket.org/petfmm/petfmm-dev/) or following the links on the [group webpage](http://barbagroup.bu.edu/Barba_group/PetFMM.html).

### Publications ###

  * “Treecode and fast multipole method for N-body simulation with CUDA”,  Rio Yokota, Lorena A Barba, Ch. 9 in _GPU Computing Gems Emerald Edition_, Wen-mei Hwu, ed.; Morgan Kaufmann/Elsevier (2011) pp. 113–132. ISBN: 978-0-12-384988-5 [http://dx.doi.org/10.1016/B978-0-12-384988-5.00009-7](http://dx.doi.org/10.1016/B978-0-12-384988-5.00009-7)
Preprint in [arXiv:1010.1482](http://arxiv.org/abs/1010.1482)