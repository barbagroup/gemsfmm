#gives the context of this project

# Fast _N_-body simulation with CUDA #

### _N_-body problem ###

The classic _N_-body problem refers to determining the motion of _N_ particles that interact via a long-distance force, such as gravitation or electrostatics.  A straightforward approach to obtaining the forces affecting each particle is the evaluation of all pair-wise interactions, resulting in O(N<sup>2</sup>) computational complexity.  This method is only reasonable for moderate-size systems, or to compute near-field interactions, in combination with a far-field approximation.

### GPU Gems ###

In the book _GPU Gems, Vol. III_, Nguyen et al. (2007) presented the acceleration of the all-pairs computation on GPUs for the case of the gravitational potential of N masses. The natural parallelism available in the all-pairs kernel allowed excellent performance on the GPU architecture, and the direct kernel of Nyland et al. (2007) achieved over 200 Gigaflops on the GeForce8800GTX, calculating more than 19 billion interactions per second with _N_=16,384.

In the present contribution, we have addressed the more involved task of implementing the fast _N_-body algorithms that are used for providing a far-field approximation:  the O(N log N) treecode (Barnes & Hut, 1986) and O(N) fast multipole method (Greengard & Rokhlin, 1987).

### The GRAPE connection ###

Before embarking on the presentation of the algorithms and how they are efficiently cast onto the GPU, let us give some context.  The _N_-body problem of astrophysics was such a strong motivator to computational science, that it drove creation of a special supercomputer in Japan.  The history of this massively successful series of machines, called GRAPE, is summarized in the book by its creators, Makino & Taiji (1998).  A popular science magazine article also gives an overview ([Taubes, 1997](http://discovermagazine.com/1997/jun/thestarmachine1148)). The GRAPE machines continued to break records into the 21<sup>st</sup> century, but the size of the problems they can tackle using the O(N<sup>2</sup>) all-pairs force evaluation is still limited by the computational complexity.  As stated in Board & Schulten (2000),
> _"complexity trumps hardware."_

### References ###


> Barnes, J. and P. Hut, 1986: A hierarchical O(N log N) force-calculation algorithm. _Nature_, 324, 446–449.

> Board, J. and K. Schulten, 2000: The fast multipole algorithm. _Computing in Science and Engineering_, 2(1), 76–79.

> Greengard, L. and V. Rokhlin, 1987: A fast algorithm for particle simulations. _J. Comput. Phys._, 73(2), 325–348.

> Makino, J. and M. Taiji, 1998: _Scientiﬁc Simulations with Special-Purpose Computers—the GRAPE Systems_. John Wiley & Sons Inc.

> Nyland, L., M. Harris, and J. Prins, 2007: Fast n-body simulation with CUDA. In _GPU Gems 3_, Addison-Wesley Professional, chap. 31, pp. 677–695.