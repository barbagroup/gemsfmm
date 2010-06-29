#ifndef __CPUKERNEL_H__
#define __CPUKERNEL_H__

#include "constants.h"

class FmmKernel
{
public:
  void direct(int numParticles);
  void precalc();
  void rotation(std::complex<double>* CnmIn, std::complex<double>* CnmOut, std::complex<double>** Dnm);
  void p2p(int numBoxIndex);
  void p2m(int numBoxIndex);
  void m2m(int numBoxIndex, int numBoxIndexOld, int numLevel);
  void m2l(int numBoxIndex, int numLevel);
  void l2l(int numBoxIndex, int numLevel);
  void l2p(int numBoxIndex);
  void m2p(int numBoxIndex, int numLevel);
};

#endif // __CPUKERNEL_H__
