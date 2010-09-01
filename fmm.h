#ifndef __FMM_H__
#define __FMM_H__

#include "kernel.h"

template<typename T>
class vec3 {
public:
  T x;
  T y;
  T z;
};

template<typename T>
class vec4 {
public:
  T x;
  T y;
  T z;
  T w;
};

#ifdef MAIN
vec3<float> *bodyAccel;
vec4<float> *bodyPos;
int maxLevel;                                    // number of FMM levels
int numBoxIndexFull;                             // full list of FMM boxes @ maxLevel
int numBoxIndexLeaf;                             // just the non-empty FMM boxes @ maxLevel
int numBoxIndexTotal;                            // total of numBoxIndexLeaf for all levels
int *permutation;                                // permutation key used for sorting particles
int **particleOffset;                            // first and last particle in each box
int *boxIndexMask;                               // link list for box index : Full -> NonEmpty
int *boxIndexFull;                               // link list for box index : NonEmpty -> Full
int *levelOffset;                                // offset of box index for each level
int *mortonIndex;                                // Morton index of each particle
int *numInteraction;                             // size of interaction list
int (*interactionList)[maxM2LInteraction];       // non-empty interaction list for P2P and M2L
int *boxOffsetStart;                             // offset of box index for GPU buffer
int *boxOffsetEnd;                               // offset of box index for GPU buffer
int *sortValue;                                  // temporary array used for Counting Sort
int *sortIndex;                                  // temporary array used for Counting Sort
int *sortValueBuffer;                            // temporary array used for Counting Sort
int *sortIndexBuffer;                            // temporary array used for Counting Sort
float rootBoxSize;                               // length of FMM domain
float *factorial;                                // factorial(n) = n!
vec3<float> boxMin;                              // axis limit of entire FMM domain
std::complex<double> (*Lnm)[numCoefficients];    // local expansion coefficients
std::complex<double> (*LnmOld)[numCoefficients]; // Lnm from previous level
std::complex<double> (*Mnm)[numCoefficients];    // multipole expansion coefficnets
std::complex<double> *Ynm;                       // spherical harmonic
std::complex<double> ***Dnm;                     // Wigner rotation matrix
double get_time(void) {                          // a simple timer
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return ((double)(tv.tv_sec+tv.tv_usec*1.0e-6));
}
double tic,t[9];                                 // event timer counter
void log_time(int stage) {                       // event timer
  struct timeval tv;
  struct timezone tz;
  double toc;
  gettimeofday(&tv, &tz);
  toc = tic;
  tic = ((double)(tv.tv_sec+tv.tv_usec*1.0e-6));
  t[stage] += tic-toc;
}
#else
extern vec3<float> *bodyAccel;
extern vec4<float> *bodyPos;
extern int maxLevel;
extern int numBoxIndexFull;
extern int numBoxIndexLeaf;
extern int numBoxIndexTotal;
extern int *permutation;
extern int **particleOffset;
extern int *boxIndexMask;
extern int *boxIndexFull;
extern int *levelOffset;
extern int *mortonIndex;
extern int *numInteraction;
extern int (*interactionList)[maxM2LInteraction];
extern int *boxOffsetStart;
extern int *boxOffsetEnd;
extern int *sortValue;
extern int *sortIndex;
extern int *sortValueBuffer;
extern int *sortIndexBuffer;
extern float rootBoxSize;
extern float *factorial;
extern vec3<float> boxMin;
extern std::complex<double> (*Lnm)[numCoefficients];
extern std::complex<double> (*LnmOld)[numCoefficients];
extern std::complex<double> (*Mnm)[numCoefficients],*Ynm,***Dnm;
extern double tic,t[9];
extern void get_time(int);
extern void log_time(int);
#endif

class FmmSystem
{
public:
  void allocate();
  void deallocate();
  void setDomainSize(int numParticles);
  void setOptimumLevel(int numParticles);
  void morton(int numParticles);
  void morton1(vec3<int> boxIndex3D, int& boxIndex, int numLevel);
  void unmorton(int boxIndex, vec3<int>& boxIndex3D);
  void sort(int numParticles);
  void sortParticles(int& numParticles);
  void unsortParticles(int& numParticles);
  void countNonEmptyBoxes(int numParticles);
  void getBoxData(int numParticles, int& numBoxIndex);
  void getBoxDataOfParent(int& numBoxIndex, int numLevel, int treeOrFMM);
  void getBoxIndexMask(int numBoxIndex, int numLevel);
  void getInteractionList(int numBoxIndex, int numLevel, int interactionType);
  void fmmMain(int numParticles, int treeOrFMM);
};
#endif // __FMM_H__
