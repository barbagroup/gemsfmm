#define MAIN
#include "fmm.h"
#undef MAIN

const int treeOrFMM = 1; // 0 : tree, 1: FMM 

int main(int argc, char *argv[]){
  int i,iteration,numParticles;
  double tic,toc,timeDirect,timeFMM,L2norm,difference,normalizer;
  vec3<float> *bodyAcceld;
  FmmKernel kernel;
  FmmSystem tree;
  std::fstream fid("time2.dat",std::ios::out);

  bodyAccel = new vec3<float>[maxParticles];
  bodyAcceld = new vec3<float>[maxParticles];
  bodyPos = new vec4<float>[maxParticles];

  for( i=0; i<maxParticles; i++ ) {
    bodyPos[i].x = rand()/(float) RAND_MAX*2*M_PI-M_PI;
    bodyPos[i].y = rand()/(float) RAND_MAX*2*M_PI-M_PI;
    bodyPos[i].z = rand()/(float) RAND_MAX*2*M_PI-M_PI;
    bodyPos[i].w = rand()/(float) RAND_MAX;
  }

  for( iteration=0; iteration<25; iteration++ ) {
    numParticles = int(pow(10,(iteration+32)/8.0));
    printf("N = %d\n",numParticles);

    tic = get_time();
    tree.fmmMain(numParticles,treeOrFMM);
    toc = get_time();
    timeFMM = toc-tic;
    printf("fmm    : %g\n",timeFMM);
    for ( i=0; i<9; i++ ) fid << t[i] << " ";
    fid << std::endl;
    for( i=0; i<numParticles; i++ ) {
      bodyAcceld[i].x = bodyAccel[i].x;
      bodyAcceld[i].y = bodyAccel[i].y;
      bodyAcceld[i].z = bodyAccel[i].z;
    }

    tic = get_time();
    kernel.direct(numParticles);
    toc = get_time();
    timeDirect = toc-tic;
    printf("direct : %g\n",timeDirect);

    L2norm = 0;
    for( i=0; i<numParticles; i++ ) {
      difference = (bodyAccel[i].x-bodyAcceld[i].x)*
             (bodyAccel[i].x-bodyAcceld[i].x)+
             (bodyAccel[i].y-bodyAcceld[i].y)*
             (bodyAccel[i].y-bodyAcceld[i].y)+
             (bodyAccel[i].z-bodyAcceld[i].z)*
             (bodyAccel[i].z-bodyAcceld[i].z);
      normalizer = bodyAccel[i].x*bodyAccel[i].x+
             bodyAccel[i].y*bodyAccel[i].y+
             bodyAccel[i].z*bodyAccel[i].z;
      L2norm += difference/normalizer/numParticles;
    }
    L2norm = sqrt(L2norm);
    printf("error  : %g\n\n",L2norm);
  }
  fid.close();
};
