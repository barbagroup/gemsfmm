#include "fmm.h"
#include <cutil.h>

unsigned int hostOffsetSize;
unsigned int hostAccelSize;
unsigned int hostPosTargetSize;
unsigned int hostPosSourceSize;
unsigned int hostMnmTargetSize;
unsigned int hostMnmSourceSize;
unsigned int hostLnmTargetSize;
unsigned int hostLnmSourceSize;
unsigned int hostYnmSize;
unsigned int hostDnmSize;
unsigned int hostConstantSize;

int *hostOffset;
float3 *hostAccel;
float3 *hostPosTarget;
float4 *hostPosSource;
float *hostMnmTarget;
float *hostMnmSource;
float *hostLnmTarget;
float *hostLnmSource;
float *hostYnm;
float *hostDnm;
float *hostConstant;

static unsigned int is_set=0;
static unsigned int deviceOffsetSize=0;
static unsigned int deviceAccelSize=0;
static unsigned int devicePosTargetSize=0;
static unsigned int devicePosSourceSize=0;
static unsigned int deviceMnmTargetSize=0;
static unsigned int deviceMnmSourceSize=0;
static unsigned int deviceLnmTargetSize=0;
static unsigned int deviceLnmSourceSize=0;
static unsigned int deviceYnmSize=0;
static unsigned int deviceDnmSize=0;

static int *deviceOffset;
static float3 *deviceAccel;
static float3 *devicePosTarget;
static float4 *devicePosSource;
static float *deviceMnmTarget;
static float *deviceMnmSource;
static float *deviceLnmTarget;
static float *deviceLnmSource;
static float *deviceYnm;
static float *deviceDnm;

__device__ __constant__ float deviceConstant[4];

#include "gpukernelcore_p3.cu"

double get_gpu_time(void)
{
  struct timeval tv;
  struct timezone tz;
  if (is_set==1) cudaThreadSynchronize();
  gettimeofday(&tv, &tz);
  return ((double)(tv.tv_sec+tv.tv_usec*1.0e-6));
}

FmmSystem tree;

// direct summation kernel
void FmmKernel::direct(int n) {
  int i,nicall,njcall,icall,iwork1,iwork2,ista,iend,ibase,isize,iblok,is,im;
  int jcall,jwork1,jwork2,jsta,jend,jbase,jsize;
  int nj,nflop;
  const int offsetStride = 2*maxP2PInteraction+1;
  double tic,toc,flops,t[10],op=0;

  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  for( i=0; i<n; i++ ) {
    bodyAccel[i].x = 0;
    bodyAccel[i].y = 0;
    bodyAccel[i].z = 0;
  }
  nicall = n/targetBufferSize+1;
  njcall = n/sourceBufferSize+1;
  iblok = (n/nicall+threadsPerBlockTypeA-1)/threadsPerBlockTypeA;

  hostOffsetSize=sizeof(int)*iblok*offsetStride;
  hostPosTargetSize=sizeof(float3)*targetBufferSize;
  hostPosSourceSize=sizeof(float4)*sourceBufferSize;
  hostAccelSize=sizeof(float3)*targetBufferSize;

  hostOffset=(int *)malloc(hostOffsetSize);
  hostPosTarget=(float3 *)malloc(hostPosTargetSize);
  hostPosSource=(float4 *)malloc(hostPosSourceSize);
  hostAccel=(float3 *)malloc(hostAccelSize);

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostPosTargetSize>devicePosTargetSize) {
    if(devicePosTargetSize!=0) CUDA_SAFE_CALL(cudaFree(devicePosTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devicePosTarget,hostPosTargetSize));
    devicePosTargetSize=hostPosTargetSize;
  }
  if (hostPosSourceSize>devicePosSourceSize) {
    if(devicePosSourceSize!=0) CUDA_SAFE_CALL(cudaFree(devicePosSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devicePosSource,hostPosSourceSize));
    devicePosSourceSize=hostPosSourceSize;
  }
  if (hostAccelSize>deviceAccelSize) {
    if(deviceAccelSize!=0) CUDA_SAFE_CALL(cudaFree(deviceAccel));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceAccel,hostAccelSize));
    deviceAccelSize=hostAccelSize;
  }
  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  if( n < 0 ) nicall = 0;
  for( icall=0; icall<nicall; icall++ ) {
    iwork1 = n/nicall;
    iwork2 = n%nicall;
    ista = icall*iwork1+std::min(icall,iwork2);
    iend = ista+iwork1-1;
    if( iwork2 > icall ) iend++;
    ibase = ista;
    isize = iend-ibase+1;
    iblok = 0;
    for( is=0; is<isize; is+=threadsPerBlockTypeA ) {
      for( i=0; i<std::min(isize-is,threadsPerBlockTypeA); i++ ) {
        im = iblok*threadsPerBlockTypeA+i;
        hostPosTarget[im] = *(float3*) &bodyPos[ibase+is+i];
      }
      for( i=isize-is; i<threadsPerBlockTypeA; i++ ) {
        im = iblok*threadsPerBlockTypeA+i;
        hostPosTarget[im].x = 0;
        hostPosTarget[im].y = 0;
        hostPosTarget[im].z = 0;
      }
      iblok++;
    }
    for( jcall=0; jcall<njcall; jcall++ ) {
      jwork1 = n/njcall;
      jwork2 = n%njcall;
      jsta = jcall*jwork1+std::min(jcall,jwork2);
      jend = jsta+jwork1;
      if( jwork2 > jcall ) jend++;
      jbase = jsta;
      jsize = jend-jbase;
      for( i=0; i<iblok; i++ ) {
        hostOffset[i*offsetStride] = 1;
        hostOffset[i*offsetStride+1] = 0;
        hostOffset[i*offsetStride+2] = jsize;
      }
      for( i=jsta; i<jend; i++ ) {
        nj = i-jsta;
        hostPosSource[nj] = *(float4*) &bodyPos[i];
      }
      op += (double) isize*jsize;

      toc=tic;
      tic=get_gpu_time();
      t[0]+=tic-toc;
      CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(devicePosTarget,hostPosTarget,hostPosTargetSize,cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(devicePosSource,hostPosSource,hostPosSourceSize,cudaMemcpyHostToDevice));
      toc=tic;
      tic=get_gpu_time();
      t[2]+=tic-toc;

      dim3 block(threadsPerBlockTypeA);
      dim3 grid(iblok);
      p2p_kernel<<< grid, block >>>(deviceOffset,devicePosTarget,devicePosSource,deviceAccel);
      CUT_CHECK_ERROR("Kernel execution failed");
      nflop = 19;

      toc=tic;
      tic=get_gpu_time();
      t[3]+=tic-toc;
      CUDA_SAFE_CALL(cudaMemcpy(hostAccel,deviceAccel,hostAccelSize,cudaMemcpyDeviceToHost));
      toc=tic;
      tic=get_gpu_time();
      t[2]+=tic-toc;

      iblok = 0;
      for( is=0; is<isize; is+=threadsPerBlockTypeA ) {
        for( i=0; i<std::min(isize-is,threadsPerBlockTypeA); i++ ) {
          im = iblok*threadsPerBlockTypeA+i;
          bodyAccel[ibase+is+i].x += inv4PI*hostAccel[im].x;
          bodyAccel[ibase+is+i].y += inv4PI*hostAccel[im].y;
          bodyAccel[ibase+is+i].z += inv4PI*hostAccel[im].z;
        }
        iblok++;
      }
    }
  }
  free(hostOffset);
  free(hostPosTarget);
  free(hostPosSource);
  free(hostAccel);

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("p2p cudaMalloc : %f s\n",t[1]);
//  printf("p2p cudaMemcpy : %f s\n",t[2]);
//  printf("p2p cudaKernel : %f s\n",t[3]);
//  printf("p2p other      : %f s\n",t[0]);
//  printf("p2p flops      : %f G\n",flops/1e9);
  tic=flops;
}

// precalculate M2L translation matrix and Wigner rotation matrix
void FmmKernel::precalc() {
  int n,m,nm,nabsm,j,k,nk,npn,nmn,npm,nmm,nmk,i,nmk1,nm1k,nmk2;
  vec3<int> boxIndex3D;
  double anmk[2][numExpansion4];
  double Dnmd[numExpansion4];
  double fnma,fnpa,pn,p,p1,p2,anmd,anmkd,xijc,yijc,zijc,rho,alpha,beta,sc,ank,ek;
  std::complex<double> expBeta[numExpansion2],eim(0,1),cnm;

  for( n=0; n<2*numExpansions; n++ ) {
    for( m=-n; m<=n; m++ ) {
      nm = n*n+n+m;
      nabsm = abs(m);
      fnma = 1.0;
      for( i=1; i<=n-nabsm; i++ ) fnma *= i;
      fnpa = 1.0;
      for( i=1; i<=n+nabsm; i++ ) fnpa *= i;
      factorial[nm] = sqrt(fnma/fnpa);
    }
  }

  pn = 1;
  for( m=0; m<2*numExpansions; m++ ) {
    p = pn;
    npn = m*m+2*m;
    nmn = m*m;
    Ynm[npn] = factorial[npn]*p;
    Ynm[nmn] = conj(Ynm[npn]);
    p1 = p;
    p = (2*m+1)*p;
    for( n=m+1; n<2*numExpansions; n++ ) {
      npm = n*n+n+m;
      nmm = n*n+n-m;
      Ynm[npm] = factorial[npm]*p;
      Ynm[nmm] = conj(Ynm[npm]);
      p2 = p1;
      p1 = p;
      p = ((2*n+1)*p1-(n+m)*p2)/(n-m+1);
    }
    pn = 0;
  }

  for( n=0; n<numExpansions; n++ ) {
    for( m=1; m<=n; m++ ) {
      anmd = n*(n+1)-m*(m-1);
      for( k=1-m; k<m; k++ ) {
        nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
        anmkd = ((double) (n*(n+1)-k*(k+1)))/(n*(n+1)-m*(m-1));
        anmk[0][nmk] = -(m+k)/sqrt(anmd);
        anmk[1][nmk] = sqrt(anmkd);
      }
    }
  }

  for( i=0; i<numRelativeBox; i++ ) {
    tree.unmorton(i,boxIndex3D);
    xijc = boxIndex3D.x-3;
    yijc = boxIndex3D.y-3;
    zijc = boxIndex3D.z-3;
    rho = sqrt(xijc*xijc+yijc*yijc+zijc*zijc)+eps;
    alpha = acos(zijc/rho);
    if( std::abs(xijc)+std::abs(yijc) < eps ) {
      beta = 0;
    } else if( std::abs(xijc) < eps ) {
      beta = yijc/std::abs(yijc)*M_PI*0.5;
    } else if( xijc > 0 ) {
      beta = atan(yijc/xijc);
    } else {
      beta = atan(yijc/xijc)+M_PI;
    }

    sc = sin(alpha)/(1+cos(alpha));
    for( n=0; n<4*numExpansions-3; n++ ) {
      expBeta[n] = exp((n-2*numExpansions+2)*beta*eim);
    }

    for( n=0; n<numExpansions; n++ )  {
      nmk = (4*n*n*n+6*n*n+5*n)/3+n*(2*n+1)+n;
      Dnmd[nmk] = pow(cos(alpha*0.5),2*n);
      for( k=n; k>=1-n; k-- ) {
        nmk = (4*n*n*n+6*n*n+5*n)/3+n*(2*n+1)+k;
        nmk1 = (4*n*n*n+6*n*n+5*n)/3+n*(2*n+1)+k-1;
        ank = ((double) n+k)/(n-k+1);
        Dnmd[nmk1] = sqrt(ank)*tan(alpha*0.5)*Dnmd[nmk];
      }
      for( m=n; m>=1; m-- ) {
        for( k=m-1; k>=1-m; k-- ){
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nmk1 = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k+1;
          nm1k = (4*n*n*n+6*n*n+5*n)/3+(m-1)*(2*n+1)+k;
          Dnmd[nm1k] = anmk[1][nmk]*Dnmd[nmk1]+anmk[0][nmk]*sc*Dnmd[nmk];
        }
      }
    }

    for( n=1; n<numExpansions; n++ ) {
      for( m=0; m<=n; m++ ) {
        for( k=-m; k<=-1; k++ ) {
          ek = pow(-1.0,k);
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nmk1 = (4*n*n*n+6*n*n+5*n)/3-k*(2*n+1)-m;
          Dnmd[nmk] = ek*Dnmd[nmk];
          Dnmd[nmk1] = pow(-1.0,m+k)*Dnmd[nmk];
        }
        for( k=0; k<=m; k++ ) {
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nmk1 = (4*n*n*n+6*n*n+5*n)/3+k*(2*n+1)+m;
          nmk2 = (4*n*n*n+6*n*n+5*n)/3-k*(2*n+1)-m;
          Dnmd[nmk1] = pow(-1.0,m+k)*Dnmd[nmk];
          Dnmd[nmk2] = Dnmd[nmk1];
        }
      }
    }

    for( n=0; n<numExpansions; n++ ) {
      for( m=0; m<=n; m++ ) {
        for( k=-n; k<=n; k++ ) {
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nk = n*(n+1)+k;
          Dnm[i][m][nk] = Dnmd[nmk]*expBeta[k+m+2*numExpansions-2];
        }
      }
    }

    alpha = -alpha;
    beta = -beta;

    sc = sin(alpha)/(1+cos(alpha));
    for( n=0; n<4*numExpansions-3; n++ ) {
      expBeta[n] = exp((n-2*numExpansions+2)*beta*eim);
    }

    for( n=0; n<numExpansions; n++ ) {
      nmk = (4*n*n*n+6*n*n+5*n)/3+n*(2*n+1)+n;
      Dnmd[nmk] = pow(cos(alpha*0.5),2*n);
      for( k=n; k>=1-n; k-- ) {
        nmk = (4*n*n*n+6*n*n+5*n)/3+n*(2*n+1)+k;
        nmk1 = (4*n*n*n+6*n*n+5*n)/3+n*(2*n+1)+k-1;
        ank = ((double) n+k)/(n-k+1);
        Dnmd[nmk1] = sqrt(ank)*tan(alpha*0.5)*Dnmd[nmk];
      }
      for( m=n; m>=1; m-- ) {
        for( k=m-1; k>=1-m; k-- ) {
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nmk1 = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k+1;
          nm1k = (4*n*n*n+6*n*n+5*n)/3+(m-1)*(2*n+1)+k;
          Dnmd[nm1k] = anmk[1][nmk]*Dnmd[nmk1]+anmk[0][nmk]*sc*Dnmd[nmk];
        }
      }
    }

    for( n=1; n<numExpansions; n++ ) {
      for( m=0; m<=n; m++ ) {
        for( k=-m; k<=-1; k++ ) {
          ek = pow(-1.0,k);
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nmk1 = (4*n*n*n+6*n*n+5*n)/3-k*(2*n+1)-m;
          Dnmd[nmk] = ek*Dnmd[nmk];
          Dnmd[nmk1] = pow(-1.0,m+k)*Dnmd[nmk];
        }
        for( k=0; k<=m; k++ ) {
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nmk1 = (4*n*n*n+6*n*n+5*n)/3+k*(2*n+1)+m;
          nmk2 = (4*n*n*n+6*n*n+5*n)/3-k*(2*n+1)-m;
          Dnmd[nmk1] = pow(-1.0,m+k)*Dnmd[nmk];
          Dnmd[nmk2] = Dnmd[nmk1];
        }
      }
    }

    for( n=0; n<numExpansions; n++ ) {
      for( m=0; m<=n; m++ ) {
        for( k=-n; k<=n; k++ ) {
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
          nk = n*(n+1)+k;
          Dnm[i+numRelativeBox][m][nk] = Dnmd[nmk]*expBeta[k+m+2*numExpansions-2];
        }
      }
    }
  }

  for( j=0; j<numBoxIndexTotal; j++ ) {
    for( i=0; i<numCoefficients; i++ ) {
      Mnm[j][i] = 0;
    }
  }
}

// p2p
void FmmKernel::p2p(int numBoxIndex) {
  int nicall,jc,jj,ii,njd,ij,icall,jcall,iblok,im,jjd,j,ibase,isize,is,i,ijc,jjdd;
  int ni,nj,nflop,*jbase,*jsize,*njcall,*njj,**interactionListOffsetStart,**interactionListOffsetEnd;
  const int offsetStride = 2*maxP2PInteraction+1;
  double tic,toc,flops,t[10],op=0;

  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  hostOffsetSize=sizeof(int)*targetBufferSize/threadsPerBlockTypeA*offsetStride;
  hostPosTargetSize=sizeof(float3)*targetBufferSize;
  hostPosSourceSize=sizeof(float4)*sourceBufferSize;
  hostAccelSize=sizeof(float3)*targetBufferSize;

  hostOffset=(int *)malloc(hostOffsetSize);
  hostPosTarget=(float3 *)malloc(hostPosTargetSize);
  hostPosSource=(float4 *)malloc(hostPosSourceSize);
  hostAccel=(float3 *)malloc(hostAccelSize);
  interactionListOffsetStart = new int* [maxM2LInteraction];
  for( i=0; i<maxM2LInteraction; i++ ) interactionListOffsetStart[i] = new int [numBoxIndexLeaf];
  interactionListOffsetEnd = new int* [maxM2LInteraction];
  for( i=0; i<maxM2LInteraction; i++ ) interactionListOffsetEnd[i] = new int [numBoxIndexLeaf];
  jbase = new int [numBoxIndexLeaf];
  jsize = new int [numBoxIndexLeaf];
  njcall = new int [numBoxIndexLeaf];
  njj = new int [numBoxIndexLeaf];

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }
  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostPosTargetSize>devicePosTargetSize) {
    if(devicePosTargetSize!=0) CUDA_SAFE_CALL(cudaFree(devicePosTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devicePosTarget,hostPosTargetSize));
    devicePosTargetSize=hostPosTargetSize;
  }
  if (hostPosSourceSize>devicePosSourceSize) {
    if(devicePosSourceSize!=0) CUDA_SAFE_CALL(cudaFree(devicePosSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devicePosSource,hostPosSourceSize));
    devicePosSourceSize=hostPosSourceSize;
  }
  if (hostAccelSize>deviceAccelSize) {
    if(deviceAccelSize!=0) CUDA_SAFE_CALL(cudaFree(deviceAccel));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceAccel,hostAccelSize));
    deviceAccelSize=hostAccelSize;
  }
  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  ni = 0;
  nj = 0;
  nicall = 0;
  boxOffsetStart[0] = 0;
  jc = 0;
  for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
  for( ii=0; ii<numBoxIndex; ii++ ) {
    if( numInteraction[ii] != 0 ) {
      njd = 0;
      jc = 0;
      interactionListOffsetStart[0][ii] = 0;
      for( ij=0; ij<numInteraction[ii]; ij++ ) {
        jj = interactionList[ii][ij];
        if( njj[jj] == 0 ) {
          nj += particleOffset[1][jj]-particleOffset[0][jj]+1;
          njj[jj] = 1;
        }
        njd += particleOffset[1][jj]-particleOffset[0][jj]+1;
        if( njd > sourceBufferSize ) {
          interactionListOffsetEnd[jc][ii] = ij-1;
          jc++;
          interactionListOffsetStart[jc][ii] = ij;
          njd = particleOffset[1][jj]-particleOffset[0][jj]+1;
        }
      }
      interactionListOffsetEnd[jc][ii] = numInteraction[ii]-1;
      ni += ((particleOffset[1][ii]-particleOffset[0][ii]+threadsPerBlockTypeA)/threadsPerBlockTypeA+1)
            *threadsPerBlockTypeA;
      if( jc != 0 ) {
        if( ii > boxOffsetStart[nicall] ) {
          njcall[nicall] = 1;
          boxOffsetEnd[nicall] = ii-1;
          nicall++;
          assert( nicall < numBoxIndexLeaf );
          boxOffsetStart[nicall] = ii;
          for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
        }
        if( ii != numBoxIndex ) {
          njcall[nicall] = jc+1;
          boxOffsetEnd[nicall] = ii;
          nicall++;
          assert( nicall < numBoxIndexLeaf );
          boxOffsetStart[nicall] = ii+1;
          for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
          ni = 0;
          nj = 0;
        }
      } else if ( ni > targetBufferSize || nj > sourceBufferSize ) {
        njcall[nicall] = jc+1;
        boxOffsetEnd[nicall] = ii-1;
        nicall++;
        assert( nicall < numBoxIndexLeaf );
        boxOffsetStart[nicall] = ii;
        for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
        ni = ((particleOffset[1][ii]-particleOffset[0][ii]+threadsPerBlockTypeA)/threadsPerBlockTypeA+1)
            *threadsPerBlockTypeA;
        nj = 0;
        for( ij=0; ij<numInteraction[ii]; ij++ ) {
          jj = interactionList[ii][ij];
          nj += particleOffset[1][jj]-particleOffset[0][jj]+1;
          njj[jj] = 1;
        }
      }
    }
  }
  njcall[nicall] = jc+1;
  boxOffsetEnd[nicall] = numBoxIndex-1;
  if(numBoxIndex != 0) nicall++;

  for( icall=0; icall<nicall; icall++ ) {
    for( jcall=0; jcall<njcall[icall]; jcall++ ) {
      iblok = 0;
      jc = 0;
      jjd = 0;
      for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
      for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
        if( numInteraction[ii] != 0 ) {
          for( ij=interactionListOffsetStart[jcall][ii]; ij<=interactionListOffsetEnd[jcall][ii]; ij++ ) {
            jj = interactionList[ii][ij];
            if( njj[jj] == 0 ) {
              jbase[jjd] = jc;
              for( j=particleOffset[0][jj]; j<=particleOffset[1][jj]; j++ ) {
                hostPosSource[jc] = *(float4*) &bodyPos[j];
                jc++;
              }
              jsize[jjd] = jc-jbase[jjd];
              jjd++;
              njj[jj] = jjd;
            }
          }
          ibase = particleOffset[0][ii];
          isize = particleOffset[1][ii]-ibase+1;
          for( is=0; is<isize; is+=threadsPerBlockTypeA ) {
            for( i=0; i<std::min(isize-is,threadsPerBlockTypeA); i++ ) {
              im = iblok*threadsPerBlockTypeA+i;
              hostPosTarget[im] = *(float3*) &bodyPos[ibase+is+i];
            }
            for( i=isize-is; i<threadsPerBlockTypeA; i++ ) {
              im = iblok*threadsPerBlockTypeA+i;
              hostPosTarget[im].x = 0;
              hostPosTarget[im].y = 0;
              hostPosTarget[im].z = 0;
            }
            hostOffset[iblok*offsetStride] = interactionListOffsetEnd[jcall][ii]
                                            -interactionListOffsetStart[jcall][ii]+1;
            ijc = 0;
            for( ij=interactionListOffsetStart[jcall][ii]; ij<=interactionListOffsetEnd[jcall][ii]; ij++ ) {
              jj = interactionList[ii][ij];
              if( njj[jj] != 0 ) {
                jjdd = njj[jj]-1;
                hostOffset[iblok*offsetStride+2*ijc+1] = jbase[jjdd];
                hostOffset[iblok*offsetStride+2*ijc+2] = jsize[jjdd];
                op += (double) threadsPerBlockTypeA*jsize[jjdd];
                ijc++;
              }
            }
            iblok++;
          }
        }
      }
      if( iblok != 0 ) {

        toc=tic;
        tic=get_gpu_time();
        t[0]+=tic-toc;
        CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(devicePosTarget,hostPosTarget,hostPosTargetSize,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(devicePosSource,hostPosSource,hostPosSourceSize,cudaMemcpyHostToDevice));
        toc=tic;
        tic=get_gpu_time();
        t[2]+=tic-toc;

        dim3 block(threadsPerBlockTypeA);
        dim3 grid(iblok);
        p2p_kernel<<< grid, block >>>(deviceOffset,devicePosTarget,devicePosSource,deviceAccel);
        CUT_CHECK_ERROR("Kernel execution failed");
        nflop = 19;

        toc=tic;
        tic=get_gpu_time();
        t[3]+=tic-toc;
        CUDA_SAFE_CALL(cudaMemcpy(hostAccel,deviceAccel,hostAccelSize,cudaMemcpyDeviceToHost));
        toc=tic;
        tic=get_gpu_time();
        t[2]+=tic-toc;

      }
      iblok = 0;
      for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
        if( numInteraction[ii] != 0 ) {
          ibase = particleOffset[0][ii];
          isize = particleOffset[1][ii]-ibase+1;
          for( is=0; is<isize; is+=threadsPerBlockTypeA ) {
            for( i=0; i<std::min(isize-is,threadsPerBlockTypeA); i++ ) {
              im = iblok*threadsPerBlockTypeA+i;
              bodyAccel[ibase+is+i].x += inv4PI*hostAccel[im].x;
              bodyAccel[ibase+is+i].y += inv4PI*hostAccel[im].y;
              bodyAccel[ibase+is+i].z += inv4PI*hostAccel[im].z;
            }
            iblok++;
          }
        }
      }
    }
  }
  free(hostOffset);
  free(hostPosTarget);
  free(hostPosSource);
  free(hostAccel);
  for( i=0; i<maxM2LInteraction; i++ ) delete[] interactionListOffsetStart[i];
  delete[] interactionListOffsetStart;
  for( i=0; i<maxM2LInteraction; i++ ) delete[] interactionListOffsetEnd[i];
  delete[] interactionListOffsetEnd;
  delete[] jbase;
  delete[] jsize;
  delete[] njcall;
  delete[] njj;

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("p2p cudaMalloc : %f s\n",t[1]);
//  printf("p2p cudaMemcpy : %f s\n",t[2]);
//  printf("p2p cudaKernel : %f s\n",t[3]);
//  printf("p2p other      : %f s\n",t[0]);
//  printf("p2p flops      : %f G\n",flops/1e9);
  tic=flops;
}

// p2m
void FmmKernel::p2m(int numBoxIndex) {
  int ncall,jj,icall,iblok,jc,jbase,j,jsize,jm;
  int i,ni,nj,nflop;
  const int offsetStride = 3;
  double tic,toc,flops,t[10],boxSize,op=0;

  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  boxSize = rootBoxSize/(1 << maxLevel);

  hostConstantSize=sizeof(float)*4;
  hostOffsetSize=sizeof(int)*targetBufferSize/threadsPerBlockTypeB*offsetStride;
  hostMnmTargetSize=sizeof(float)*2*targetBufferSize;
  hostPosSourceSize=sizeof(float4)*sourceBufferSize;

  hostConstant=(float *)malloc(hostConstantSize);
  hostOffset=(int *)malloc(hostOffsetSize);
  hostMnmTarget=(float *)malloc(hostMnmTargetSize);
  hostPosSource=(float4 *)malloc(hostPosSourceSize);

  hostConstant[0]=(float) boxSize;
  hostConstant[1]=(float) boxMin.x;
  hostConstant[2]=(float) boxMin.y;
  hostConstant[3]=(float) boxMin.z;

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }
  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostMnmTargetSize>deviceMnmTargetSize) {
    if(deviceMnmTargetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceMnmTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceMnmTarget,hostMnmTargetSize));
    deviceMnmTargetSize=hostMnmTargetSize;
  }
  if (hostPosSourceSize>devicePosSourceSize) {
    if(devicePosSourceSize!=0) CUDA_SAFE_CALL(cudaFree(devicePosSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devicePosSource,hostPosSourceSize));
    devicePosSourceSize=hostPosSourceSize;
  }
  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  ni = 0;
  nj = 0;
  ncall = 0;
  boxOffsetStart[0] = 0;
  for( jj=0; jj<numBoxIndex; jj++ ) {
    ni += ((numCoefficients+threadsPerBlockTypeB)/threadsPerBlockTypeB+1)*threadsPerBlockTypeB;
    nj += particleOffset[1][jj]-particleOffset[0][jj]+1;
    if( ni > targetBufferSize || nj > sourceBufferSize ) {
      boxOffsetEnd[ncall] = jj-1;
      ncall++;
      boxOffsetStart[ncall] = jj;
      ni = ((numCoefficients+threadsPerBlockTypeB)/threadsPerBlockTypeB+1)*threadsPerBlockTypeB;
      nj = particleOffset[1][jj]-particleOffset[0][jj]+1;
    }
  }
  boxOffsetEnd[ncall] = numBoxIndex-1;
  if(numBoxIndex != 0) ncall++;

  for( icall=0; icall<ncall; icall++ ) {
    iblok = 0;
    jc = 0;
    for( jj=boxOffsetStart[icall]; jj<=boxOffsetEnd[icall]; jj++ ) {
      jbase = jc;
      for( j=particleOffset[0][jj]; j<=particleOffset[1][jj]; j++ ) {
        hostPosSource[jc] = *(float4*) &bodyPos[j];
        jc++;
      }
      jsize = jc-jbase;
      hostOffset[iblok*offsetStride] = boxIndexFull[jj];
      hostOffset[iblok*offsetStride+1] = jbase;
      hostOffset[iblok*offsetStride+2] = jsize;
      op += threadsPerBlockTypeB*jsize;
      iblok++;
    }

    toc=tic;
    tic=get_gpu_time();
    t[0]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceConstant,hostConstant,hostConstantSize));
    CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(devicePosSource,hostPosSource,hostPosSourceSize,cudaMemcpyHostToDevice));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    dim3 block(threadsPerBlockTypeB);
    dim3 grid(iblok);
    p2m_kernel<<< grid, block >>>(deviceOffset,deviceMnmTarget,devicePosSource);
    CUT_CHECK_ERROR("Kernel execution failed");
    nflop = 20;

    toc=tic;
    tic=get_gpu_time();
    t[3]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpy(hostMnmTarget,deviceMnmTarget,hostMnmTargetSize,cudaMemcpyDeviceToHost));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    iblok = 0;
    for( jj=boxOffsetStart[icall]; jj<=boxOffsetEnd[icall]; jj++ ) {
      for( j=0; j<numCoefficients; j++ ) {
        jm = iblok*threadsPerBlockTypeB+j;
        Mnm[jj][j] = std::complex<double>(hostMnmTarget[2*jm+0],hostMnmTarget[2*jm+1]);
      }
      iblok++;
    }
  }

  free(hostConstant);
  free(hostOffset);
  free(hostMnmTarget);
  free(hostPosSource);

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("p2m cudaMalloc : %f s\n",t[1]);
//  printf("p2m cudaMemcpy : %f s\n",t[2]);
//  printf("p2m cudaKernel : %f s\n",t[3]);
//  printf("p2m other      : %f s\n",t[0]);
//  printf("p2m flops      : %f G\n",flops/1e9);
  tic=flops;
}

// m2m
void FmmKernel::m2m(int numBoxIndex, int numBoxIndexOld, int numLevel) {
  int n,m,npm,nmm,je,k,nmk,ii,ib,j,ncall,jj,icall;
  int iblok,jc,nfic,jb,jbase,jsize,nfip,jm;
  int i,nj,nk,nflop;
  vec3<int> boxIndex3D;
  const int offsetStride = 3;
  double tic,toc,flops,t[10],boxSize,op=0;
  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  boxSize = rootBoxSize/(1 << numLevel);

  hostConstantSize=sizeof(float)*4;
  hostOffsetSize=sizeof(int)*targetBufferSize/threadsPerBlockTypeB*offsetStride;
  hostMnmTargetSize=sizeof(float)*2*targetBufferSize;
  hostMnmSourceSize=sizeof(float)*2*sourceBufferSize;
  hostYnmSize=sizeof(float)*2*numExpansion2;
  hostDnmSize=sizeof(float)*2*DnmSize*2*numRelativeBox;

  hostConstant=(float *)malloc(hostConstantSize);
  hostOffset=(int *)malloc(hostOffsetSize);
  hostMnmTarget=(float *)malloc(hostMnmTargetSize);
  hostMnmSource=(float *)malloc(hostMnmSourceSize);
  hostYnm=(float *)malloc(hostYnmSize);
  hostDnm=(float *)malloc(hostDnmSize);

  hostConstant[0]=(float) boxSize;
  hostConstant[1]=0;
  hostConstant[2]=0;
  hostConstant[3]=0;

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }
  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostMnmTargetSize>deviceMnmTargetSize) {
    if(deviceMnmTargetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceMnmTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceMnmTarget,hostMnmTargetSize));
    deviceMnmTargetSize=hostMnmTargetSize;
  }
  if (hostMnmSourceSize>deviceMnmSourceSize) {
    if(deviceMnmSourceSize!=0) CUDA_SAFE_CALL(cudaFree(deviceMnmSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceMnmSource,hostMnmSourceSize));
    deviceMnmSourceSize=hostMnmSourceSize;
  }
  if (hostYnmSize>deviceYnmSize) {
    if(deviceYnmSize!=0) CUDA_SAFE_CALL(cudaFree(deviceYnm));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceYnm,hostYnmSize));
    deviceYnmSize=hostYnmSize;
  }
  if (hostDnmSize>deviceDnmSize) {
    if(deviceDnmSize!=0) CUDA_SAFE_CALL(cudaFree(deviceDnm));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceDnm,hostDnmSize));
    deviceDnmSize=hostDnmSize;
  }
  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  for( m=0; m<numExpansions; m++ ) {
    for( n=m; n<numExpansions; n++ ) {
      npm = n*n+n+m;
      nmm = n*n+n-m;
      hostYnm[2*npm+0] = std::real(Ynm[npm]);
      hostYnm[2*nmm+0] = std::real(Ynm[nmm]);
      hostYnm[2*npm+1] = std::imag(Ynm[npm]);
      hostYnm[2*nmm+1] = std::imag(Ynm[nmm]);
    }
  }
  for( je=0; je<2*numRelativeBox; je++ ) {
    for( n=0; n<numExpansions; n++ ) {
      for( m=0; m<=n; m++ ) {
        for( k=-n; k<=n; k++ ) {
          nk = n*(n+1)+k;
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k+je*DnmSize;
          hostDnm[2*nmk+0] = std::real(Dnm[je][m][nk]);
          hostDnm[2*nmk+1] = std::imag(Dnm[je][m][nk]);
        }
      }
    }
  }

  for( ii=0; ii<numBoxIndex; ii++ ) {
    ib = ii+levelOffset[numLevel-1];
    for( j=0; j<numCoefficients; j++ ) {
      Mnm[ib][j] = 0;
    }
  }

  nj = 0;
  ncall = 0;
  boxOffsetStart[0] = 0;
  for( jj=0; jj<numBoxIndexOld; jj++ ) {
    nj += threadsPerBlockTypeB;
    if( nj > sourceBufferSize ) {
      boxOffsetEnd[ncall] = jj-1;
      ncall++;
      boxOffsetStart[ncall] = jj;
      nj = threadsPerBlockTypeB;
    }
  }
  boxOffsetEnd[ncall] = numBoxIndexOld-1;
  if(numBoxIndexOld != 0) ncall++;

  for( icall=0; icall<ncall; icall++ ) {
    iblok = 0;
    jc = 0;
    for( jj=boxOffsetStart[icall]; jj<=boxOffsetEnd[icall]; jj++ ) {
      jb = jj+levelOffset[numLevel];
      nfic = boxIndexFull[jb]%8;
      tree.unmorton(nfic,boxIndex3D);
      boxIndex3D.x = 4-boxIndex3D.x*2;
      boxIndex3D.y = 4-boxIndex3D.y*2;
      boxIndex3D.z = 4-boxIndex3D.z*2;
      tree.morton1(boxIndex3D,je,3);
      jbase = jc;
      for( j=0; j<numCoefficients; j++ ) {
        hostMnmSource[2*jc+0] = std::real(Mnm[jb][j]);
        hostMnmSource[2*jc+1] = std::imag(Mnm[jb][j]);
        jc++;
      }
      jsize = jc-jbase;
      hostOffset[iblok*offsetStride] = 1;
      hostOffset[iblok*offsetStride+1] = jbase;
      hostOffset[iblok*offsetStride+2] = je+1;
      op += threadsPerBlockTypeB*jsize;
      iblok++;
    }

    toc=tic;
    tic=get_gpu_time();
    t[0]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceConstant,hostConstant,hostConstantSize));
    CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceMnmSource,hostMnmSource,hostMnmSourceSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceYnm,hostYnm,hostYnmSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceDnm,hostDnm,hostDnmSize,cudaMemcpyHostToDevice));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    dim3 block(threadsPerBlockTypeB);
    dim3 grid(iblok);
    m2m_kernel<<< grid, block >>>(deviceOffset,deviceMnmTarget,deviceMnmSource,deviceYnm,deviceDnm);
    CUT_CHECK_ERROR("Kernel execution failed");
    nflop = 48;

    toc=tic;
    tic=get_gpu_time();
    t[3]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpy(hostMnmTarget,deviceMnmTarget,hostMnmTargetSize,cudaMemcpyDeviceToHost));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    iblok = 0;
    for( jj=boxOffsetStart[icall]; jj<=boxOffsetEnd[icall]; jj++ ) {
      jb = jj+levelOffset[numLevel];
      nfip = boxIndexFull[jb]/8;
      ib = boxIndexMask[nfip]+levelOffset[numLevel-1];
      for( j=0; j<numCoefficients; j++ ) {
        jm = iblok*threadsPerBlockTypeB+j;
        Mnm[ib][j] += std::complex<double>(hostMnmTarget[2*jm+0],hostMnmTarget[2*jm+1]);
      }
      iblok++;
    }
  }

  free(hostConstant);
  free(hostOffset);
  free(hostMnmTarget);
  free(hostMnmSource);
  free(hostYnm);
  free(hostDnm);

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("m2m cudaMalloc : %f s\n",t[1]);
//  printf("m2m cudaMemcpy : %f s\n",t[2]);
//  printf("m2m cudaKernel : %f s\n",t[3]);
//  printf("m2m other      : %f s\n",t[0]);
//  printf("m2m flops      : %f G\n",flops/1e9);
  tic=flops;
}

// m2l
void FmmKernel::m2l(int numBoxIndex, int numLevel) {
  int i,j,m,n,npm,nmm,je,k,nmk,ncall,jj,ii,ib,ij,icall,iblok,jc,jjd;
  int jb,jbd,ix,iy,iz,is,jjdd,jx,jy,jz,isize,im;
  int ni,nj,nk,nflop,*jbase,*jsize,*njj;
  vec3<int> boxIndex3D;
  const int offsetStride = 2*maxM2LInteraction+1;
  double tic,toc,flops,t[10],boxSize,op=0;
  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  boxSize = rootBoxSize/(1 << numLevel);

  hostConstantSize=sizeof(float)*4;
  hostOffsetSize=sizeof(int)*targetBufferSize/threadsPerBlockTypeB*offsetStride;
  hostLnmTargetSize=sizeof(float)*2*targetBufferSize;
  hostMnmSourceSize=sizeof(float)*2*sourceBufferSize;
  hostYnmSize=sizeof(float)*2*numExpansion2;
  hostDnmSize=sizeof(float)*2*DnmSize*2*numRelativeBox;

  hostConstant=(float *)malloc(hostConstantSize);
  hostOffset=(int *)malloc(hostOffsetSize);
  hostLnmTarget=(float *)malloc(hostLnmTargetSize);
  hostMnmSource=(float *)malloc(hostMnmSourceSize);
  hostYnm=(float *)malloc(hostYnmSize);
  hostDnm=(float *)malloc(hostDnmSize);
  jbase = new int [numBoxIndexLeaf];
  jsize = new int [numBoxIndexLeaf];
  njj = new int [numBoxIndexLeaf];

  hostConstant[0]=(float) boxSize;
  hostConstant[1]=0;
  hostConstant[2]=0;
  hostConstant[3]=0;

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }
  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostLnmTargetSize>deviceLnmTargetSize) {
    if(deviceLnmTargetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceLnmTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceLnmTarget,hostLnmTargetSize));
    deviceLnmTargetSize=hostLnmTargetSize;
  }
  if (hostMnmSourceSize>deviceMnmSourceSize) {
    if(deviceMnmSourceSize!=0) CUDA_SAFE_CALL(cudaFree(deviceMnmSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceMnmSource,hostMnmSourceSize));
    deviceMnmSourceSize=hostMnmSourceSize;
  }
  if (hostYnmSize>deviceYnmSize) {
    if(deviceYnmSize!=0) CUDA_SAFE_CALL(cudaFree(deviceYnm));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceYnm,hostYnmSize));
    deviceYnmSize=hostYnmSize;
  }
  if (hostDnmSize>deviceDnmSize) {
    if(deviceDnmSize!=0) CUDA_SAFE_CALL(cudaFree(deviceDnm));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceDnm,hostDnmSize));
    deviceDnmSize=hostDnmSize;
  }
  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  for( m=0; m<numExpansions; m++ ) {
    for( n=m; n<numExpansions; n++ ) {
      npm = n*n+n+m;
      nmm = n*n+n-m;
      hostYnm[2*npm+0] = std::real(Ynm[npm]);
      hostYnm[2*nmm+0] = std::real(Ynm[nmm]);
      hostYnm[2*npm+1] = std::imag(Ynm[npm]);
      hostYnm[2*nmm+1] = std::imag(Ynm[nmm]);
    }
  }
  for( je=0; je<2*numRelativeBox; je++ ) {
    for( n=0; n<numExpansions; n++ ) {
      for( m=0; m<=n; m++ ) {
        for( k=-n; k<=n; k++ ) {
          nk = n*(n+1)+k;
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k+je*DnmSize;
          hostDnm[2*nmk+0] = std::real(Dnm[je][m][nk]);
          hostDnm[2*nmk+1] = std::imag(Dnm[je][m][nk]);
        }
      }
    }
  }

  if( numLevel == 2 ) {
    for( i=0; i<numBoxIndex; i++ ) {
      for( j=0; j<numCoefficients; j++ ) {
        Lnm[i][j] = 0;
      }
    }
  }

  ni = 0;
  nj = 0;
  ncall = 0;
  boxOffsetStart[0] = 0;
  for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
  for( ii=0; ii<numBoxIndex; ii++ ) {
    ni += ((numCoefficients+threadsPerBlockTypeB)/threadsPerBlockTypeB)*threadsPerBlockTypeB;
    if( numInteraction[ii] != 0 ) {
      for( ij=0; ij<numInteraction[ii]; ij++ ) {
        jj = interactionList[ii][ij];
        if( njj[jj] == 0 ) {
          nj += numCoefficients;
          njj[jj] = 1;
        }
      }
    }
    if ( ni > targetBufferSize || nj > sourceBufferSize ) {
      boxOffsetEnd[ncall] = ii-1;
      ncall++;
      boxOffsetStart[ncall] = ii;
      ni = ((numCoefficients+threadsPerBlockTypeB)/threadsPerBlockTypeB)*threadsPerBlockTypeB;
      nj = 0;
      for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
      for( ij=0; ij<numInteraction[ii]; ij++ ) {
        jj = interactionList[ii][ij];
        nj += numCoefficients;
        njj[jj] = 1;
      }
    }
  }
  boxOffsetEnd[ncall] = numBoxIndex-1;
  if(numBoxIndex != 0) ncall++;

  for( icall=0; icall<ncall; icall++ ) {
    iblok = 0;
    jc = 0;
    jjd = 0;
    for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
    for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
      for( ij=0; ij<numInteraction[ii]; ij++ ) {
        jj = interactionList[ii][ij];
        jb = jj+levelOffset[numLevel-1];
        if( njj[jj] == 0 ) {
          jbase[jjd] = jc;
          for( j=0; j<numCoefficients; j++ ) {
            hostMnmSource[2*jc+0] = std::real(Mnm[jb][j]);
            hostMnmSource[2*jc+1] = std::imag(Mnm[jb][j]);
            jc++;
          }
          jsize[jjd] = jc-jbase[jjd];
          jjd++;
          njj[jj] = jjd;
        }
      }
      ib = ii+levelOffset[numLevel-1];
      tree.unmorton(boxIndexFull[ib],boxIndex3D);
      ix = boxIndex3D.x;
      iy = boxIndex3D.y;
      iz = boxIndex3D.z;
      isize = numCoefficients;
      for( is=0; is<isize; is+=threadsPerBlockTypeB ) {
        hostOffset[iblok*offsetStride] = numInteraction[ii];
        for( ij=0; ij<numInteraction[ii]; ij++ ) {
          jj = interactionList[ii][ij];
          jbd = jj+levelOffset[numLevel-1];
          jjdd = njj[jj]-1;
          tree.unmorton(boxIndexFull[jbd],boxIndex3D);
          jx = boxIndex3D.x;
          jy = boxIndex3D.y;
          jz = boxIndex3D.z;
          boxIndex3D.x = (ix-jx)+3;
          boxIndex3D.y = (iy-jy)+3;
          boxIndex3D.z = (iz-jz)+3;
          tree.morton1(boxIndex3D,je,3);
          hostOffset[iblok*offsetStride+2*ij+1] = jbase[jjdd];
          hostOffset[iblok*offsetStride+2*ij+2] = je+1;
          op += (double) threadsPerBlockTypeB*jsize[jjdd];
        }
        iblok++;
      }
    }

    toc=tic;
    tic=get_gpu_time();
    t[0]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceConstant,hostConstant,hostConstantSize));
    CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceMnmSource,hostMnmSource,hostMnmSourceSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceYnm,hostYnm,hostYnmSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceDnm,hostDnm,hostDnmSize,cudaMemcpyHostToDevice));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    dim3 block(threadsPerBlockTypeB);
    dim3 grid(iblok);
    m2l_kernel<<< grid, block >>>(deviceOffset,deviceLnmTarget,deviceMnmSource,deviceYnm,deviceDnm);
    CUT_CHECK_ERROR("Kernel execution failed");
    nflop = 48;

    toc=tic;
    tic=get_gpu_time();
    t[3]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpy(hostLnmTarget,deviceLnmTarget,hostLnmTargetSize,cudaMemcpyDeviceToHost));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    iblok = 0;
    for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
      isize = numCoefficients;
      for( is=0; is<isize; is+=threadsPerBlockTypeB ) {
        for( i=0; i<std::min(isize-is,threadsPerBlockTypeB); i++ ) {
          im = iblok*threadsPerBlockTypeB+i;
          Lnm[ii][is+i] += std::complex<double>(hostLnmTarget[2*im+0],hostLnmTarget[2*im+1]);
        }
        iblok++;
      }
    }
  }
  for( jj=0; jj<numBoxIndex; jj++ ) {
    jb = jj+levelOffset[numLevel-1];
    for( j=0; j<numCoefficients; j++ ) {
      Mnm[jb][j] = 0;
    }
  }

  free(hostConstant);
  free(hostOffset);
  free(hostLnmTarget);
  free(hostMnmSource);
  free(hostYnm);
  free(hostDnm);
  delete[] jbase;
  delete[] jsize;
  delete[] njj;

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("m2l cudaMalloc : %f s\n",t[1]);
//  printf("m2l cudaMemcpy : %f s\n",t[2]);
//  printf("m2l cudaKernel : %f s\n",t[3]);
//  printf("m2l other      : %f s\n",t[0]);
//  printf("m2l flops      : %f G\n",flops/1e9);
  tic=flops;
}

// l2l
void FmmKernel::l2l(int numBoxIndex, int numLevel) {
  int m,n,npm,nmm,je,k,nmk,numBoxIndexOld,ii,i;
  int ncall,icall,iblok,ic,nfip,nfic,ib,jbase,jsize,im;
  int ni,nk,nflop;
  vec3<int> boxIndex3D;
  const int offsetStride = 3;
  double tic,toc,flops,t[10],boxSize,op=0;
  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  boxSize = rootBoxSize/(1 << numLevel);

  hostConstantSize=sizeof(float)*4;
  hostOffsetSize=sizeof(int)*targetBufferSize/threadsPerBlockTypeB*offsetStride;
  hostLnmTargetSize=sizeof(float)*2*targetBufferSize;
  hostLnmSourceSize=sizeof(float)*2*sourceBufferSize;
  hostYnmSize=sizeof(float)*2*numExpansion2;
  hostDnmSize=sizeof(float)*2*DnmSize*2*numRelativeBox;

  hostConstant=(float *)malloc(hostConstantSize);
  hostOffset=(int *)malloc(hostOffsetSize);
  hostLnmTarget=(float *)malloc(hostLnmTargetSize);
  hostLnmSource=(float *)malloc(hostLnmSourceSize);
  hostYnm=(float *)malloc(hostYnmSize);
  hostDnm=(float *)malloc(hostDnmSize);

  hostConstant[0]=(float) boxSize;
  hostConstant[1]=0;
  hostConstant[2]=0;
  hostConstant[3]=0;

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }
  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostLnmTargetSize>deviceLnmTargetSize) {
    if(deviceLnmTargetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceLnmTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceLnmTarget,hostLnmTargetSize));
    deviceLnmTargetSize=hostLnmTargetSize;
  }
  if (hostLnmSourceSize>deviceLnmSourceSize) {
    if(deviceLnmSourceSize!=0) CUDA_SAFE_CALL(cudaFree(deviceLnmSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceLnmSource,hostLnmSourceSize));
    deviceLnmSourceSize=hostLnmSourceSize;
  }
  if (hostYnmSize>deviceYnmSize) {
    if(deviceYnmSize!=0) CUDA_SAFE_CALL(cudaFree(deviceYnm));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceYnm,hostYnmSize));
    deviceYnmSize=hostYnmSize;
  }
  if (hostDnmSize>deviceDnmSize) {
    if(deviceDnmSize!=0) CUDA_SAFE_CALL(cudaFree(deviceDnm));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceDnm,hostDnmSize));
    deviceDnmSize=hostDnmSize;
  }
  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  for( m=0; m<numExpansions; m++ ) {
    for( n=m; n<numExpansions; n++ ) {
      npm = n*n+n+m;
      nmm = n*n+n-m;
      hostYnm[2*npm+0] = std::real(Ynm[npm]);
      hostYnm[2*nmm+0] = std::real(Ynm[nmm]);
      hostYnm[2*npm+1] = std::imag(Ynm[npm]);
      hostYnm[2*nmm+1] = std::imag(Ynm[nmm]);
    }
  }
  for( je=0; je<2*numRelativeBox; je++ ) {
    for( n=0; n<numExpansions; n++ ) {
      for( m=0; m<=n; m++ ) {
        for( k=-n; k<=n; k++ ) {
          nk = n*(n+1)+k;
          nmk = (4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k+je*DnmSize;
          hostDnm[2*nmk+0] = std::real(Dnm[je][m][nk]);
          hostDnm[2*nmk+1] = std::imag(Dnm[je][m][nk]);
        }
      }
    }
  }

  int nbc,neo[numBoxIndexFull];
  nbc = -1;
  numBoxIndexOld = 0;
  for( i=0; i<numBoxIndexFull; i++ ) neo[i] = -1;
  for( ii=0; ii<numBoxIndex; ii++ ) {
    ib = ii+levelOffset[numLevel-1];
    if( nbc != boxIndexFull[ib]/8 ) {
      nbc = boxIndexFull[ib]/8;
      neo[nbc] = numBoxIndexOld;
      numBoxIndexOld++;
    }
  }

  numBoxIndexOld = numBoxIndex;
  if( numBoxIndexOld < 8 ) numBoxIndexOld = 8;
  for( ii=0; ii<numBoxIndexOld; ii++ ) {
    for( i=0; i<numCoefficients; i++ ) {
      LnmOld[ii][i] = Lnm[ii][i];
    }
  }

  ni = 0;
  ncall = 0;
  boxOffsetStart[0] = 0;
  for( ii=0; ii<numBoxIndex; ii++ ) {
    ni += threadsPerBlockTypeB;
    if( ni > sourceBufferSize ) {
      boxOffsetEnd[ncall] = ii-1;
      ncall++;
      boxOffsetStart[ncall] = ii;
      ni = threadsPerBlockTypeB;
    }
  }
  boxOffsetEnd[ncall] = numBoxIndex-1;
  if(numBoxIndex != 0) ncall++;

  for( icall=0; icall<ncall; icall++ ) {
    iblok = 0;
    ic = 0;
    for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
      ib = ii+levelOffset[numLevel-1];
      nfip = boxIndexFull[ib]/8;
      nfic = boxIndexFull[ib]%8;
      tree.unmorton(nfic,boxIndex3D);
      boxIndex3D.x = boxIndex3D.x*2+2;
      boxIndex3D.y = boxIndex3D.y*2+2;
      boxIndex3D.z = boxIndex3D.z*2+2;
      tree.morton1(boxIndex3D,je,3);
      ib = neo[nfip];
      jbase = ic;
      for( i=0; i<numCoefficients; i++ ) {
        hostLnmSource[2*ic+0] = std::real(LnmOld[ib][i]);
        hostLnmSource[2*ic+1] = std::imag(LnmOld[ib][i]);
        ic++;
      }
      jsize = ic-jbase;
      hostOffset[iblok*offsetStride] = 1;
      hostOffset[iblok*offsetStride+1] = jbase;
      hostOffset[iblok*offsetStride+2] = je+1;
      op += threadsPerBlockTypeB*jsize;
      iblok++;
    }

    toc=tic;
    tic=get_gpu_time();
    t[0]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceConstant,hostConstant,hostConstantSize));
    CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceLnmSource,hostLnmSource,hostLnmSourceSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceYnm,hostYnm,hostYnmSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceDnm,hostDnm,hostDnmSize,cudaMemcpyHostToDevice));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    dim3 block(threadsPerBlockTypeB);
    dim3 grid(iblok);
    l2l_kernel<<< grid, block >>>(deviceOffset,deviceLnmTarget,deviceLnmSource,deviceYnm,deviceDnm);
    CUT_CHECK_ERROR("Kernel execution failed");
    nflop = 48;

    toc=tic;
    tic=get_gpu_time();
    t[3]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpy(hostLnmTarget,deviceLnmTarget,hostLnmTargetSize,cudaMemcpyDeviceToHost));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    iblok = 0;
    for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
      for( i=0; i<numCoefficients; i++ ) {
        im = iblok*threadsPerBlockTypeB+i;
        Lnm[ii][i] = std::complex<double>(hostLnmTarget[2*im+0],hostLnmTarget[2*im+1]);
      }
      iblok++;
    }
  }

  free(hostConstant);
  free(hostOffset);
  free(hostLnmTarget);
  free(hostLnmSource);
  free(hostYnm);
  free(hostDnm);

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("l2l cudaMalloc : %f s\n",t[1]);
//  printf("l2l cudaMemcpy : %f s\n",t[2]);
//  printf("l2l cudaKernel : %f s\n",t[3]);
//  printf("l2l other      : %f s\n",t[0]);
//  printf("l2l flops      : %f G\n",flops/1e9);
  tic=flops;
}

// l2p
void FmmKernel::l2p(int numBoxIndex) {
  int ncall,ii,icall,iblok,jc,jbase,j,jsize,ibase,isize,is,i,im;
  int ni,nj,nflop;
  vec3<int> boxIndex3D;
  const int offsetStride = 5;
  double tic,toc,flops,t[10],boxSize,op=0;
  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  boxSize = rootBoxSize/(1 << maxLevel);

  hostConstantSize=sizeof(float)*4;
  hostOffsetSize=sizeof(int)*targetBufferSize/threadsPerBlockTypeB*offsetStride;
  hostPosTargetSize=sizeof(float3)*targetBufferSize;
  hostLnmSourceSize=sizeof(float)*2*sourceBufferSize;
  hostAccelSize=sizeof(float3)*targetBufferSize;

  hostConstant=(float *)malloc(hostConstantSize);
  hostOffset=(int *)malloc(hostOffsetSize);
  hostPosTarget=(float3 *)malloc(hostPosTargetSize);
  hostLnmSource=(float *)malloc(hostLnmSourceSize);
  hostAccel=(float3 *)malloc(hostAccelSize);

  hostConstant[0]=(float) boxSize;
  hostConstant[1]=(float) boxMin.x;
  hostConstant[2]=(float) boxMin.y;
  hostConstant[3]=(float) boxMin.z;

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }
  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostPosTargetSize>devicePosTargetSize) {
    if(devicePosTargetSize!=0) CUDA_SAFE_CALL(cudaFree(devicePosTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devicePosTarget,hostPosTargetSize));
    devicePosTargetSize=hostPosTargetSize;
  }
  if (hostLnmSourceSize>deviceLnmSourceSize) {
    if(deviceLnmSourceSize!=0) CUDA_SAFE_CALL(cudaFree(deviceLnmSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceLnmSource,hostLnmSourceSize));
    deviceLnmSourceSize=hostLnmSourceSize;
  }
  if (hostAccelSize>deviceAccelSize) {
    if(deviceAccelSize!=0) CUDA_SAFE_CALL(cudaFree(deviceAccel));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceAccel,hostAccelSize));
    deviceAccelSize=hostAccelSize;
  }
  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  ni = 0;
  nj = 0;
  ncall = 0;
  boxOffsetStart[0] = 0;
  for( ii=0; ii<numBoxIndex; ii++ ) {
    ni += ((particleOffset[1][ii]-particleOffset[0][ii]+threadsPerBlockTypeB)
           /threadsPerBlockTypeB+1)*threadsPerBlockTypeB;
    nj += numCoefficients;
    if( ni > targetBufferSize || nj > sourceBufferSize ) {
      boxOffsetEnd[ncall] = ii-1;
      ncall++;
      boxOffsetStart[ncall] = ii;
      ni = ((particleOffset[1][ii]-particleOffset[0][ii]+threadsPerBlockTypeB)
            /threadsPerBlockTypeB+1)*threadsPerBlockTypeB;
      nj = numCoefficients;
    }
  }
  boxOffsetEnd[ncall] = numBoxIndex-1;
  if(numBoxIndex != 0) ncall++;

  for( icall=0; icall<ncall; icall++ ) {
    iblok = 0;
    jc = 0;
    for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
      jbase = jc;
      for( j=0; j<numCoefficients; j++ ) {
        hostLnmSource[2*jc+0] = std::real(Lnm[ii][j]);
        hostLnmSource[2*jc+1] = std::imag(Lnm[ii][j]);
        jc++;
      }
      jsize = jc-jbase;
      ibase = particleOffset[0][ii];
      isize = particleOffset[1][ii]-ibase+1;
      for( is=0; is<isize; is+=threadsPerBlockTypeB ) {
        for( i=0; i<std::min(isize-is,threadsPerBlockTypeB); i++ ) {
          im = iblok*threadsPerBlockTypeB+i;
          hostPosTarget[im] = *(float3*) &bodyPos[ibase+is+i];
        }
        for( i=isize-is; i<threadsPerBlockTypeB; i++ ) {
          im = iblok*threadsPerBlockTypeB+i;
          hostPosTarget[im].x = 0;
          hostPosTarget[im].y = 0;
          hostPosTarget[im].z = 0;
        }
        tree.unmorton(boxIndexFull[ii],boxIndex3D);
        hostOffset[iblok*offsetStride] = jbase;
        hostOffset[iblok*offsetStride+1] = jsize;
        hostOffset[iblok*offsetStride+2] = boxIndex3D.x;
        hostOffset[iblok*offsetStride+3] = boxIndex3D.y;
        hostOffset[iblok*offsetStride+4] = boxIndex3D.z;
        op += threadsPerBlockTypeB*jsize;
        iblok++;
      }
    }

    toc=tic;
    tic=get_gpu_time();
    t[0]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceConstant,hostConstant,hostConstantSize));
    CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(devicePosTarget,hostPosTarget,hostPosTargetSize,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(deviceLnmSource,hostLnmSource,hostLnmSourceSize,cudaMemcpyHostToDevice));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    dim3 block(threadsPerBlockTypeB);
    dim3 grid(iblok);
    l2p_kernel<<< grid, block >>>(deviceOffset,devicePosTarget,deviceLnmSource,deviceAccel);
    CUT_CHECK_ERROR("Kernel execution failed");
    nflop = 56;

    toc=tic;
    tic=get_gpu_time();
    t[3]+=tic-toc;
    CUDA_SAFE_CALL(cudaMemcpy(hostAccel,deviceAccel,hostAccelSize,cudaMemcpyDeviceToHost));
    toc=tic;
    tic=get_gpu_time();
    t[2]+=tic-toc;

    iblok = 0;
    for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
      ibase = particleOffset[0][ii];
      isize = particleOffset[1][ii]-ibase+1;
      for( is=0; is<isize; is+=threadsPerBlockTypeB ) {
        for( i=0; i<std::min(isize-is,threadsPerBlockTypeB); i++ ) {
          im = iblok*threadsPerBlockTypeB+i;
          bodyAccel[ibase+is+i].x += inv4PI*hostAccel[im].x;
          bodyAccel[ibase+is+i].y += inv4PI*hostAccel[im].y;
          bodyAccel[ibase+is+i].z += inv4PI*hostAccel[im].z;
        }
        iblok++;
      }
    }
  }

  free(hostConstant);
  free(hostOffset);
  free(hostPosTarget);
  free(hostLnmSource);
  free(hostAccel);

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("l2p cudaMalloc : %f s\n",t[1]);
//  printf("l2p cudaMemcpy : %f s\n",t[2]);
//  printf("l2p cudaKernel : %f s\n",t[3]);
//  printf("l2p other      : %f s\n",t[0]);
//  printf("l2p flops      : %f G\n",flops/1e9);
  tic=flops;
}

// m2p
void FmmKernel::m2p(int numBoxIndex, int numLevel) {
  int nicall,jc,jj,ii,njd,ij,icall,jcall,iblok,im,jjd,jb,j,ibase,isize,is,i,ijc,jjdd;
  int ni,nj,nflop,*jbase,*jsize,*njcall,*njj,**interactionListOffsetStart,**interactionListOffsetEnd;
  vec3<int> boxIndex3D;
  const int offsetStride = 4*maxM2LInteraction+1;
  double tic,toc,flops,t[10],boxSize,op=0;

  for(i=0;i<10;i++) t[i]=0;
  tic=get_gpu_time();

  boxSize = rootBoxSize/(1 << numLevel);

  hostConstantSize=sizeof(float)*4;
  hostOffsetSize=sizeof(int)*targetBufferSize/threadsPerBlockTypeB*offsetStride;
  hostPosTargetSize=sizeof(float3)*targetBufferSize;
  hostMnmSourceSize=sizeof(float)*2*sourceBufferSize;
  hostAccelSize=sizeof(float3)*targetBufferSize;

  hostConstant=(float *)malloc(hostConstantSize);
  hostOffset=(int *)malloc(hostOffsetSize);
  hostPosTarget=(float3 *)malloc(hostPosTargetSize);
  hostMnmSource=(float *)malloc(hostMnmSourceSize);
  hostAccel=(float3 *)malloc(hostAccelSize);

  interactionListOffsetStart = new int* [maxM2LInteraction];
  for( i=0; i<maxM2LInteraction; i++ ) interactionListOffsetStart[i] = new int [numBoxIndexLeaf];
  interactionListOffsetEnd = new int* [maxM2LInteraction];
  for( i=0; i<maxM2LInteraction; i++ ) interactionListOffsetEnd[i] = new int [numBoxIndexLeaf];
  jbase = new int [numBoxIndexLeaf];
  jsize = new int [numBoxIndexLeaf];
  njcall = new int [numBoxIndexLeaf];
  njj = new int [numBoxIndexLeaf];

  hostConstant[0]=(float) boxSize;
  hostConstant[1]=(float) boxMin.x;
  hostConstant[2]=(float) boxMin.y;
  hostConstant[3]=(float) boxMin.z;

  if (is_set==0) {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    is_set=1;
  }
  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  if (hostOffsetSize>deviceOffsetSize) {
    if(deviceOffsetSize!=0) CUDA_SAFE_CALL(cudaFree(deviceOffset));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceOffset,hostOffsetSize));
    deviceOffsetSize=hostOffsetSize;
  }
  if (hostPosTargetSize>devicePosTargetSize) {
    if(devicePosTargetSize!=0) CUDA_SAFE_CALL(cudaFree(devicePosTarget));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devicePosTarget,hostPosTargetSize));
    devicePosTargetSize=hostPosTargetSize;
  }
  if (hostMnmSourceSize>deviceMnmSourceSize) {
    if(deviceMnmSourceSize!=0) CUDA_SAFE_CALL(cudaFree(deviceMnmSource));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceMnmSource,hostMnmSourceSize));
    deviceMnmSourceSize=hostMnmSourceSize;
  }
  if (hostAccelSize>deviceAccelSize) {
    if(deviceAccelSize!=0) CUDA_SAFE_CALL(cudaFree(deviceAccel));
    CUDA_SAFE_CALL(cudaMalloc((void**)&deviceAccel,hostAccelSize));
    deviceAccelSize=hostAccelSize;
  }

  toc=tic;
  tic=get_gpu_time();
  t[1]+=tic-toc;

  ni = 0;
  nj = 0;
  nicall = 0;
  boxOffsetStart[0] = 0;
  jc = 0;
  for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
  for( ii=0; ii<numBoxIndex; ii++ ) {
    if( numInteraction[ii] != 0 ) {
      njd = 0;
      jc = 0;
      interactionListOffsetStart[0][ii] = 0;
      for( ij=0; ij<numInteraction[ii]; ij++ ) {
        jj = interactionList[ii][ij];
        if( njj[jj] == 0 ) {
          nj += numCoefficients;
          njj[jj] = 1;
        }
        njd += numCoefficients;
        if( njd > sourceBufferSize ) {
          interactionListOffsetEnd[jc][ii] = ij-1;
          jc++;
          interactionListOffsetStart[jc][ii] = ij;
          njd = numCoefficients;
        }
      }
      interactionListOffsetEnd[jc][ii] = numInteraction[ii]-1;
      ni += ((particleOffset[1][ii]-particleOffset[0][ii]+threadsPerBlockTypeB)/threadsPerBlockTypeB+1)
            *threadsPerBlockTypeB;
      if( jc != 0 ) {
        if( ii > boxOffsetStart[nicall] ) {
          njcall[nicall] = 1;
          boxOffsetEnd[nicall] = ii-1;
          nicall++;
          assert( nicall < numBoxIndexLeaf );
          boxOffsetStart[nicall] = ii;
          for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
        }
        if( ii != numBoxIndex ) {
          njcall[nicall] = jc+1;
          boxOffsetEnd[nicall] = ii;
          nicall++;
          assert( nicall < numBoxIndexLeaf );
          boxOffsetStart[nicall] = ii+1;
          for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
          ni = 0;
          nj = 0;
        }
      } else if ( ni > targetBufferSize || nj > sourceBufferSize ) {
        njcall[nicall] = jc+1;
        boxOffsetEnd[nicall] = ii-1;
        nicall++;
        assert( nicall < numBoxIndexLeaf );
        boxOffsetStart[nicall] = ii;
        for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
        ni = ((particleOffset[1][ii]-particleOffset[0][ii]+threadsPerBlockTypeB)/threadsPerBlockTypeB+1)
             *threadsPerBlockTypeB;
        nj = 0;
        for( ij=0; ij<numInteraction[ii]; ij++ ) {
          jj = interactionList[ii][ij];
          nj += numCoefficients;
          njj[jj] = 1;
        }
      }
    }
  }
  njcall[nicall] = jc+1;
  boxOffsetEnd[nicall] = numBoxIndex-1;
  if(numBoxIndex != 0) nicall++;

  for( icall=0; icall<nicall; icall++ ) {
    for( jcall=0; jcall<njcall[icall]; jcall++ ) {
      iblok = 0;
      jc = 0;
      jjd = 0;
      for( jj=0; jj<numBoxIndexLeaf; jj++ ) njj[jj] = 0;
      for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
        if( numInteraction[ii] != 0 ) {
          for( ij=interactionListOffsetStart[jcall][ii]; ij<=interactionListOffsetEnd[jcall][ii]; ij++ ) {
            jj = interactionList[ii][ij];
            jb = jj+levelOffset[numLevel-1];
            if( njj[jj] == 0 ) {
              jbase[jjd] = jc;
              for( j=0; j<numCoefficients; j++ ) {
                hostMnmSource[2*jc+0] = std::real(Mnm[jb][j]);
                hostMnmSource[2*jc+1] = std::imag(Mnm[jb][j]);
                jc++;
              }
              jsize[jjd] = jc-jbase[jjd];
              jjd++;
              njj[jj] = jjd;
            }
          }
          ibase = particleOffset[0][ii];
          isize = particleOffset[1][ii]-ibase+1;
          for( is=0; is<isize; is+=threadsPerBlockTypeB ) {
            for( i=0; i<std::min(isize-is,threadsPerBlockTypeB); i++ ) {
              im = iblok*threadsPerBlockTypeB+i;
              hostPosTarget[im] = *(float3*) &bodyPos[ibase+is+i];
            }
            for( i=isize-is; i<threadsPerBlockTypeB; i++ ) {
              im = iblok*threadsPerBlockTypeB+i;
              hostPosTarget[im].x = 0;
              hostPosTarget[im].y = 0;
              hostPosTarget[im].z = 0;
            }
            hostOffset[iblok*offsetStride] = interactionListOffsetEnd[jcall][ii]
                                            -interactionListOffsetStart[jcall][ii]+1;
            ijc = 0;
            for( ij=interactionListOffsetStart[jcall][ii]; ij<=interactionListOffsetEnd[jcall][ii]; ij++ ) {
              jj = interactionList[ii][ij];
              jb = jj+levelOffset[numLevel-1];
              if( njj[jj] != 0 ) {
                jjdd = njj[jj]-1;
                tree.unmorton(boxIndexFull[jb],boxIndex3D);
                hostOffset[iblok*offsetStride+4*ijc+1] = jbase[jjdd];
                hostOffset[iblok*offsetStride+4*ijc+2] = boxIndex3D.x;
                hostOffset[iblok*offsetStride+4*ijc+3] = boxIndex3D.y;
                hostOffset[iblok*offsetStride+4*ijc+4] = boxIndex3D.z;
                op += (double) threadsPerBlockTypeB*jsize[jjdd];
                ijc++;
              }
            }
            iblok++;
          }
        }
      }

      if( iblok != 0 ) {
        toc=tic;
        tic=get_gpu_time();
        t[0]+=tic-toc;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceConstant,hostConstant,hostConstantSize));
        CUDA_SAFE_CALL(cudaMemcpy(deviceOffset,hostOffset,hostOffsetSize,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(devicePosTarget,hostPosTarget,hostPosTargetSize,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(deviceMnmSource,hostMnmSource,hostMnmSourceSize,cudaMemcpyHostToDevice));
        toc=tic;
        tic=get_gpu_time();
        t[2]+=tic-toc;

        dim3 block(threadsPerBlockTypeB);
        dim3 grid(iblok);
        m2p_kernel<<< grid, block >>>(deviceOffset,devicePosTarget,deviceMnmSource,deviceAccel);
        CUT_CHECK_ERROR("Kernel execution failed");
        nflop = 56;

        toc=tic;
        tic=get_gpu_time();
        t[3]+=tic-toc;
        CUDA_SAFE_CALL(cudaMemcpy(hostAccel,deviceAccel,hostAccelSize,cudaMemcpyDeviceToHost));
        toc=tic;
        tic=get_gpu_time();
        t[2]+=tic-toc;

      }

      iblok = 0;
      for( ii=boxOffsetStart[icall]; ii<=boxOffsetEnd[icall]; ii++ ) {
        if( numInteraction[ii] != 0 ) {
          ibase = particleOffset[0][ii];
          isize = particleOffset[1][ii]-ibase+1;
          for( is=0; is<isize; is+=threadsPerBlockTypeB ) {
            for( i=0; i<std::min(isize-is,threadsPerBlockTypeB); i++ ) {
              im = iblok*threadsPerBlockTypeB+i;
              bodyAccel[ibase+is+i].x += inv4PI*hostAccel[im].x;
              bodyAccel[ibase+is+i].y += inv4PI*hostAccel[im].y;
              bodyAccel[ibase+is+i].z += inv4PI*hostAccel[im].z;
            }
            iblok++;
          }
        }
      }
    }
  }
  free(hostOffset);
  free(hostPosTarget);
  free(hostMnmSource);
  free(hostAccel);
  for( i=0; i<maxM2LInteraction; i++ ) delete[] interactionListOffsetStart[i];
  delete[] interactionListOffsetStart;
  for( i=0; i<maxM2LInteraction; i++ ) delete[] interactionListOffsetEnd[i];
  delete[] interactionListOffsetEnd;
  delete[] jbase;
  delete[] jsize;
  delete[] njcall;
  delete[] njj;

  toc=tic;
  tic=get_gpu_time();
  t[0]+=tic-toc;
  for(i=0;i<9;i++) t[9]+=t[i];
  flops=op*((double)nflop)/t[9];
//  printf("m2p cudaMalloc : %f s\n",t[1]);
//  printf("m2p cudaMemcpy : %f s\n",t[2]);
//  printf("m2p cudaKernel : %f s\n",t[3]);
//  printf("m2p other      : %f s\n",t[0]);
//  printf("m2p flops      : %f G\n",flops/1e9);
  tic=flops;
}
