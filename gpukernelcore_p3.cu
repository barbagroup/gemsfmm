#define ODDEVEN(n) ((n & 1 == 1) ? -1 : 1)

__device__ void cart2sph(float& r, float& theta, float& phi, float dx, float dy, float dz)
{
  r=sqrtf(dx*dx+dy*dy+dz*dz)+eps;
  theta=acosf(dz/r);
  if(fabs(dx)+fabs(dy)<eps){
    phi=0;
  }
  else if(fabs(dx)<eps){
    phi=dy/fabs(dy)*M_PI*0.5;
  }
  else if(dx>0){
    phi=atanf(dy/dx);
  }
  else{
    phi=atanf(dy/dx)+M_PI;
  }
}

__device__ float3 p2p_kernel_core(float3 ai, float3 bi, float4 bj)
{
  float3 dist;
  dist.x = bi.x - bj.x;
  dist.y = bi.y - bj.y;
  dist.z = bi.z - bj.z;
  float invDist = rsqrtf(dist.x*dist.x+dist.y*dist.y+dist.z*dist.z+eps);
  float invDistCube = invDist*invDist*invDist;
  float s = bj.w*invDistCube;
  ai.x -= dist.x * s;
  ai.y -= dist.y * s;
  ai.z -= dist.z * s;
  return ai;
}

__global__ void p2p_kernel(int *deviceOffset,float3 *devicePosTarget,float4 *devicePosSource,float3 *deviceAccel)
{
  int jbase,jsize,jblok,numInteraction;
  int j,ij,jj;
  const int threadsPerBlock=threadsPerBlockTypeA;
  const int offsetStride=2*maxP2PInteraction+1;
  float3 posTarget;
  float3 accel = {0.0f, 0.0f, 0.0f};
  __shared__ float4 sharedPosSource[threadsPerBlock];
  posTarget=devicePosTarget[blockIdx.x*threadsPerBlock+threadIdx.x];
  numInteraction=deviceOffset[blockIdx.x*offsetStride];
  for(ij=0;ij<numInteraction;ij++){
    jbase=deviceOffset[blockIdx.x*offsetStride+2*ij+1];
    jsize=deviceOffset[blockIdx.x*offsetStride+2*ij+2];
    jblok=(jsize+threadsPerBlock-1)/threadsPerBlock;
    for(j=0;j<jblok-1;j++){
      sharedPosSource[threadIdx.x]=devicePosSource[jbase+j*threadsPerBlock+threadIdx.x];
      __syncthreads();
#pragma unroll 32
      for(jj=0;jj<threadsPerBlock;jj++){
        accel = p2p_kernel_core(accel,posTarget,sharedPosSource[jj]);
      }
      __syncthreads();
    }
    sharedPosSource[threadIdx.x]=devicePosSource[jbase+j*threadsPerBlock+threadIdx.x];
    __syncthreads();
    for(jj=0;jj<jsize-(j*threadsPerBlock);jj++){
      accel = p2p_kernel_core(accel,posTarget,sharedPosSource[jj]);
    }
    __syncthreads();
  }
  deviceAccel[blockIdx.x*threadsPerBlock+threadIdx.x]=accel;
}

__device__ void p2m_kernel_core(float& MnmReal, float& MnmImag,
                                int nn, int mm, float xi, float yi, float zi, float4 bj)
{
  int m,n,nm;
  float3 dist;
  float rho,alpha,beta;
  float xx,s2,p,pn,p1,p2,fact,ere,eim,rhom,rhon;
  __shared__ float sharedFactorial[2*numExpansions];
  __shared__ float sharedYnm[numExpansion2];
  fact=1.0;
  for(m=0;m<2*numExpansions;m++) {
    sharedFactorial[m]=fact;
    fact=fact*(m+1);
  }
  dist.x=bj.x-xi;
  dist.y=bj.y-yi;
  dist.z=bj.z-zi;
  cart2sph(rho,alpha,beta,dist.x,dist.y,dist.z);
  xx=cosf(alpha);
  s2=sqrtf((1-xx)*(1+xx));
  fact=1;
  pn=1;
  rhom=1;
  for(m=0;m<=mm;m++){
    p=pn;
    nm=m*m+2*m;
    sharedYnm[nm]=rhom*p*rsqrtf(sharedFactorial[2*m]);
    p1=p;
    p=xx*(2*m+1)*p;
    rhom*=rho;
    rhon=rhom;
    for(n=m+1;n<=nn;n++){
      nm=n*n+n+m;
      sharedYnm[nm]=rhon*p*rsqrtf(sharedFactorial[n+m]/sharedFactorial[n-m]);
      p2=p1;
      p1=p;
      p=(xx*(2*n+1)*p1-(n+m)*p2)/(n-m+1);
      rhon*=rho;
    }
    pn=-pn*fact*s2;
    fact=fact+2;
  }
  nm=nn*nn+nn+mm;
  ere=cosf(-mm*beta);
  eim=sinf(-mm*beta);
  MnmReal+=bj.w*sharedYnm[nm]*ere;
  MnmImag+=bj.w*sharedYnm[nm]*eim;
}

__global__ void p2m_kernel(int *deviceOffset,float *deviceMnmTarget,float4 *devicePosSource)
{
  int i,j,k,m,n,jj,jbase,jsize,jblok,nms;
  int nb,mortonIndex3D[3],nd;
  const int threadsPerBlock=threadsPerBlockTypeB;
  const int offsetStride=3;
  float boxSize=deviceConstant[0];
  float3 boxMin = {deviceConstant[1], deviceConstant[2], deviceConstant[3]};
  float3 boxCenter;
  float MnmTarget[2] = {0.0f, 0.0f};
  __shared__ int mg[threadsPerBlock],ng[threadsPerBlock];
  __shared__ float4 sharedPosSource[threadsPerBlock];
  for(i=0;i<threadsPerBlock;i++){
    ng[i]=0;
    mg[i]=0;
  }
  for(n=0;n<numExpansions;n++){
    for(m=0;m<=n;m++){
      nms=n*(n+1)/2+m;
      ng[nms]=n;
      mg[nms]=m;
    }
  }
  __syncthreads();
  nb=deviceOffset[blockIdx.x*offsetStride];
  jbase=deviceOffset[blockIdx.x*offsetStride+1];
  jsize=deviceOffset[blockIdx.x*offsetStride+2];
  for(i=0;i<3;i++) mortonIndex3D[i]=0;
  k=0;
  i=1;
  while(nb!=0){
    j=2-k;
    mortonIndex3D[j]=mortonIndex3D[j]+nb%2*i;
    nb=nb/2;
    j=k+1;
    k=j%3;
    if(k==0) i=i*2;
  }
  nd=mortonIndex3D[0];
  mortonIndex3D[0]=mortonIndex3D[1];
  mortonIndex3D[1]=mortonIndex3D[2];
  mortonIndex3D[2]=nd;
  boxCenter.x=boxMin.x+(mortonIndex3D[0]+0.5)*boxSize;
  boxCenter.y=boxMin.y+(mortonIndex3D[1]+0.5)*boxSize;
  boxCenter.z=boxMin.z+(mortonIndex3D[2]+0.5)*boxSize;
  jblok=(jsize+threadsPerBlock-1)/threadsPerBlock;
  for(j=0;j<jblok-1;j++){
    sharedPosSource[threadIdx.x]=devicePosSource[jbase+j*threadsPerBlock+threadIdx.x];
    __syncthreads();
    for(jj=0;jj<threadsPerBlock;jj++){
      p2m_kernel_core(MnmTarget[0],MnmTarget[1],
                      ng[threadIdx.x],mg[threadIdx.x],boxCenter.x,boxCenter.y,boxCenter.z,
                      sharedPosSource[jj]);
      __syncthreads();
    }
  }
  sharedPosSource[threadIdx.x]=devicePosSource[jbase+j*threadsPerBlock+threadIdx.x];
  __syncthreads();
  for(jj=0;jj<jsize-(j*threadsPerBlock);jj++){
    p2m_kernel_core(MnmTarget[0],MnmTarget[1],
                    ng[threadIdx.x],mg[threadIdx.x],boxCenter.x,boxCenter.y,boxCenter.z,
                    sharedPosSource[jj]);
    __syncthreads();
  }
  for(i=0;i<2;i++) deviceMnmTarget[2*(blockIdx.x*threadsPerBlock+threadIdx.x)+i]=MnmTarget[i];
}

__global__ void m2m_kernel(int *deviceOffset,float *deviceMnmTarget,float *deviceMnmSource,
                           float *deviceYnm,float *deviceDnm)
{
  int i,j,k,m,n,ij,numInteraction,jbase,jblok,je,nms,nmk,nks,jks,jnk;
  const int threadsPerBlock=threadsPerBlockTypeB;
  const int offsetStride=3;
  float boxSize=deviceConstant[0];
  float rho,rhon,CnmReal,CnmImag,DnmReal,DnmImag;
  float sr,ank,ajk,ajn,fnmm,fnpm;
  float tempTarget[2];
  float MnmTarget[2] = {0.0f, 0.0f};
  __shared__ int mg[threadsPerBlock],ng[threadsPerBlock];
  __shared__ float sharedMnmSource[2*threadsPerBlock];
  __shared__ float sharedYnmReal[numExpansion2];
  __shared__ float sharedYnmImag[numExpansion2];
  numInteraction=deviceOffset[blockIdx.x*offsetStride];
  for(i=0;i<threadsPerBlock;i++){
    ng[i]=0;
    mg[i]=0;
  }
  for(n=0;n<numExpansions;n++){
    for(m=0;m<=n;m++){
      nms=n*(n+1)/2+m;
      ng[nms]=n;
      mg[nms]=m;
    }
  }
  jblok=(numExpansion2+threadsPerBlock-1)/threadsPerBlock;
  for(j=0;j<jblok-1;j++){
    sharedYnmReal[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+0];
    sharedYnmImag[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+1];
    __syncthreads();
  }
  if(j*threadsPerBlock+threadIdx.x<numExpansion2){
    sharedYnmReal[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+0];
    sharedYnmImag[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+1];
  }
  __syncthreads();
  for(ij=0;ij<numInteraction;ij++){
    jbase=deviceOffset[blockIdx.x*offsetStride+2*ij+1];
    je=deviceOffset[blockIdx.x*offsetStride+2*ij+2];
    for(i=0;i<2;i++) sharedMnmSource[2*threadIdx.x+i]=deviceMnmSource[2*(jbase+threadIdx.x)+i];
    __syncthreads();
    rho=boxSize*sqrtf(3.0f)/4;
    jbase=(je-1)*DnmSize;
    n=ng[threadIdx.x];
    m=mg[threadIdx.x];
    nms=n*(n+1)/2+m;
    for(i=0;i<2;i++) tempTarget[i]=0;
    for(k=-n;k<0;k++){
      nks=n*(n+1)/2-k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]+=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]-=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    for(k=0;k<=n;k++){
      nks=n*(n+1)/2+k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]-=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    __syncthreads();
    for(i=0;i<2;i++) sharedMnmSource[2*nms+i]=tempTarget[i];
    __syncthreads();
    j=ng[threadIdx.x];
    k=mg[threadIdx.x];
    jks=j*(j+1)/2+k;
    for(i=0;i<2;i++) tempTarget[i]=0;
    fnmm=1.0;
    for(i=0;i<j-k;i++) fnmm=fnmm*(i+1);
    fnpm=1.0;
    for(i=0;i<j+k;i++) fnpm=fnpm*(i+1);
    ajk=ODDEVEN(j)*rsqrt(fnmm*fnpm);
    rhon=1.0;
    for(n=0;n<=j-abs(k);n++){
      nks=(j-n)*(j-n+1)/2+k;
      jnk=n*n+n;
      fnmm=1.0;
      for(i=0;i<j-n-k;i++) fnmm=fnmm*(i+1);
      fnpm=1.0;
      for(i=0;i<j-n+k;i++) fnpm=fnpm*(i+1);
      ank=ODDEVEN(j-n)*rsqrt(fnmm*fnpm);
      fnpm=1.0;
      for(i=0;i<n;i++) fnpm=fnpm*(i+1);
      ajn=ODDEVEN(n)/fnpm;
      sr=ODDEVEN(n)*ank*ajn/ajk;
      CnmReal=sr*sharedYnmReal[jnk]*rhon;
      CnmImag=sr*sharedYnmImag[jnk]*rhon;
      tempTarget[0]+=sharedMnmSource[2*nks+0]*CnmReal;
      tempTarget[0]-=sharedMnmSource[2*nks+1]*CnmImag;
      tempTarget[1]+=sharedMnmSource[2*nks+0]*CnmImag;
      tempTarget[1]+=sharedMnmSource[2*nks+1]*CnmReal;
      rhon*=rho;
    }
    __syncthreads();
    for(i=0;i<2;i++) sharedMnmSource[2*jks+i]=tempTarget[i];
    __syncthreads();
    jbase=(je+numRelativeBox-1)*DnmSize;
    n=ng[threadIdx.x];
    m=mg[threadIdx.x];
    nms=n*(n+1)/2+m;
    for(i=0;i<2;i++) tempTarget[i]=0;
    for(k=-n;k<0;k++){
      nks=n*(n+1)/2-k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]+=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]-=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    for(k=0;k<=n;k++){
      nks=n*(n+1)/2+k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]-=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    for(i=0;i<2;i++) MnmTarget[i]+=tempTarget[i];
    __syncthreads();
  }
  for(i=0;i<2;i++) deviceMnmTarget[2*(blockIdx.x*threadsPerBlock+threadIdx.x)+i]=MnmTarget[i];
}

__global__ void m2l_kernel(int *deviceOffset,float *deviceLnmTarget,float *deviceMnmSource,
                           float *deviceYnm,float *deviceDnm)
{
  int i,j,k,m,n,ij,numInteraction,jbase,jblok,je,nms,nmk,nks,jks,jnk;
  int nb,mortonIndex3D[3],nd;
  const int threadsPerBlock=threadsPerBlockTypeB;
  const int offsetStride=2*maxM2LInteraction+1;
  float boxSize=deviceConstant[0];
  float rho,rhon,CnmReal,CnmImag,DnmReal,DnmImag;
  float sr,ank,ajk,ajn,fnmm,fnpm;
  float tempTarget[2];
  float3 dist;
  __shared__ int mg[threadsPerBlock],ng[threadsPerBlock];
  __shared__ float sharedLnmTarget[6*threadsPerBlock];
  __shared__ float sharedMnmSource[2*threadsPerBlock];
  __shared__ float sharedYnmReal[numExpansion2];
  __shared__ float sharedYnmImag[numExpansion2];
  numInteraction=deviceOffset[blockIdx.x*offsetStride];
  for(i=0;i<threadsPerBlock;i++){
    ng[i]=0;
    mg[i]=0;
  }
  for(n=0;n<numExpansions;n++){
    for(m=0;m<=n;m++){
      nms=n*(n+1)/2+m;
      ng[nms]=n;
      mg[nms]=m;
    }
  }
  jblok=(numExpansion2+threadsPerBlock-1)/threadsPerBlock;
  for(j=0;j<jblok-1;j++){
    sharedYnmReal[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+0];
    sharedYnmImag[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+1];
    __syncthreads();
  }
  if(j*threadsPerBlock+threadIdx.x<numExpansion2){
    sharedYnmReal[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+0];
    sharedYnmImag[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+1];
  }
  __syncthreads();
  for(i=0;i<6;i++) sharedLnmTarget[6*threadIdx.x+i]=0;
  __syncthreads();
  for(ij=0;ij<numInteraction;ij++){
    jbase=deviceOffset[blockIdx.x*offsetStride+2*ij+1];
    je=deviceOffset[blockIdx.x*offsetStride+2*ij+2];
    for(i=0;i<2;i++) sharedMnmSource[2*threadIdx.x+i]=deviceMnmSource[2*(jbase+threadIdx.x)+i];
    __syncthreads();
    for(i=0;i<3;i++) mortonIndex3D[i]=0;
    nb=je-1;
    k=0;
    i=1;
    while(nb!=0){
      j=2-k;
      mortonIndex3D[j]=mortonIndex3D[j]+nb%2*i;
      nb=nb/2;
      j=k+1;
      k=j%3;
      if(k==0) i=i*2;
    }
    nd=mortonIndex3D[0];
    mortonIndex3D[0]=mortonIndex3D[1];
    mortonIndex3D[1]=mortonIndex3D[2];
    mortonIndex3D[2]=nd;
    dist.x=(mortonIndex3D[0]-3)*boxSize;
    dist.y=(mortonIndex3D[1]-3)*boxSize;
    dist.z=(mortonIndex3D[2]-3)*boxSize;
    rho=sqrt(dist.x*dist.x+dist.y*dist.y+dist.z*dist.z)+eps;
    jbase=(je-1)*DnmSize;
    n=ng[threadIdx.x];
    m=mg[threadIdx.x];
    nms=n*(n+1)/2+m;
    for(i=0;i<2;i++) tempTarget[i]=0;
    for(k=-n;k<0;k++){
      nks=n*(n+1)/2-k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]+=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]-=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    for(k=0;k<=n;k++){
      nks=n*(n+1)/2+k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]-=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    __syncthreads();
    for(i=0;i<2;i++) sharedMnmSource[2*nms+i]=tempTarget[i];
    __syncthreads();
    j=ng[threadIdx.x];
    k=mg[threadIdx.x];
    jks=j*(j+1)/2+k;
    for(i=0;i<2;i++) tempTarget[i]=0;
    fnmm=1.0;
    for(i=0;i<j-k;i++) fnmm=fnmm*(i+1);
    fnpm=1.0;
    for(i=0;i<j+k;i++) fnpm=fnpm*(i+1);
    ajk=ODDEVEN(j)*rsqrt(fnmm*fnpm);
    rhon=1.0/pow(rho,j+k+1);
    for(n=abs(k);n<numExpansions;n++){
      nks=n*(n+1)/2+k;
      jnk=(j+n)*(j+n)+j+n;
      fnmm=1.0;
      for(i=0;i<n-k;i++) fnmm=fnmm*(i+1);
      fnpm=1.0;
      for(i=0;i<n+k;i++) fnpm=fnpm*(i+1);
      ank=ODDEVEN(n)*rsqrt(fnmm*fnpm);
      fnpm=1.0;
      for(i=0;i<j+n;i++) fnpm=fnpm*(i+1);
      ajn=ODDEVEN(j+n)/fnpm;
      sr=ODDEVEN(j+k)*ank*ajk/ajn;
      CnmReal=sr*sharedYnmReal[jnk]*rhon;
      CnmImag=sr*sharedYnmImag[jnk]*rhon;
      tempTarget[0]+=sharedMnmSource[2*nks+0]*CnmReal;
      tempTarget[0]-=sharedMnmSource[2*nks+1]*CnmImag;
      tempTarget[1]+=sharedMnmSource[2*nks+0]*CnmImag;
      tempTarget[1]+=sharedMnmSource[2*nks+1]*CnmReal;
      rhon/=rho;
    }
    __syncthreads();
    for(i=0;i<2;i++) sharedMnmSource[2*jks+i]=tempTarget[i];
    __syncthreads();
    jbase=(je+numRelativeBox-1)*DnmSize;
    n=ng[threadIdx.x];
    m=mg[threadIdx.x];
    nms=n*(n+1)/2+m;
    for(i=0;i<2;i++) tempTarget[i]=0;
    for(k=-n;k<0;k++){
      nks=n*(n+1)/2-k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]+=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]-=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    for(k=0;k<=n;k++){
      nks=n*(n+1)/2+k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedMnmSource[2*nks+0];
      tempTarget[0]-=DnmImag*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmReal*sharedMnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedMnmSource[2*nks+0];
    }
    for(i=0;i<2;i++) sharedLnmTarget[6*threadIdx.x+3*i]+=tempTarget[i];
    __syncthreads();
  }
  for(i=0;i<2;i++) deviceLnmTarget[2*(blockIdx.x*threadsPerBlock+threadIdx.x)+i]=sharedLnmTarget[6*threadIdx.x+3*i];
}

__global__ void l2l_kernel(int *deviceOffset,float *deviceLnmTarget,float *deviceLnmSource,
                           float *deviceYnm,float *deviceDnm)
{
  int i,j,k,m,n,ij,numInteraction,jbase,jblok,je,nms,nmk,nks,jks,jnk;
  const int threadsPerBlock=threadsPerBlockTypeB;
  const int offsetStride=3;
  float boxSize=deviceConstant[0];
  float rho,rhon,CnmReal,CnmImag,DnmReal,DnmImag;
  float sr,ank,ajk,ajn,fnmm,fnpm;
  float tempTarget[2];
  float LnmTarget[2] = {0.0f, 0.0f};
  __shared__ int mg[threadsPerBlock],ng[threadsPerBlock];
  __shared__ float sharedLnmSource[2*threadsPerBlock];
  __shared__ float sharedYnmReal[numExpansion2];
  __shared__ float sharedYnmImag[numExpansion2];
  numInteraction=deviceOffset[blockIdx.x*offsetStride];
  for(i=0;i<threadsPerBlock;i++){
    ng[i]=0;
    mg[i]=0;
  }
  for(n=0;n<numExpansions;n++){
    for(m=0;m<=n;m++){
      nms=n*(n+1)/2+m;
      ng[nms]=n;
      mg[nms]=m;
    }
  }
  jblok=(numExpansion2+threadsPerBlock-1)/threadsPerBlock;
  for(j=0;j<jblok-1;j++){
    sharedYnmReal[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+0];
    sharedYnmImag[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+1];
    __syncthreads();
  }
  if(j*threadsPerBlock+threadIdx.x<numExpansion2){
    sharedYnmReal[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+0];
    sharedYnmImag[j*threadsPerBlock+threadIdx.x]=deviceYnm[2*(j*threadsPerBlock+threadIdx.x)+1];
  }
  __syncthreads();
  for(ij=0;ij<numInteraction;ij++){
    jbase=deviceOffset[blockIdx.x*offsetStride+2*ij+1];
    je=deviceOffset[blockIdx.x*offsetStride+2*ij+2];
    for(i=0;i<2;i++) sharedLnmSource[2*threadIdx.x+i]=deviceLnmSource[2*(jbase+threadIdx.x)+i];
    __syncthreads();
    rho=boxSize*sqrtf(3.0f)/2;
    jbase=(je-1)*DnmSize;
    n=ng[threadIdx.x];
    m=mg[threadIdx.x];
    nms=n*(n+1)/2+m;
    for(i=0;i<2;i++) tempTarget[i]=0;
    for(k=-n;k<0;k++){
      nks=n*(n+1)/2-k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedLnmSource[2*nks+0];
      tempTarget[0]+=DnmImag*sharedLnmSource[2*nks+1];
      tempTarget[1]-=DnmReal*sharedLnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedLnmSource[2*nks+0];
    }
    for(k=0;k<=n;k++){
      nks=n*(n+1)/2+k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedLnmSource[2*nks+0];
      tempTarget[0]-=DnmImag*sharedLnmSource[2*nks+1];
      tempTarget[1]+=DnmReal*sharedLnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedLnmSource[2*nks+0];
    }
    __syncthreads();
    for(i=0;i<2;i++) sharedLnmSource[2*nms+i]=tempTarget[i];
    __syncthreads();
    j=ng[threadIdx.x];
    k=mg[threadIdx.x];
    jks=j*(j+1)/2+k;
    for(i=0;i<2;i++) tempTarget[i]=0;
    fnmm=1.0;
    for(i=0;i<j-k;i++) fnmm=fnmm*(i+1);
    fnpm=1.0;
    for(i=0;i<j+k;i++) fnpm=fnpm*(i+1);
    ajk=ODDEVEN(j)*rsqrt(fnmm*fnpm);
    rhon=1.0;
    for(n=j;n<numExpansions;n++){
      nks=n*(n+1)/2+k;
      jnk=(n-j)*(n-j)+n-j;
      fnmm=1.0;
      for(i=0;i<n-k;i++) fnmm=fnmm*(i+1);
      fnpm=1.0;
      for(i=0;i<n+k;i++) fnpm=fnpm*(i+1);
      ank=ODDEVEN(n)*rsqrt(fnmm*fnpm);
      fnpm=1.0;
      for(i=0;i<n-j;i++) fnpm=fnpm*(i+1);
      ajn=ODDEVEN(n-j)/fnpm;
      sr=ajn*ajk/ank;
      CnmReal=sr*sharedYnmReal[jnk]*rhon;
      CnmImag=sr*sharedYnmImag[jnk]*rhon;
      tempTarget[0]+=sharedLnmSource[2*nks+0]*CnmReal;
      tempTarget[0]-=sharedLnmSource[2*nks+1]*CnmImag;
      tempTarget[1]+=sharedLnmSource[2*nks+0]*CnmImag;
      tempTarget[1]+=sharedLnmSource[2*nks+1]*CnmReal;
      rhon*=rho;
    }
    __syncthreads();
    for(i=0;i<2;i++) sharedLnmSource[2*jks+i]=tempTarget[i];
    __syncthreads();
    jbase=(je+numRelativeBox-1)*DnmSize;
    n=ng[threadIdx.x];
    m=mg[threadIdx.x];
    nms=n*(n+1)/2+m;
    for(i=0;i<2;i++) tempTarget[i]=0;
    for(k=-n;k<0;k++){
      nks=n*(n+1)/2-k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedLnmSource[2*nks+0];
      tempTarget[0]+=DnmImag*sharedLnmSource[2*nks+1];
      tempTarget[1]-=DnmReal*sharedLnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedLnmSource[2*nks+0];
    }
    for(k=0;k<=n;k++){
      nks=n*(n+1)/2+k;
      nmk=jbase+(4*n*n*n+6*n*n+5*n)/3+m*(2*n+1)+k;
      DnmReal=deviceDnm[2*nmk+0];
      DnmImag=deviceDnm[2*nmk+1];
      tempTarget[0]+=DnmReal*sharedLnmSource[2*nks+0];
      tempTarget[0]-=DnmImag*sharedLnmSource[2*nks+1];
      tempTarget[1]+=DnmReal*sharedLnmSource[2*nks+1];
      tempTarget[1]+=DnmImag*sharedLnmSource[2*nks+0];
    }
    for(i=0;i<2;i++) LnmTarget[i]+=tempTarget[i];
    __syncthreads();
  }
  for(i=0;i<2;i++) deviceLnmTarget[2*(blockIdx.x*threadsPerBlock+threadIdx.x)+i]=LnmTarget[i];
}

__global__ void l2p_kernel(int *deviceOffset,float3 *devicePosTarget,float *deviceLnmSource,float3 *deviceAccel)
{
  int i,m,n,jbase,nms;
  const int threadsPerBlock=threadsPerBlockTypeB;
  const int offsetStride=5;
  float boxSize=deviceConstant[0];
  float3 boxMin = {deviceConstant[1], deviceConstant[2], deviceConstant[3]};
  float3 boxCenter,dist;
  float r,theta,phi;
  float xx,yy,s2,p,pn,p1,p2,fact,ere,eim,rhom,rhon,realj,imagj;
  float accelR,accelTheta,accelPhi;
  float anm,YnmReal,YnmRealTheta;
  float3 accel = {0.0f, 0.0f, 0.0f};
  __shared__ float sharedLnmSource[2*threadsPerBlock];
  __shared__ float sharedFactorial[2*numExpansions];
  fact=1.0;
  for(i=0;i<2*numExpansions;i++) {
    sharedFactorial[i]=fact;
    fact=fact*(i+1);
  }
  jbase=deviceOffset[blockIdx.x*offsetStride];
  boxCenter.x=boxMin.x+(deviceOffset[blockIdx.x*offsetStride+2]+0.5)*boxSize;
  boxCenter.y=boxMin.y+(deviceOffset[blockIdx.x*offsetStride+3]+0.5)*boxSize;
  boxCenter.z=boxMin.z+(deviceOffset[blockIdx.x*offsetStride+4]+0.5)*boxSize;
  for(i=0;i<2;i++) sharedLnmSource[2*threadIdx.x+i]=deviceLnmSource[2*(jbase+threadIdx.x)+i];
  __syncthreads();
  dist.x=devicePosTarget[blockIdx.x*threadsPerBlock+threadIdx.x].x-boxCenter.x;
  dist.y=devicePosTarget[blockIdx.x*threadsPerBlock+threadIdx.x].y-boxCenter.y;
  dist.z=devicePosTarget[blockIdx.x*threadsPerBlock+threadIdx.x].z-boxCenter.z;
  cart2sph(r,theta,phi,dist.x,dist.y,dist.z);
  accelR=0;
  accelTheta=0;
  accelPhi=0;
  xx=cosf(theta);
  yy=sinf(theta);
  if(abs(yy)<eps) yy=1/eps;
  s2=sqrtf((1-xx)*(1+xx));
  fact=1;
  pn=1;
  rhom=1;
  for(m=0;m<numExpansions;m++){
    p=pn;
    nms=m*(m+1)/2+m;
    ere=cosf(m*phi);
    if(m==0) ere=0.5;
    eim=sinf(m*phi);
    anm=rhom*rsqrtf(sharedFactorial[2*m]);
    YnmReal=anm*p;
    p1=p;
    p=xx*(2*m+1)*p;
    YnmRealTheta=anm*(p-(m+1)*xx*p1)/yy;
    realj=ere*sharedLnmSource[2*nms+0]-eim*sharedLnmSource[2*nms+1];
    imagj=eim*sharedLnmSource[2*nms+0]+ere*sharedLnmSource[2*nms+1];
    accelR+=2*m/r*YnmReal*realj;
    accelTheta+=2*YnmRealTheta*realj;
    accelPhi-=2*m*YnmReal*imagj;
    rhom*=r;
    rhon=rhom;
    for(n=m+1;n<numExpansions;n++){
      nms=n*(n+1)/2+m;
      anm=rhon*rsqrtf(sharedFactorial[n+m]/sharedFactorial[n-m]);
      YnmReal=anm*p;
      p2=p1;
      p1=p;
      p=(xx*(2*n+1)*p1-(n+m)*p2)/(n-m+1);
      YnmRealTheta=anm*((n-m+1)*p-(n+1)*xx*p1)/yy;
      realj=ere*sharedLnmSource[2*nms+0]-eim*sharedLnmSource[2*nms+1];
      imagj=eim*sharedLnmSource[2*nms+0]+ere*sharedLnmSource[2*nms+1];
      accelR+=2*n/r*YnmReal*realj;
      accelTheta+=2*YnmRealTheta*realj;
      accelPhi-=2*m*YnmReal*imagj;
      rhon*=r;
    }
    pn=-pn*fact*s2;
    fact=fact+2;
  }
  accel.x+=sinf(theta)*cosf(phi)*accelR+cosf(theta)*cosf(phi)/r*accelTheta-sinf(phi)/r/yy*accelPhi;
  accel.y+=sinf(theta)*sinf(phi)*accelR+cosf(theta)*sinf(phi)/r*accelTheta+cosf(phi)/r/yy*accelPhi;
  accel.z+=cosf(theta)*accelR-sinf(theta)/r*accelTheta;
  deviceAccel[blockIdx.x*threadsPerBlock+threadIdx.x]=accel;
}

__global__ void m2p_kernel(int *deviceOffset,float3 *devicePosTarget,float *deviceMnmSource,float3 *deviceAccel)
{
  int i,m,n,jx,jy,jz,ij,numInteraction,jbase,nms;
  const int threadsPerBlock=threadsPerBlockTypeB;
  const int offsetStride=4*maxM2LInteraction+1;
  float boxSize=deviceConstant[0];
  float3 boxMin = {deviceConstant[1], deviceConstant[2], deviceConstant[3]};
  float3 boxCenter,dist;
  float r,theta,phi,rhom,rhon;
  float xx,yy,s2,p,pn,p1,p2,fact,ere,eim,realj,imagj;
  float accelR,accelTheta,accelPhi;
  float anm,YnmReal,YnmRealTheta;
  float3 accel = {0.0f, 0.0f, 0.0f};
  float3 posTarget;
  __shared__ float sharedMnmSource[2*threadsPerBlock];
  __shared__ float sharedFactorial[2*numExpansions];
  numInteraction=deviceOffset[blockIdx.x*offsetStride];
  fact=1.0;
  for(i=0;i<2*numExpansions;i++) {
    sharedFactorial[i]=fact;
    fact=fact*(i+1);
  }
  posTarget=devicePosTarget[blockIdx.x*threadsPerBlock+threadIdx.x];
  __syncthreads();
  for(ij=0;ij<numInteraction;ij++){
    jbase=deviceOffset[blockIdx.x*offsetStride+4*ij+1];
    jx=deviceOffset[blockIdx.x*offsetStride+4*ij+2];
    jy=deviceOffset[blockIdx.x*offsetStride+4*ij+3];
    jz=deviceOffset[blockIdx.x*offsetStride+4*ij+4];
    boxCenter.x=boxMin.x+(jx+0.5)*boxSize;
    boxCenter.y=boxMin.y+(jy+0.5)*boxSize;
    boxCenter.z=boxMin.z+(jz+0.5)*boxSize;
    for(i=0;i<2;i++) sharedMnmSource[2*threadIdx.x+i]=deviceMnmSource[2*(jbase+threadIdx.x)+i];
    __syncthreads();
    dist.x=posTarget.x-boxCenter.x;
    dist.y=posTarget.y-boxCenter.y;
    dist.z=posTarget.z-boxCenter.z;
    cart2sph(r,theta,phi,dist.x,dist.y,dist.z);
    accelR=0;
    accelTheta=0;
    accelPhi=0;
    xx=cosf(theta);
    yy=sinf(theta);
    if(fabs(yy)<eps) yy=1/eps;
    s2=sqrtf((1-xx)*(1+xx));
    fact=1;
    pn=1;
    rhom=1.0/r;
    for(m=0;m<numExpansions;m++){
      p=pn;
      nms=m*(m+1)/2+m;
      ere=cosf(m*phi);
      if(m==0) ere=0.5;
      eim=sinf(m*phi);
      anm=rhom*rsqrt(sharedFactorial[2*m]);
      YnmReal=anm*p;
      p1=p;
      p=xx*(2*m+1)*p;
      YnmRealTheta=anm*(p-(m+1)*xx*p1)/yy;
      realj=ere*sharedMnmSource[2*nms+0]-eim*sharedMnmSource[2*nms+1];
      imagj=eim*sharedMnmSource[2*nms+0]+ere*sharedMnmSource[2*nms+1];
      accelR-=2*(m+1)/r*YnmReal*realj;
      accelTheta+=2*YnmRealTheta*realj;
      accelPhi-=2*m*YnmReal*imagj;
      rhom/=r;
      rhon=rhom;
      for(n=m+1;n<numExpansions;n++){
        nms=n*(n+1)/2+m;
        anm=rhon*rsqrt(sharedFactorial[n+m]/sharedFactorial[n-m]);
        YnmReal=anm*p;
        p2=p1;
        p1=p;
        p=(xx*(2*n+1)*p1-(n+m)*p2)/(n-m+1);
        YnmRealTheta=anm*((n-m+1)*p-(n+1)*xx*p1)/yy;
        realj=ere*sharedMnmSource[2*nms+0]-eim*sharedMnmSource[2*nms+1];
        imagj=eim*sharedMnmSource[2*nms+0]+ere*sharedMnmSource[2*nms+1];
        accelR-=2*(n+1)/r*YnmReal*realj;
        accelTheta+=2*YnmRealTheta*realj;
        accelPhi-=2*m*YnmReal*imagj;
        rhon/=r;
      }
      pn=-pn*fact*s2;
      fact=fact+2;
    }
    accel.x+=sinf(theta)*cosf(phi)*accelR+cosf(theta)*cosf(phi)/r*accelTheta-sinf(phi)/r/yy*accelPhi;
    accel.y+=sinf(theta)*sinf(phi)*accelR+cosf(theta)*sinf(phi)/r*accelTheta+cosf(phi)/r/yy*accelPhi;
    accel.z+=cosf(theta)*accelR-sinf(theta)/r*accelTheta;
  }
  deviceAccel[blockIdx.x*threadsPerBlock+threadIdx.x]=accel;
}
