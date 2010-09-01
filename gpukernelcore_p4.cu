#define ODDEVEN(n) ((n & 1  ==  1) ? -1 : 1)

__device__ void cart2sph(float& r, float& theta, float& phi, float dx, float dy, float dz)
{
  r = sqrtf(dx * dx + dy * dy + dz * dz)+eps;
  theta = acosf( dz / r );
  if(fabs(dx) + fabs(dy) < eps){
    phi = 0;
  }
  else if(fabs(dx) < eps){
    phi = dy/fabs(dy) * M_PI * 0.5;
  }
  else if(dx > 0){
    phi = atanf(dy/dx);
  }
  else{
    phi = atanf(dy/dx) + M_PI;
  }
}

__device__ float3 p2p_kernel_core(float3 accel,
                                  float3 posTarget, float4 sharedPosSource)
{
  float3 dist;
  dist.x = posTarget.x - sharedPosSource.x;
  dist.y = posTarget.y - sharedPosSource.y;
  dist.z = posTarget.z - sharedPosSource.z;
  float invDist = rsqrtf(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z + eps);
  float invDistCube = invDist * invDist * invDist;
  float s = sharedPosSource.w * invDistCube;
  accel.x  -=  dist.x * s;
  accel.y  -=  dist.y * s;
  accel.z  -=  dist.z * s;
  return accel;
}

__global__ void p2p_kernel(int* deviceOffset, float3* devicePosTarget,
                           float4* devicePosSource, float3* deviceAccel)
{
  int jbase, jsize, jblok, numInteraction;
  int j, ij, jj, jb;
  const int threadsPerBlock = threadsPerBlockTypeA;
  const int offsetStride = 2 * maxP2PInteraction + 1;
  float3 posTarget;
  float3 accel = {0.0f, 0.0f, 0.0f};
  __shared__ float4 sharedPosSource[threadsPerBlock];
  posTarget = devicePosTarget[blockIdx.x * threadsPerBlock + threadIdx.x];
  numInteraction = deviceOffset[blockIdx.x * offsetStride];
  for(ij = 0; ij < numInteraction; ij++){
    jbase = deviceOffset[blockIdx.x * offsetStride + 2 * ij + 1];
    jsize = deviceOffset[blockIdx.x * offsetStride + 2 * ij + 2];
    jblok = (jsize + threadsPerBlock - 1) / threadsPerBlock;
    for(j = 0; j < jblok-1; j++){
      jb = jbase + j * threadsPerBlock + threadIdx.x;
      sharedPosSource[threadIdx.x] = devicePosSource[jb];
      __syncthreads();
#pragma unroll 32
      for(jj = 0; jj < threadsPerBlock; jj++){
        accel = p2p_kernel_core(accel, posTarget, sharedPosSource[jj]);
      }
      __syncthreads();
    }
    jb = jbase + j * threadsPerBlock + threadIdx.x;
    sharedPosSource[threadIdx.x] = devicePosSource[jb];
    __syncthreads();
    for(jj = 0; jj < jsize - (j * threadsPerBlock); jj++){
      accel = p2p_kernel_core(accel, posTarget, sharedPosSource[jj]);
    }
    __syncthreads();
  }
  deviceAccel[blockIdx.x * threadsPerBlock + threadIdx.x] = accel;
}

__device__ void p2m_kernel_core(float* MnmTarget,
                                int nn, float3 boxCenter,
                                float* sharedFactorial, float4 sharedPosSource)
{
  int m, n, mm, nm;
  float3 dist;
  float rho, alpha, beta;
  float xx, s2, p, pn, p1, p2, fact, ere, eim, rhom, rhon;
  __shared__ float sharedYnm[numExpansion2];
  mm = 0;
  for(m = 0; m<=nn; m++) mm += m;
  mm = threadIdx.x - mm;

  dist.x = sharedPosSource.x - boxCenter.x;
  dist.y = sharedPosSource.y - boxCenter.y;
  dist.z = sharedPosSource.z - boxCenter.z;
  cart2sph(rho, alpha, beta, dist.x, dist.y, dist.z);
  xx = cosf(alpha);
  s2 = sqrtf((1 - xx) * (1 + xx));
  fact = 1;
  pn = 1;
  rhom = 1;
  for(m = 0; m <= numExpansions; m++){
    if(m > mm) break;
    p = pn;
    nm = m * m + 2 * m;
    sharedYnm[nm] = rhom * p * rsqrtf(sharedFactorial[2 * m]);
    p1 = p;
    p = xx * (2 * m + 1) * p;
    rhom *= rho;
    rhon = rhom;
    for(n = m + 1; n <= numExpansions; n++){
      if(n > nn) break;
      nm = n * n + n + m;
      sharedYnm[nm] = rhon * p * rsqrtf(sharedFactorial[n + m] / sharedFactorial[n - m]);
      p2 = p1;
      p1 = p;
      p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      rhon *= rho;
    }
    pn = -pn * fact * s2;
    fact = fact + 2;
  }
  nm = nn * nn + nn + mm;
  ere = cosf(-mm * beta);
  eim = sinf(-mm * beta);
  MnmTarget[0] += sharedPosSource.w * sharedYnm[nm] * ere;
  MnmTarget[1] += sharedPosSource.w * sharedYnm[nm] * eim;
}

__global__ void p2m_kernel(int* deviceOffset, float* deviceMnmTarget,
                           float4* devicePosSource)
{
  int i, j, m, n, jj, ib, jb, jbase, jsize, jblok;
  const int threadsPerBlock = threadsPerBlockTypeB;
  const int offsetStride = 5;
  float3 boxCenter;
  float boxSize = deviceConstant[0];
  float3 boxMin = {deviceConstant[1], deviceConstant[2], deviceConstant[3]};
  float fact;
  float MnmTarget[2] = {0.0f, 0.0f};
  __shared__ int sharedN[threadsPerBlock];
  __shared__ float4 sharedPosSource[threadsPerBlock];
  __shared__ float sharedFactorial[2 * numExpansions];
  for(i = 0; i < threadsPerBlock; i++){
    sharedN[i] = 0;
  }
  for(n = 0; n < numExpansions; n++){
    for(m = 0; m <= n; m++){
      i = n * (n + 1) / 2 + m;
      sharedN[i] = n;
    }
  }
  fact = 1.0;
  for(m = 0; m < 2 * numExpansions; m++) {
    sharedFactorial[m] = fact;
    fact = fact * (m + 1);
  }

  jbase = deviceOffset[blockIdx.x * offsetStride];
  jsize = deviceOffset[blockIdx.x * offsetStride + 1];
  boxCenter.x = boxMin.x + (deviceOffset[blockIdx.x * offsetStride + 2] + 0.5) * boxSize;
  boxCenter.y = boxMin.y + (deviceOffset[blockIdx.x * offsetStride + 3] + 0.5) * boxSize;
  boxCenter.z = boxMin.z + (deviceOffset[blockIdx.x * offsetStride + 4] + 0.5) * boxSize;
  jblok = (jsize + threadsPerBlock - 1) / threadsPerBlock;
  for(j = 0; j < jblok - 1; j++){
    jb = jbase + j * threadsPerBlock + threadIdx.x;
    __syncthreads();
    sharedPosSource[threadIdx.x] = devicePosSource[jb];
    __syncthreads();
    for(jj = 0; jj < threadsPerBlock; jj++){
      p2m_kernel_core(MnmTarget, sharedN[threadIdx.x], boxCenter,
                      sharedFactorial, sharedPosSource[jj]);
    }
  }
  jb = jbase + j * threadsPerBlock + threadIdx.x;
  __syncthreads();
  sharedPosSource[threadIdx.x] = devicePosSource[jb];
  __syncthreads();
  for(jj = 0; jj < jsize - (j * threadsPerBlock); jj++){
    p2m_kernel_core(MnmTarget, sharedN[threadIdx.x], boxCenter,
                    sharedFactorial, sharedPosSource[jj]);
  }
  ib = blockIdx.x * threadsPerBlock + threadIdx.x;
  for(i = 0; i < 2; i++) deviceMnmTarget[2 * ib + i] = MnmTarget[i];
}

__device__ void m2m_calculate_ynm(float* sharedYnm,
                                  float rho, float alpha, float* sharedFactorial)
{
  int m, n, npn, nmn, npm, nmm;
  float xx, s2, fact, pn, p, p1, p2, rhom, rhon;
  xx = cosf(alpha);
  s2 = sqrt((1 - xx) * (1 + xx));
  fact = 1;
  pn = 1;
  rhom = 1;
  for(m = 0;m < numExpansions; m++){
    p = pn;
    npn = m * m + 2 * m;
    nmn = m * m;
    sharedYnm[npn] = rhom * p / sharedFactorial[2 * m];
    sharedYnm[nmn] = sharedYnm[npn];
    p1 = p;
    p = xx * (2 * m + 1) * p;
    rhom *= -rho;
    rhon = rhom;
    for(n = m + 1; n < numExpansions; n++){
      npm = n * n + n + m;
      nmm = n * n + n - m;
      sharedYnm[npm] = rhon * p / sharedFactorial[n+m];
      sharedYnm[nmm] = sharedYnm[npm];
      p2 = p1;
      p1 = p;
      p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      rhon *= -rho;
    }
    pn = -pn * fact * s2;
    fact = fact + 2;
  }
}

__device__ void m2m_kernel_core(float* MnmTarget,
                                int j, float beta, 
                                float* sharedFactorial,
                                float* sharedYnm,
                                float* sharedMnmSource)
{
  int i, k, m, n, nm, jnkms;
  float ere, eim, ajk, ajnkm, cnm, CnmReal, CnmImag;
  k = 0;
  for(i = 0; i <= j; i++ ) k += i;
  k = threadIdx.x - k; 
  ajk = ODDEVEN(j) * rsqrtf(sharedFactorial[j - k] * sharedFactorial[j + k]);
  for(i = 0; i < 2; i++) MnmTarget[i] = 0;
  for(n = 0; n <= j; n++){
    for(m = -n; m <= min(k-1,n); m++){
      if(j - n >= k - m){
        nm = n * n + n + m;
        jnkms = (j - n) * (j - n + 1) / 2 + k - m;
        ere = cosf(-m * beta);
        eim = sinf(-m * beta);
        ajnkm = rsqrtf(sharedFactorial[j - n - k + m]
                     * sharedFactorial[j - n + k - m]);
        cnm = ODDEVEN(m + j);
        cnm *= ajnkm / ajk * sharedYnm[nm];
        CnmReal = cnm * ere;
        CnmImag = cnm * eim;
        MnmTarget[0] += sharedMnmSource[2 * jnkms + 0] * CnmReal;
        MnmTarget[0] -= sharedMnmSource[2 * jnkms + 1] * CnmImag;
        MnmTarget[1] += sharedMnmSource[2 * jnkms + 0] * CnmImag;
        MnmTarget[1] += sharedMnmSource[2 * jnkms + 1] * CnmReal;
      }
    }
    for(m = k; m <= n; m++){
      if(j - n >= m - k){
        nm = n * n + n + m;
        jnkms = (j - n) * (j - n + 1) / 2 - k + m;
        ere = cosf(-m * beta);
        eim = sinf(-m * beta);
        ajnkm = rsqrtf(sharedFactorial[j - n - k + m]
                     * sharedFactorial[j - n + k - m]);
        cnm = ODDEVEN(k + j + m);
        cnm *= ajnkm / ajk * sharedYnm[nm];
        CnmReal = cnm * ere;
        CnmImag = cnm * eim;
        MnmTarget[0] += sharedMnmSource[2 * jnkms + 0] * CnmReal;
        MnmTarget[0] += sharedMnmSource[2 * jnkms + 1] * CnmImag;
        MnmTarget[1] += sharedMnmSource[2 * jnkms + 0] * CnmImag;
        MnmTarget[1] -= sharedMnmSource[2 * jnkms + 1] * CnmReal;
      }
    }
  }
}

__global__ void m2m_kernel(int* deviceOffset, float* deviceMnmTarget,
                           float* deviceMnmSource)
{
  int i, j, k, ij, ib, numInteraction, jbase;
  const int threadsPerBlock = threadsPerBlockTypeB;
  const int offsetStride = 5;
  float3 dist;
  float boxSize = deviceConstant[0];
  float rho, alpha, beta, fact;
  float MnmTarget[2] = {0.0f, 0.0f};
  __shared__ int sharedJ[threadsPerBlock];
  __shared__ float sharedMnmSource[2 * threadsPerBlock];
  __shared__ float sharedYnm[numExpansion2];
  __shared__ float sharedFactorial[2 * numExpansions];
  numInteraction = deviceOffset[blockIdx.x * offsetStride];
  for(i = 0; i < threadsPerBlock; i++){
    sharedJ[i] = 0;
  }
  for(j = 0;j < numExpansions; j++){
    for(k = 0; k <= j; k++){
      i = j * (j + 1) / 2 + k;
      sharedJ[i] = j;
    }
  }
  fact = 1.0;
  for(i = 0; i < 2 * numExpansions; i++) {
    sharedFactorial[i] = fact;
    fact = fact * (i + 1);
  }

  for(ij = 0; ij < numInteraction; ij++){
    jbase = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 1];
    dist.x = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 2] * boxSize / 4;
    dist.y = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 3] * boxSize / 4;
    dist.z = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 4] * boxSize / 4;
    for(i = 0; i < 2; i++) sharedMnmSource[2 * threadIdx.x + i] =
                           deviceMnmSource[2 * (jbase + threadIdx.x) + i];
    __syncthreads();
    cart2sph(rho, alpha, beta, dist.x, dist.y, dist.z);

    m2m_calculate_ynm(sharedYnm, rho, alpha, sharedFactorial);

    m2m_kernel_core(MnmTarget, sharedJ[threadIdx.x], beta,
                    sharedFactorial, sharedYnm, sharedMnmSource);
    __syncthreads();
  }
  ib = blockIdx.x * threadsPerBlock + threadIdx.x;
  for(i = 0; i < 2; i++) deviceMnmTarget[2 * ib + i] = MnmTarget[i];
}

__device__ void m2l_calculate_ynm(float* sharedYnm,
                                  float rho, float alpha, float* sharedFactorial)
{
  int i, m, n;
  float x, s, fact, pn, p, p1, p2, rhom, rhon;
  x = cosf(alpha);
  s = sqrt(1 - x * x);
  fact = 1;
  pn = 1;
  rhom = 1.0 / rho;
  for(m = 0; m < 2 * numExpansions; m++){
    p = pn;
    i = m * (m + 1) /2 + m;
    sharedYnm[i] = rhom * p;
    p1 = p;
    p = x * (2 * m + 1) * p;
    rhom /= rho;
    rhon = rhom;
    for(n = m + 1; n < 2 * numExpansions; n++){
      i = n * (n + 1) / 2 + m;
      sharedYnm[i] = rhon * p * sharedFactorial[n - m];
      p2 = p1;
      p1 = p;
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      rhon /= rho;
    }
    pn = -pn * fact * s;
    fact = fact + 2;
  }
}

__device__ void m2l_kernel_core(float* LnmTarget,
                                float beta,
                                float* sharedFactorial,
                                float* sharedYnm,
                                float* sharedMnmSource)
{
  int i, j, k, m, n, jnkm;
  float ere, eim, anm, ajk, cnm, CnmReal, CnmImag;
  j = floor(sqrt(2*threadIdx.x+0.25)-0.5);
  k = 0; 
  for(i = 0; i <= j; i++) k += i;
  k = threadIdx.x - k;
  ajk = ODDEVEN(j) * rsqrtf(sharedFactorial[j - k] * sharedFactorial[j + k]);
  for(n = 0; n < numExpansions; n++){
    for(m = -n; m < 0; m++){
      i = n * (n + 1) / 2 - m;
      jnkm = (j + n) * (j + n + 1) / 2 - m + k;
      ere = cosf((m - k) * beta);
      eim = sinf((m - k) * beta);
      anm = rsqrtf(sharedFactorial[n - m] * sharedFactorial[n + m]);
      cnm = anm * ajk * sharedYnm[jnkm];
      CnmReal = cnm * ere;
      CnmImag = cnm * eim;
      LnmTarget[0] += sharedMnmSource[2 * i + 0] * CnmReal;
      LnmTarget[0] += sharedMnmSource[2 * i + 1] * CnmImag;
      LnmTarget[1] += sharedMnmSource[2 * i + 0] * CnmImag;
      LnmTarget[1] -= sharedMnmSource[2 * i + 1] * CnmReal;
    }
    for(m = 0; m <= n; m++){
      i = n * (n + 1) / 2 + m;
      jnkm = (j + n) * (j + n + 1) / 2 + abs(m - k);
      ere = cosf((m - k) * beta);
      eim = sinf((m - k) * beta);
      anm = rsqrtf(sharedFactorial[n - m] * sharedFactorial[n + m]);
      cnm = ODDEVEN((abs(k - m) - k - m) / 2);
      cnm *= anm * ajk * sharedYnm[jnkm];
      CnmReal = cnm * ere;
      CnmImag = cnm * eim;
      LnmTarget[0] += sharedMnmSource[2 * i + 0] * CnmReal;
      LnmTarget[0] -= sharedMnmSource[2 * i + 1] * CnmImag;
      LnmTarget[1] += sharedMnmSource[2 * i + 0] * CnmImag;
      LnmTarget[1] += sharedMnmSource[2 * i + 1] * CnmReal;
    }
  }
}

__global__ void m2l_kernel(int* deviceOffset, float* deviceLnmTarget,
                           float* deviceMnmSource)
{
  int i, ij, ib, numInteraction, jbase;
  const int threadsPerBlock = threadsPerBlockTypeB;
  const int offsetStride = 4*maxM2LInteraction+1;
  float3 dist;
  float boxSize = deviceConstant[0];
  float rho, alpha, beta, fact;
  float LnmTarget[2] = {0.0f, 0.0f};
  __shared__ float sharedMnmSource[2 * threadsPerBlock];
  __shared__ float sharedYnm[numCoefficients];
  __shared__ float sharedFactorial[2 * numExpansions];
  numInteraction = deviceOffset[blockIdx.x * offsetStride];
  fact = 1.0;
  for(i = 0; i < 2 * numExpansions; i++) {
    sharedFactorial[i] = fact;
    fact = fact * (i + 1);
  }
  for(ij = 0; ij < numInteraction; ij++){
    jbase  = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 1];
    dist.x = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 2] * boxSize;
    dist.y = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 3] * boxSize;
    dist.z = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 4] * boxSize;
    for(i=0;i<2;i++) sharedMnmSource[2 * threadIdx.x + i] = 
                     deviceMnmSource[2 * (jbase + threadIdx.x) + i];
    __syncthreads();
    cart2sph(rho, alpha, beta, dist.x, dist.y, dist.z);
    m2l_calculate_ynm(sharedYnm, rho, alpha, sharedFactorial);
    m2l_kernel_core(LnmTarget, beta, sharedFactorial, sharedYnm, sharedMnmSource);
    __syncthreads();
  }
  ib = blockIdx.x * threadsPerBlock + threadIdx.x;
  for(i=0;i<2;i++) deviceLnmTarget[2 * ib + i] = LnmTarget[i];
}

__device__ void l2l_calculate_ynm(float* sharedYnm,
                                  float rho, float alpha, float* sharedFactorial)
{
  int m, n, npn, nmn, npm, nmm;
  float xx, s2, fact, pn, p, p1, p2, rhom, rhon, anm;
  xx = cosf(alpha);
  s2 = sqrt((1 - xx) * (1 + xx));
  fact = 1;
  pn = 1;
  rhom = 1;
  for(m = 0; m < numExpansions; m++){
    p = pn;
    npn = m * m + 2 * m;
    nmn = m * m;
    anm = 1 / sharedFactorial[2 * m];
    sharedYnm[npn] = rhom * p * anm;
    sharedYnm[nmn] = sharedYnm[npn];
    p1 = p;
    p = xx * (2 * m + 1) * p;
    rhom *= -rho;
    rhon = rhom;
    for(n = m + 1; n < numExpansions; n++){
      npm = n * n + n + m;
      nmm = n * n + n - m;
      anm = 1.0 / sharedFactorial[n + m];
      sharedYnm[npm] = rhon * p * anm;
      sharedYnm[nmm] = sharedYnm[npm];
      p2 = p1;
      p1 = p;
      p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      rhon *= -rho;
    }
    pn = -pn * fact * s2;
    fact = fact + 2;
  }
}

__device__ void l2l_kernel_core(float* LnmTarget,
                                int j, float beta,
                                float* sharedFactorial,
                                float* sharedYnm,
                                float* sharedLnmSource)
{
  int i, k, m, n, jnkm;
  float ere, eim, ajk, anm, cnm, CnmReal, CnmImag;
  k = 0;
  for(i = 0; i <= j; i++ ) k += i;
  k = threadIdx.x - k;
  ajk = ODDEVEN(j) * rsqrtf(sharedFactorial[j - k] * sharedFactorial[j + k]);
  for(n = j; n < numExpansions; n++){
    for(m = j + k - n; m < 0; m++){
      i = n * (n + 1) / 2 - m;
      jnkm = (n - j) * (n - j) + n - j + m - k;
      ere = cosf((m - k) * beta);
      eim = sinf((m - k) * beta);
      anm = rsqrtf(sharedFactorial[n - m] * sharedFactorial[n + m]);
      cnm = ODDEVEN(k - n) * ajk / anm * sharedYnm[jnkm];
      CnmReal = cnm * ere;
      CnmImag = cnm * eim;
      LnmTarget[0] += sharedLnmSource[2 * i + 0] * CnmReal;
      LnmTarget[0] += sharedLnmSource[2 * i + 1] * CnmImag;
      LnmTarget[1] += sharedLnmSource[2 * i + 0] * CnmImag;
      LnmTarget[1] -= sharedLnmSource[2 * i + 1] * CnmReal;
    }
    for(m = 0; m <= n; m++){
      if(n - j >= abs(m - k)){
        i = n * (n + 1) / 2 + m;
        jnkm = (n - j) * (n - j) + n - j + m - k;
        ere = cosf((m - k) * beta);
        eim = sinf((m - k) * beta);
        anm = rsqrtf(sharedFactorial[n - m] * sharedFactorial[n + m]);
        cnm = ODDEVEN((m - k - abs(m - k)) / 2 - n);
        cnm *= ajk / anm*sharedYnm[jnkm];
        CnmReal = cnm * ere;
        CnmImag = cnm * eim;
        LnmTarget[0] += sharedLnmSource[2 * i + 0] * CnmReal;
        LnmTarget[0] -= sharedLnmSource[2 * i + 1] * CnmImag;
        LnmTarget[1] += sharedLnmSource[2 * i + 0] * CnmImag;
        LnmTarget[1] += sharedLnmSource[2 * i + 1] * CnmReal;
      }
    }
  }
}

__global__ void l2l_kernel(int* deviceOffset, float* deviceLnmTarget,
                           float* deviceLnmSource)
{
  int i, j, k, ij, ib, numInteraction, jbase;
  const int threadsPerBlock = threadsPerBlockTypeB;
  const int offsetStride = 5;
  float3 dist;
  float boxSize = deviceConstant[0];
  float rho,alpha,beta,fact;
  float LnmTarget[2] = {0.0f, 0.0f};
  __shared__ int sharedJ[threadsPerBlock];
  __shared__ float sharedLnmSource[2 * threadsPerBlock];
  __shared__ float sharedYnm[numExpansion2];
  __shared__ float sharedFactorial[2 * numExpansions];
  numInteraction = deviceOffset[blockIdx.x * offsetStride];
  for(i = 0; i < threadsPerBlock; i++){
    sharedJ[i] = 0;
  }
  for(j = 0; j < numExpansions; j++){
    for(k = 0; k <= j; k++){
      i = j * (j + 1) / 2 + k;
      sharedJ[i] = j;
    }
  }
  fact = 1.0;
  for(i = 0; i < 2 * numExpansions; i++) {
    sharedFactorial[i] = fact;
    fact = fact * (i + 1);
  }

  for(ij = 0; ij < numInteraction; ij++){
    jbase  = deviceOffset[blockIdx.x * offsetStride + 4 *ij + 1];
    dist.x = deviceOffset[blockIdx.x * offsetStride + 4 *ij + 2] * boxSize / 2;
    dist.y = deviceOffset[blockIdx.x * offsetStride + 4 *ij + 3] * boxSize / 2;
    dist.z = deviceOffset[blockIdx.x * offsetStride + 4 *ij + 4] * boxSize / 2;
    for(i = 0; i < 2; i++) sharedLnmSource[2 * threadIdx.x + i] =
                           deviceLnmSource[2 * (jbase + threadIdx.x) + i];
    __syncthreads();
    cart2sph(rho, alpha, beta, dist.x, dist.y, dist.z);

    l2l_calculate_ynm(sharedYnm, rho, alpha, sharedFactorial);

    l2l_kernel_core(LnmTarget, sharedJ[threadIdx.x], beta,
                    sharedFactorial, sharedYnm, sharedLnmSource);
    __syncthreads();
  }
  ib = blockIdx.x * threadsPerBlock + threadIdx.x;
  for(i = 0; i < 2; i++) deviceLnmTarget[2 * ib + i] = LnmTarget[i];
}

__device__ float3 l2p_kernel_core(float3 accel,
                                  float r, float theta, float phi,
                                  float* sharedFactorial, float* sharedLnmSource)
{
  int i, m, n;
  float xx, yy, s2, p, pn, p1, p2, fact, ere, eim, rhom, rhon, realj, imagj;
  float accelR, accelTheta, accelPhi;
  float anm, Ynm, YnmTheta;
  accelR = 0;
  accelTheta = 0;
  accelPhi = 0;
  xx = cosf(theta);
  yy = sinf(theta);
  if(abs(yy) < eps) yy = 1 / eps;
  s2 = sqrtf((1 - xx) * (1 + xx));
  fact = 1;
  pn = 1;
  rhom = 1;
  for(m = 0; m < numExpansions; m++){
    p = pn;
    i = m * (m + 1) / 2 + m;
    ere = cosf(m * phi);
    if(m == 0) ere = 0.5;
    eim = sinf(m * phi);
    anm = rhom * rsqrtf(sharedFactorial[2 * m]);
    Ynm = anm * p;
    p1 = p;
    p = xx * (2 * m + 1) * p;
    YnmTheta = anm * (p - (m + 1) * xx * p1) / yy;
    realj = ere*sharedLnmSource[2 * i + 0] - eim * sharedLnmSource[2 * i + 1];
    imagj = eim*sharedLnmSource[2 * i + 0] + ere * sharedLnmSource[2 * i + 1];
    accelR += 2 * m / r * Ynm * realj;
    accelTheta += 2 * YnmTheta * realj;
    accelPhi -= 2 * m * Ynm * imagj;
    rhom *= r;
    rhon = rhom;
    for(n = m + 1; n < numExpansions; n++){
      i = n * (n + 1) / 2 + m;
      anm = rhon * rsqrtf(sharedFactorial[n + m] / sharedFactorial[n - m]);
      Ynm = anm * p;
      p2 = p1;
      p1 = p;
      p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      YnmTheta = anm * ((n - m + 1) * p - (n + 1) * xx * p1) / yy;
      realj = ere*sharedLnmSource[2 * i + 0] - eim * sharedLnmSource[2 * i + 1];
      imagj = eim*sharedLnmSource[2 * i + 0] + ere * sharedLnmSource[2 * i + 1];
      accelR += 2 * n / r * Ynm * realj;
      accelTheta += 2 * YnmTheta * realj;
      accelPhi -= 2 * m * Ynm * imagj;
      rhon *= r;
    }
    pn = -pn * fact * s2;
    fact = fact + 2;
  }
  accel.x += sinf(theta) * cosf(phi) * accelR
           + cosf(theta) * cosf(phi) / r * accelTheta
           - sinf(phi) / r / yy * accelPhi;
  accel.y += sinf(theta) * sinf(phi) * accelR
           + cosf(theta) * sinf(phi) / r * accelTheta
           + cosf(phi) / r / yy * accelPhi;
  accel.z += cosf(theta) * accelR
           - sinf(theta) / r * accelTheta;
  return accel;
}

__global__ void l2p_kernel(int* deviceOffset, float3* devicePosTarget,
                           float* deviceLnmSource, float3* deviceAccel)
{
  int i, jbase;
  const int threadsPerBlock = threadsPerBlockTypeB;
  const int offsetStride = 5;
  float boxSize = deviceConstant[0];
  float r, theta, phi, fact;
  float3 boxMin = {deviceConstant[1], deviceConstant[2], deviceConstant[3]};
  float3 boxCenter, dist;
  float3 accel = {0.0f, 0.0f, 0.0f};
  __shared__ float sharedLnmSource[2 * threadsPerBlock];
  __shared__ float sharedFactorial[2 * numExpansions];
  fact = 1.0;
  for(i = 0; i < 2 * numExpansions; i++) {
    sharedFactorial[i] = fact;
    fact = fact * (i + 1);
  }
  jbase = deviceOffset[blockIdx.x * offsetStride];
  boxCenter.x = boxMin.x + (deviceOffset[blockIdx.x * offsetStride + 2] + 0.5) * boxSize;
  boxCenter.y = boxMin.y + (deviceOffset[blockIdx.x * offsetStride + 3] + 0.5) * boxSize;
  boxCenter.z = boxMin.z + (deviceOffset[blockIdx.x * offsetStride + 4] + 0.5) * boxSize;
  for(i = 0; i < 2; i++) sharedLnmSource[2 * threadIdx.x + i] =
                         deviceLnmSource[2 * (jbase + threadIdx.x) + i];
  __syncthreads();
  dist.x = devicePosTarget[blockIdx.x * threadsPerBlock + threadIdx.x].x - boxCenter.x;
  dist.y = devicePosTarget[blockIdx.x * threadsPerBlock + threadIdx.x].y - boxCenter.y;
  dist.z = devicePosTarget[blockIdx.x * threadsPerBlock + threadIdx.x].z - boxCenter.z;
  cart2sph(r, theta, phi, dist.x, dist.y, dist.z);
  accel = l2p_kernel_core(accel, r, theta, phi, sharedFactorial, sharedLnmSource);
  deviceAccel[blockIdx.x * threadsPerBlock + threadIdx.x] = accel;
}

__device__ float3 m2p_kernel_core(float3 accel,
                                  float r, float theta, float phi,
                                  float* sharedFactorial, float* sharedMnmSource)
{
  int i, m ,n;
  float xx, yy, s2, p, pn, p1, p2, fact, ere, eim, rhom, rhon, realj, imagj;
  float accelR, accelTheta, accelPhi;
  float anm, Ynm, YnmTheta;
  accelR = 0;
  accelTheta = 0;
  accelPhi = 0;
  xx = cosf(theta);
  yy = sinf(theta);
  if(fabs(yy) < eps) yy = 1 / eps;
  s2 = sqrtf((1 - xx) * (1 + xx));
  fact = 1;
  pn = 1;
  rhom = 1.0 / r;
  for(m = 0; m < numExpansions; m++){
    p = pn;
    i = m * (m + 1) / 2 + m;
    ere = cosf(m * phi);
    if(m == 0) ere = 0.5;
    eim = sinf(m * phi);
    anm = rhom * rsqrt(sharedFactorial[2 * m]);
    Ynm = anm * p;
    p1 = p;
    p = xx * (2 * m + 1) * p;
    YnmTheta = anm * (p - (m + 1) * xx * p1) / yy;
    realj = ere*sharedMnmSource[2 * i + 0] - eim * sharedMnmSource[2 * i + 1];
    imagj = eim*sharedMnmSource[2 * i + 0] + ere * sharedMnmSource[2 * i + 1];
    accelR -= 2 * (m + 1) / r * Ynm * realj;
    accelTheta += 2 * YnmTheta * realj;
    accelPhi -= 2 * m * Ynm * imagj;
    rhom /= r;
    rhon = rhom;
    for(n = m + 1; n < numExpansions; n++){
      i = n * (n + 1) / 2 + m;
      anm = rhon * rsqrt(sharedFactorial[n + m] / sharedFactorial[n - m]);
      Ynm = anm * p;
      p2 = p1;
      p1 = p;
      p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      YnmTheta = anm * ((n - m + 1) * p - (n + 1) * xx * p1) / yy;
      realj = ere * sharedMnmSource[2 * i + 0] - eim * sharedMnmSource[2 * i + 1];
      imagj = eim * sharedMnmSource[2 * i + 0] + ere * sharedMnmSource[2 * i + 1];
      accelR -= 2 * (n + 1) / r * Ynm * realj;
      accelTheta += 2 * YnmTheta * realj;
      accelPhi -= 2 * m * Ynm * imagj;
      rhon /= r;
    }
    pn = -pn * fact * s2;
    fact = fact+2;
  }
  accel.x += sinf(theta) * cosf(phi) * accelR
           + cosf(theta) * cosf(phi) / r * accelTheta
           - sinf(phi) / r / yy * accelPhi;
  accel.y += sinf(theta) * sinf(phi) * accelR
           + cosf(theta) * sinf(phi) / r * accelTheta
           + cosf(phi) / r / yy * accelPhi;
  accel.z += cosf(theta) * accelR
           - sinf(theta) / r * accelTheta;
  return accel;
}

__global__ void m2p_kernel(int* deviceOffset, float3* devicePosTarget,
                           float* deviceMnmSource, float3* deviceAccel)
{
  int i, jx, jy, jz, ij, numInteraction, jbase;
  const int threadsPerBlock = threadsPerBlockTypeB;
  const int offsetStride = 4 * maxM2LInteraction + 1;
  float boxSize = deviceConstant[0];
  float3 boxMin = {deviceConstant[1], deviceConstant[2], deviceConstant[3]};
  float3 boxCenter, dist;
  float r, theta, phi, fact;
  float3 accel = {0.0f, 0.0f, 0.0f};
  float3 posTarget;
  __shared__ float sharedMnmSource[2 * threadsPerBlock];
  __shared__ float sharedFactorial[2 * numExpansions];
  numInteraction = deviceOffset[blockIdx.x * offsetStride];
  fact = 1.0;
  for(i = 0; i < 2 * numExpansions; i++) {
    sharedFactorial[i] = fact;
    fact = fact * (i + 1);
  }
  __syncthreads();
  posTarget = devicePosTarget[blockIdx.x * threadsPerBlock + threadIdx.x];
  __syncthreads();
  for(ij = 0; ij < numInteraction; ij++){
    jbase = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 1];
    jx    = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 2];
    jy    = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 3];
    jz    = deviceOffset[blockIdx.x * offsetStride + 4 * ij + 4];
    boxCenter.x = boxMin.x + (jx + 0.5) * boxSize;
    boxCenter.y = boxMin.y + (jy + 0.5) * boxSize;
    boxCenter.z = boxMin.z + (jz + 0.5) * boxSize;
    for(i = 0; i < 2; i++) sharedMnmSource[2 * threadIdx.x + i] =
                           deviceMnmSource[2 * (jbase + threadIdx.x) + i];
    __syncthreads();
    dist.x = posTarget.x - boxCenter.x;
    dist.y = posTarget.y - boxCenter.y;
    dist.z = posTarget.z - boxCenter.z;
    cart2sph(r, theta, phi, dist.x, dist.y, dist.z);
    accel = m2p_kernel_core(accel, r, theta, phi, sharedFactorial, sharedMnmSource);
  }
  deviceAccel[blockIdx.x * threadsPerBlock + threadIdx.x] = accel;
}
