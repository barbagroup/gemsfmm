#include "fmm.h"

// Dynamically allocate memory for non-empty boxes
void FmmSystem::allocate() {
  int i,j;

  particleOffset = new int* [2];
  for( i=0; i<2; i++ ) particleOffset[i] = new int [numBoxIndexLeaf];
  boxIndexMask = new int [numBoxIndexFull];
  boxIndexFull = new int [numBoxIndexTotal];
  levelOffset = new int [maxLevel];
  numInteraction = new int [numBoxIndexLeaf];
  interactionList = new int [numBoxIndexLeaf][maxM2LInteraction];
  boxOffsetStart = new int [numBoxIndexLeaf];
  boxOffsetEnd = new int [numBoxIndexLeaf];

  factorial = new float [4*numExpansion2];
  Lnm = new std::complex<double> [numBoxIndexLeaf][numCoefficients];
  LnmOld = new std::complex<double> [numBoxIndexLeaf][numCoefficients];
  Mnm = new std::complex<double> [numBoxIndexTotal][numCoefficients];
  Ynm = new std::complex<double> [4*numExpansion2];
  Dnm = new std::complex<double>** [2*numRelativeBox];
  for( i=0; i<2*numRelativeBox; i++ ) {
    Dnm[i] = new std::complex<double>* [numExpansions];
    for( j=0; j<numExpansions; j++ ) Dnm[i][j] = new std::complex<double> [numExpansion2];
  }
}

// Free memory corresponding to allocate()
void FmmSystem::deallocate() {
  int i,j;

  for( i=0; i<2; i++ ) delete[] particleOffset[i];
  delete[] particleOffset;
  delete[] boxIndexMask;
  delete[] boxIndexFull;
  delete[] levelOffset;
  delete[] numInteraction;
  delete[] interactionList;
  delete[] boxOffsetStart;
  delete[] boxOffsetEnd;
  delete[] mortonIndex;
  delete[] sortValue;
  delete[] sortIndex;
  delete[] sortValueBuffer;
  delete[] sortIndexBuffer;

  delete[] factorial;
  delete[] Lnm;
  delete[] LnmOld;
  delete[] Mnm;
  delete[] Ynm;
  for( i=0; i<2*numRelativeBox; i++ ) {
    for( j=0; j<numExpansions; j++ ) delete[] Dnm[i][j];
    delete[] Dnm[i];
  }
  delete[] Dnm;
}

// Calculate range of FMM domain from particle positions
void FmmSystem::setDomainSize(int numParticles) {
  int i;
  float xmin,xmax,ymin,ymax,zmin,zmax;

// Initialize the minimum and maximum values
  xmin = 1000000;
  xmax = -1000000;
  ymin = 1000000;
  ymax = -1000000;
  zmin = 1000000;
  zmax = -1000000;
// Calculate the minimum and maximum of particle positions
  for( i=0; i<numParticles; i++ ) {
    xmin = std::min(xmin,bodyPos[i].x);
    xmax = std::max(xmax,bodyPos[i].x);
    ymin = std::min(ymin,bodyPos[i].y);
    ymax = std::max(ymax,bodyPos[i].y);
    zmin = std::min(zmin,bodyPos[i].z);
    zmax = std::max(zmax,bodyPos[i].z);
  }
  boxMin.x = xmin;
  boxMin.y = ymin;
  boxMin.z = zmin;
// Calculat the domain size
  rootBoxSize = 0;
  rootBoxSize = std::max(rootBoxSize,xmax-xmin);
  rootBoxSize = std::max(rootBoxSize,ymax-ymin);
  rootBoxSize = std::max(rootBoxSize,zmax-zmin);
  rootBoxSize *= 1.00001; // Keep particle on the edge from falling out
}

// Calculate leaf level optimally from tabulated theshold values of numParticles
void FmmSystem::setOptimumLevel(int numParticles) {
//  float level_switch[6]={2e4,1.7e5,1.3e6,1e7,7e7,5e8}; // cpu-tree
//  float level_switch[6]={1.3e4,1e5,7e5,5e6,3e7,1.5e8}; // cpu-fmm
//  float level_switch[6]={1e5,5e5,5e6,3e7,2e8,1.5e9}; // gpu-tree
  float level_switch[6]={1e5,7e5,7e6,5e7,3e8,2e9}; // gpu-fmm

  maxLevel = 1;
  if( numParticles < level_switch[0] ) {
    maxLevel += 1;
  } else if( numParticles < level_switch[1] ) {
    maxLevel += 2;
  } else if( numParticles < level_switch[2] ) {
    maxLevel += 3;
  } else if( numParticles < level_switch[3] ) {
    maxLevel += 4;
  } else if( numParticles < level_switch[4] ) {
    maxLevel += 5;
  } else if( numParticles < level_switch[5] ) {
    maxLevel += 6;
  } else {
    maxLevel += 7;
  }
  printf("level  : %d\n",maxLevel);
  numBoxIndexFull = 1 << 3*maxLevel;
}

// Generate Morton index from particle coordinates
void FmmSystem::morton(int numParticles) {
  int i,j,nx,ny,nz,boxIndex;
  float boxSize;
  boxSize = rootBoxSize/(1 << maxLevel);

  for( j=0; j<numParticles; j++ ) {
    nx = int((bodyPos[j].x-boxMin.x)/boxSize);
    ny = int((bodyPos[j].y-boxMin.y)/boxSize);
    nz = int((bodyPos[j].z-boxMin.z)/boxSize);
    if( nx >= (1 << maxLevel) ) nx--;
    if( ny >= (1 << maxLevel) ) ny--;
    if( nz >= (1 << maxLevel) ) nz--;
    boxIndex = 0;
    for( i=0; i<maxLevel; i++ ) {
      boxIndex += nx%2 << (3*i+1);
      nx >>= 1;

      boxIndex += ny%2 << (3*i);
      ny >>= 1;

      boxIndex += nz%2 << (3*i+2);
      nz >>= 1;
    }
    mortonIndex[j] = boxIndex;
  }
}

// Generate Morton index for a box center to use in M2L translation
void FmmSystem::morton1(vec3<int> boxIndex3D, int& boxIndex, int numLevel) {
  int i,nx,ny,nz;
  boxIndex = 0;
  for( i=0; i<numLevel; i++ ) {
    nx = boxIndex3D.x%2;
    boxIndex3D.x >>= 1;
    boxIndex += nx*(1 << (3*i+1));

    ny = boxIndex3D.y%2;
    boxIndex3D.y >>= 1;
    boxIndex += ny*(1 << (3*i));

    nz = boxIndex3D.z%2;
    boxIndex3D.z >>= 1;
    boxIndex += nz*(1 << (3*i+2));
  }
}

// Returns 3D box index from Morton index
void FmmSystem::unmorton(int boxIndex, vec3<int>& boxIndex3D) {
  int i,j,k,n,mortonIndex3D[3];

  for( i=0; i<3; i++ ) mortonIndex3D[i] = 0;
  n = boxIndex;
  k = 0;
  i = 0;
  while( n != 0 ) {
    j = 2-k;
    mortonIndex3D[j] += (n%2)*(1 << i);
    n >>= 1;
    k = (k+1)%3;
    if( k == 0 ) i++;
  }
  boxIndex3D.x = mortonIndex3D[1];
  boxIndex3D.y = mortonIndex3D[2];
  boxIndex3D.z = mortonIndex3D[0];
}

// Prepare for binning particles by first sorting the Morton index
void FmmSystem::sort(int numParticles) {
  int i;

  for( i=0; i<numBoxIndexFull; i++ ) sortIndexBuffer[i] = 0;
  for( i=0; i<numParticles; i++ ) sortIndexBuffer[sortValue[i]]++;
  for( i=1; i<numBoxIndexFull; i++ ) sortIndexBuffer[i] += sortIndexBuffer[i-1];
  for( i=numParticles-1; i>=0; i-- ) {
    sortIndexBuffer[sortValue[i]]--;
    sortValueBuffer[sortIndexBuffer[sortValue[i]]] = sortValue[i];
    sortIndex[sortIndexBuffer[sortValue[i]]] = i;
  }
  for( i=0; i<numParticles; i++ ) sortValue[i] = sortValueBuffer[i];
}

// Sort the particles according to the previously sorted Morton index
void FmmSystem::sortParticles(int& numParticles) {
  int i;

  permutation = new int [numParticles];

  morton(numParticles);
  for( i=0; i<numParticles; i++ ) {
    sortValue[i] = mortonIndex[i];
    sortIndex[i] = i;
  }
  sort(numParticles);
  for( i=0; i<numParticles; i++ ) {
    permutation[i] = sortIndex[i];
  }

  vec4<float> *sortBuffer;
  sortBuffer = new vec4<float> [numParticles];
  for( i=0; i<numParticles; i++ ) {
    sortBuffer[i] = bodyPos[permutation[i]];
  }
  for( i=0; i<numParticles; i++ ) {
    bodyPos[i] = sortBuffer[i];
  }
  delete[] sortBuffer;
}

// Unsorting particles upon exit (optional)
void FmmSystem::unsortParticles(int& numParticles) {
  int i;
  vec3<float> *sortBuffer;
  sortBuffer = new vec3<float> [numParticles];
  for( i=0; i<numParticles; i++ ) {
    sortBuffer[permutation[i]] = bodyAccel[i];
  }
  for( i=0; i<numParticles; i++ ) {
    bodyAccel[i] = sortBuffer[i];
  }
  delete[] sortBuffer;
  vec4<float> *sortBuffer2;
  sortBuffer2 = new vec4<float> [numParticles];
  for( i=0; i<numParticles; i++ ) {
    sortBuffer2[permutation[i]] = bodyPos[i];
  }
  for( i=0; i<numParticles; i++ ) {
    bodyPos[i] = sortBuffer2[i];
  }
  delete[] sortBuffer2;
  delete[] permutation;
}

// Estimate storage requirements adaptively to skip empty boxes
void FmmSystem::countNonEmptyBoxes(int numParticles) {
  int i,currentIndex,numLevel;

  morton(numParticles);
  for( i=0; i<numParticles; i++ ) {
    sortValue[i] = mortonIndex[i];
    sortIndex[i] = i;
  }
  sort(numParticles);

// Count non-empty boxes at leaf level
  numBoxIndexLeaf = 0; // counter
  currentIndex = -1;
  for( i=0; i<numParticles; i++ ) {
    if( sortValue[i] != currentIndex ) {
      numBoxIndexLeaf++;
      currentIndex = sortValue[i];
    }
  }

// Count non-empty boxes for all levels
  numBoxIndexTotal = numBoxIndexLeaf;
  for( numLevel=maxLevel-1; numLevel>=2; numLevel-- ) {
    currentIndex = -1;
    for( i=0; i<numParticles; i++ ) {
      if( sortValue[i]/(1 << 3*(maxLevel-numLevel)) != currentIndex ) {
        numBoxIndexTotal++;
        currentIndex = sortValue[i]/(1 << 3*(maxLevel-numLevel));
      }
    }
  }
}

// Obtain two-way link list between non-empty and full box indices, and offset of particle index
void FmmSystem::getBoxData(int numParticles, int& numBoxIndex) {
  int i,currentIndex;

  morton(numParticles);

  numBoxIndex = 0;
  currentIndex = -1;
  for( i=0; i<numBoxIndexFull; i++ ) boxIndexMask[i] = -1;
  for( i=0; i<numParticles; i++ ) {
    if( mortonIndex[i] != currentIndex ) {
      boxIndexMask[mortonIndex[i]] = numBoxIndex;
      boxIndexFull[numBoxIndex] = mortonIndex[i];
      particleOffset[0][numBoxIndex] = i;
      if( numBoxIndex > 0 ) particleOffset[1][numBoxIndex-1] = i-1;
      currentIndex = mortonIndex[i];
      numBoxIndex++;
    }
  }
  particleOffset[1][numBoxIndex-1] = numParticles-1;
}

// Propagate non-empty/full link list to parent boxes
void FmmSystem::getBoxDataOfParent(int& numBoxIndex, int numLevel, int treeOrFMM) {
  int i,numBoxIndexOld,currentIndex,boxIndex;
  levelOffset[numLevel-1] = levelOffset[numLevel]+numBoxIndex;
  numBoxIndexOld = numBoxIndex;
  numBoxIndex = 0;
  currentIndex = -1;
  for( i=0; i<numBoxIndexFull; i++ ) boxIndexMask[i] = -1;
  for( i=0; i<numBoxIndexOld; i++ ) {
    boxIndex = i+levelOffset[numLevel];
    if( currentIndex != boxIndexFull[boxIndex]/8 ) {
      currentIndex = boxIndexFull[boxIndex]/8;
      boxIndexMask[currentIndex] = numBoxIndex;
      boxIndexFull[numBoxIndex+levelOffset[numLevel-1]] = currentIndex;
      if( treeOrFMM == 0 ) {
        particleOffset[0][numBoxIndex] = particleOffset[0][i];
        if( numBoxIndex > 0 ) particleOffset[1][numBoxIndex-1] = particleOffset[0][i]-1;
      }
      numBoxIndex++;
    }
  }
  if( treeOrFMM == 0 ) particleOffset[1][numBoxIndex-1] = particleOffset[1][numBoxIndexOld-1];
}

// Recalculate non-empty box index for current level
void FmmSystem::getBoxIndexMask(int numBoxIndex, int numLevel) {
  int i,boxIndex;
  for( i=0; i<numBoxIndexFull; i++ ) boxIndexMask[i] = -1;
  for( i=0; i<numBoxIndex; i++ ) {
    boxIndex = i+levelOffset[numLevel-1];
    boxIndexMask[boxIndexFull[boxIndex]] = i;
  }
}

// Calculate the interaction list for P2P and M2L
void FmmSystem::getInteractionList(int numBoxIndex, int numLevel, int interactionType) {
  int jxmin,jxmax,jymin,jymax,jzmin,jzmax,ii,ib,jj,jb,ix,iy,iz,jx,jy,jz,boxIndex;
  int ixp,iyp,izp,jxp,jyp,jzp;
  vec3<int> boxIndex3D;

// Initialize the minimum and maximum values
  jxmin = 1000000;
  jxmax = -1000000;
  jymin = 1000000;
  jymax = -1000000;
  jzmin = 1000000;
  jzmax = -1000000;
// Calculate the minimum and maximum of boxIndex3D
  for( jj=0; jj<numBoxIndex; jj++ ) {
    jb = jj+levelOffset[numLevel-1];
    unmorton(boxIndexFull[jb],boxIndex3D);
    jxmin = std::min(jxmin,boxIndex3D.x);
    jxmax = std::max(jxmax,boxIndex3D.x);
    jymin = std::min(jymin,boxIndex3D.y);
    jymax = std::max(jymax,boxIndex3D.y);
    jzmin = std::min(jzmin,boxIndex3D.z);
    jzmax = std::max(jzmax,boxIndex3D.z);
  }
// P2P
  if( interactionType == 0 ) {
    for( ii=0; ii<numBoxIndex; ii++ ) {
      ib = ii+levelOffset[numLevel-1];
      numInteraction[ii] = 0;
      unmorton(boxIndexFull[ib],boxIndex3D);
      ix = boxIndex3D.x;
      iy = boxIndex3D.y;
      iz = boxIndex3D.z;
      for( jx=std::max(ix-1,jxmin); jx<=std::min(ix+1,jxmax); jx++ ) {
        for( jy=std::max(iy-1,jymin); jy<=std::min(iy+1,jymax); jy++ ) {
          for( jz=std::max(iz-1,jzmin); jz<=std::min(iz+1,jzmax); jz++ ) {
            boxIndex3D.x = jx;
            boxIndex3D.y = jy;
            boxIndex3D.z = jz;
            morton1(boxIndex3D,boxIndex,numLevel);
            jj = boxIndexMask[boxIndex];
            if( jj != -1 ) {
              interactionList[ii][numInteraction[ii]] = jj;
              numInteraction[ii]++;
            }
          }
        }
      }
    }
// M2L at level 2
  } else if( interactionType == 1 ) {
    for( ii=0; ii<numBoxIndex; ii++ ) {
      ib = ii+levelOffset[numLevel-1];
      numInteraction[ii] = 0;
      unmorton(boxIndexFull[ib],boxIndex3D);
      ix = boxIndex3D.x;
      iy = boxIndex3D.y;
      iz = boxIndex3D.z;
      for( jj=0; jj<numBoxIndex; jj++ ) {
        jb = jj+levelOffset[numLevel-1];
        unmorton(boxIndexFull[jb],boxIndex3D);
        jx = boxIndex3D.x;
        jy = boxIndex3D.y;
        jz = boxIndex3D.z;
        if( jx < ix-1 || ix+1 < jx || jy < iy-1 || iy+1 < jy || jz < iz-1 || iz+1 < jz ) {
          interactionList[ii][numInteraction[ii]] = jj;
          numInteraction[ii]++;
        }
      }
    }
// M2L at lower levels
  } else if( interactionType == 2 ) {
    for( ii=0; ii<numBoxIndex; ii++ ) {
      ib = ii+levelOffset[numLevel-1];
      numInteraction[ii] = 0;
      unmorton(boxIndexFull[ib],boxIndex3D);
      ix = boxIndex3D.x;
      iy = boxIndex3D.y;
      iz = boxIndex3D.z;
      ixp = (ix+2)/2;
      iyp = (iy+2)/2;
      izp = (iz+2)/2;
      for( jxp=ixp-1; jxp<=ixp+1; jxp++ ) {
        for( jyp=iyp-1; jyp<=iyp+1; jyp++ ) {
          for( jzp=izp-1; jzp<=izp+1; jzp++ ) {
            for( jx=std::max(2*jxp-2,jxmin); jx<=std::min(2*jxp-1,jxmax); jx++ ) {
              for( jy=std::max(2*jyp-2,jymin); jy<=std::min(2*jyp-1,jymax); jy++ ) {
                for( jz=std::max(2*jzp-2,jzmin); jz<=std::min(2*jzp-1,jzmax); jz++ ) {
                  if( jx < ix-1 || ix+1 < jx || jy < iy-1 || iy+1 < jy || jz < iz-1 || iz+1 < jz ) {
                    boxIndex3D.x = jx;
                    boxIndex3D.y = jy;
                    boxIndex3D.z = jz;
                    morton1(boxIndex3D,boxIndex,numLevel);
                    jj = boxIndexMask[boxIndex];
                    if( jj != -1 ) {
                      interactionList[ii][numInteraction[ii]] = jj;
                      numInteraction[ii]++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Main part of the FMM/treecode
void FmmSystem::fmmMain(int numParticles, int treeOrFMM){
  int i,numLevel,numBoxIndex,numBoxIndexOld;
  FmmKernel kernel;
  log_time(0);
  for( i=0; i<9; i++ ) t[i] = 0;

  mortonIndex = new int [numParticles];
  sortValue  = new int [numParticles];
  sortIndex  = new int [numParticles];
  sortValueBuffer  = new int [numParticles];
  sortIndexBuffer  = new int [numParticles];

  setDomainSize(numParticles);

  setOptimumLevel(numParticles);

  log_time(7);
  sortParticles(numParticles);
  log_time(6);

  countNonEmptyBoxes(numParticles);

  allocate();

  numLevel = maxLevel;

  levelOffset[numLevel-1] = 0;

  kernel.precalc();

  getBoxData(numParticles,numBoxIndex);

// P2P

  getInteractionList(numBoxIndex,numLevel,0);

  for( i=0; i<numParticles; i++ ) {
    bodyAccel[i].x = 0;
    bodyAccel[i].y = 0;
    bodyAccel[i].z = 0;
  }

  log_time(7);
  kernel.p2p(numBoxIndex);
  log_time(0);

  numLevel = maxLevel;

// P2M

  kernel.p2m(numBoxIndex);
  log_time(1);

  if(maxLevel > 2) {

    for( numLevel=maxLevel-1; numLevel>=2; numLevel-- ) {

      if( treeOrFMM == 0 ) {

// M2P at lower levels

        getInteractionList(numBoxIndex,numLevel+1,2);

        log_time(7);
        kernel.m2p(numBoxIndex,numLevel+1);
        log_time(3);

      }

// M2M

      numBoxIndexOld = numBoxIndex;

      getBoxDataOfParent(numBoxIndex,numLevel,treeOrFMM);

      log_time(7);
      kernel.m2m(numBoxIndex,numBoxIndexOld,numLevel);
      log_time(2);

    }

    numLevel = 2;

  } else {

    getBoxIndexMask(numBoxIndex,numLevel);

  }

  if( treeOrFMM == 0 ) {

// M2P at level 2

    getInteractionList(numBoxIndex,numLevel,1);

    log_time(7);
    kernel.m2p(numBoxIndex,numLevel);
    log_time(3);

  } else {

// M2L at level 2

    getInteractionList(numBoxIndex,numLevel,1);

    log_time(7);
    kernel.m2l(numBoxIndex,numLevel);
    log_time(3);


// L2L

    if( maxLevel > 2 ) {

      for( numLevel=3; numLevel<=maxLevel; numLevel++ ) {

        numBoxIndex = levelOffset[numLevel-2]-levelOffset[numLevel-1];

        log_time(7);
        kernel.l2l(numBoxIndex,numLevel);
        log_time(4);

        getBoxIndexMask(numBoxIndex,numLevel);

// M2L at lower levels

        getInteractionList(numBoxIndex,numLevel,2);

        log_time(7);
        kernel.m2l(numBoxIndex,numLevel);
        log_time(3);

      }

      numLevel = maxLevel;

    }

// L2P

    log_time(7);
    kernel.l2p(numBoxIndex);
    log_time(5);

  }

  unsortParticles(numParticles);

  deallocate();
  log_time(7);
}
