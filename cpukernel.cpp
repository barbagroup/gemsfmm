#include "fmm.h"

float Anm[numExpansion4];
float anm[4*numExpansion2];
FmmSystem tree;

void cart2sph(double& r, double& theta, double& phi, double dx, double dy, double dz) {
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

// direct summation kernel
void FmmKernel::direct(int n) {
  int i,j;
  vec3<double> dist;
  double invDist,invDistCube;
  for( i=0; i<n; i++ ) {
    vec3<double> ai = {0.0, 0.0, 0.0};
    for( j=0; j<n; j++ ){
      dist.x = bodyPos[i].x-bodyPos[j].x;
      dist.y = bodyPos[i].y-bodyPos[j].y;
      dist.z = bodyPos[i].z-bodyPos[j].z;
      invDist = 1.0/sqrt(dist.x*dist.x+dist.y*dist.y+dist.z*dist.z+eps);
      invDistCube = bodyPos[j].w*invDist*invDist*invDist;
      ai.x -= dist.x*invDistCube;
      ai.y -= dist.y*invDistCube;
      ai.z -= dist.z*invDistCube;
    }
    bodyAccel[i].x = inv4PI*ai.x;
    bodyAccel[i].y = inv4PI*ai.y;
    bodyAccel[i].z = inv4PI*ai.z;
  }
}

// precalculate M2L translation matrix and Wigner rotation matrix
void FmmKernel::precalc() {
  int n,m,nm,nabsm,j,k,nk,npn,nmn,npm,nmm,nmk,i,nmk1,nm1k,nmk2;
  vec3<int> boxIndex3D;
  vec3<double> dist;
  double anmk[2][numExpansion4];
  double Dnmd[numExpansion4];
  double fnma,fnpa,pn,p,p1,p2,anmd,anmkd,rho,alpha,beta,sc,ank,ek;
  std::complex<double> expBeta[numExpansion2],I(0.0,1.0);

  int jk,jkn,jnk;
  double fnmm,fnpm,fad;

  for( n=0; n<2*numExpansions; n++ ) {
    for( m=-n; m<=n; m++ ) {
      nm = n*n+n+m;
      nabsm = abs(m);
      fnmm = 1.0;
      for( i=1; i<=n-m; i++ ) fnmm *= i;
      fnpm = 1.0;
      for( i=1; i<=n+m; i++ ) fnpm *= i;
      fnma = 1.0;
      for( i=1; i<=n-nabsm; i++ ) fnma *= i;
      fnpa = 1.0;
      for( i=1; i<=n+nabsm; i++ ) fnpa *= i;
      factorial[nm] = sqrt(fnma/fnpa);
      fad = sqrt(fnmm*fnpm);
      anm[nm] = pow(-1.0,n)/fad;
    }
  }

  for( j=0; j<numExpansions; j++) {
    for( k=-j; k<=j; k++ ){
      jk = j*j+j+k;
      for( n=abs(k); n<numExpansions; n++ ) {
        nk = n*n+n+k;
        jkn = jk*numExpansion2+nk;
        jnk = (j+n)*(j+n)+j+n;
        Anm[jkn] = pow(-1.0,j+k)*anm[nk]*anm[jk]/anm[jnk];
      }
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
    dist.x = boxIndex3D.x-3;
    dist.y = boxIndex3D.y-3;
    dist.z = boxIndex3D.z-3;
    cart2sph(rho,alpha,beta,dist.x,dist.y,dist.z);

    sc = sin(alpha)/(1+cos(alpha));
    for( n=0; n<4*numExpansions-3; n++ ) {
      expBeta[n] = exp((n-2*numExpansions+2)*beta*I);
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
      expBeta[n] = exp((n-2*numExpansions+2)*beta*I);
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

// Spherical harmonic rotation
void FmmKernel::rotation(std::complex<double>* Cnm, std::complex<double>* CnmOut, std::complex<double>** Dnm) {
  int n,m,nms,k,nk,nks;
  std::complex<double> CnmScalar;

  for( n=0; n<numExpansions; n++ ) {
    for( m=0; m<=n; m++ ) {
      nms = n*(n+1)/2+m;
      CnmScalar = 0;
      for( k=-n; k<=-1; k++ ) {
        nk = n*(n+1)+k;
        nks = n*(n+1)/2-k;
        CnmScalar += Dnm[m][nk]*conj(Cnm[nks]);
      }
      for( k=0; k<=n; k++ ) {
        nk = n*(n+1)+k;
        nks = n*(n+1)/2+k;
        CnmScalar += Dnm[m][nk]*Cnm[nks];
      }
      CnmOut[nms] = CnmScalar;
    }
  }
}

// p2p
void FmmKernel::p2p(int numBoxIndex) {
  int ii,ij,jj,i,j;
  vec3<double> dist;

  for( ii=0; ii<numBoxIndex; ii++ ) {
    for( ij=0; ij<numInteraction[ii]; ij++ ) {
      jj = interactionList[ii][ij];
      for( i=particleOffset[0][ii]; i<=particleOffset[1][ii]; i++ ) {
        vec3<double> ai = {0.0, 0.0, 0.0};
        for( j=particleOffset[0][jj]; j<=particleOffset[1][jj]; j++ ) {
          dist.x = bodyPos[i].x-bodyPos[j].x;
          dist.y = bodyPos[i].y-bodyPos[j].y;
          dist.z = bodyPos[i].z-bodyPos[j].z;
          double invDist = 1.0/sqrt(dist.x*dist.x+dist.y*dist.y+dist.z*dist.z+eps);
          double invDistCube = invDist*invDist*invDist;
          double s = bodyPos[j].w*invDistCube;
          ai.x -= dist.x*s;
          ai.y -= dist.y*s;
          ai.z -= dist.z*s;
        }
        bodyAccel[i].x += inv4PI*ai.x;
        bodyAccel[i].y += inv4PI*ai.y;
        bodyAccel[i].z += inv4PI*ai.z;
      }
    }
  }
}

// p2m
void FmmKernel::p2m(int numBoxIndex) {
  int jj,j,n,m,nm,nms;
  vec3<int> boxIndex3D;
  vec3<float> boxCenter;
  vec3<double> dist;
  double boxSize,rho,alpha,beta;
  double xx,s2,fact,pn,p,p1,p2,rhom,rhon;
  double YnmReal[numExpansion2];
  std::complex<double> MnmVector[numCoefficients],I(0.0,1.0),eim;

  boxSize = rootBoxSize/(1 << maxLevel);
  for( jj=0; jj<numBoxIndex; jj++ ) {
    tree.unmorton(boxIndexFull[jj],boxIndex3D);
    boxCenter.x = boxMin.x+(boxIndex3D.x+0.5)*boxSize;
    boxCenter.y = boxMin.y+(boxIndex3D.y+0.5)*boxSize;
    boxCenter.z = boxMin.z+(boxIndex3D.z+0.5)*boxSize;
    for( j=0; j<numCoefficients; j++ ) {
      MnmVector[j] = 0;
    }
    for( j=particleOffset[0][jj]; j<=particleOffset[1][jj]; j++ ) {
      dist.x = bodyPos[j].x-boxCenter.x;
      dist.y = bodyPos[j].y-boxCenter.y;
      dist.z = bodyPos[j].z-boxCenter.z;
      cart2sph(rho,alpha,beta,dist.x,dist.y,dist.z);
      xx = cos(alpha);
      s2 = sqrt((1-xx)*(1+xx));
      fact = 1;
      pn = 1;
      rhom = 1;
      for( m=0; m<numExpansions; m++ ) {
        p = pn;
        nm = m*m+2*m;
        YnmReal[nm] = rhom*factorial[nm]*p;
        p1 = p;
        p = xx*(2*m+1)*p;
        rhom *= rho;
        rhon = rhom;
        for( n=m+1; n<numExpansions; n++ ) {
          nm = n*n+n+m;
          YnmReal[nm] = rhon*factorial[nm]*p;
          p2 = p1;
          p1 = p;
          p = (xx*(2*n+1)*p1-(n+m)*p2)/(n-m+1);
          rhon *=rho;
        }
        pn = -pn*fact*s2;
        fact += 2;
      }
      for( n=0; n<numExpansions; n++ ) {
        for( m=0; m<=n; m++ ) {
          nm = n*n+n+m;
          nms = n*(n+1)/2+m;
          eim = exp(-m*beta*I);
          MnmVector[nms] += ((std::complex<double>) bodyPos[j].w)*YnmReal[nm]*eim;
        }
      }
    }
    for( j=0; j<numCoefficients; j++ ) {
      Mnm[jj][j] = MnmVector[j];
    }
  }
}

// m2m
void FmmKernel::m2m(int numBoxIndex, int numBoxIndexOld, int numLevel) {
  int ii,ib,j,jj,nfjp,nfjc,jb,je,k,jk,jks,n,jnk,jnks,nm;
  vec3<int> boxIndex3D;
  double boxSize,rho;
  std::complex<double> cnm,MnmScalar;
  std::complex<double> MnmVectorB[numCoefficients],MnmVectorA[numCoefficients];

  boxSize = rootBoxSize/(1 << numLevel);
  for( ii=0; ii<numBoxIndex; ii++ ) {
    ib = ii+levelOffset[numLevel-1];
    for( j=0; j<numCoefficients; j++ ) {
      Mnm[ib][j] = 0;
    }
  }
  for( jj=0; jj<numBoxIndexOld; jj++ ) {
    jb = jj+levelOffset[numLevel];
    nfjp = boxIndexFull[jb]/8;
    nfjc = boxIndexFull[jb]%8;
    ib = boxIndexMask[nfjp]+levelOffset[numLevel-1];
    tree.unmorton(nfjc,boxIndex3D);
    boxIndex3D.x = 4-boxIndex3D.x*2;
    boxIndex3D.y = 4-boxIndex3D.y*2;
    boxIndex3D.z = 4-boxIndex3D.z*2;
    tree.morton1(boxIndex3D,je,3);
    rho = boxSize*sqrt(3.0)/4;
    for( j=0; j<numCoefficients; j++ ) {
      MnmVectorA[j] = Mnm[jb][j];
    }
    rotation(MnmVectorA,MnmVectorB,Dnm[je]);
    for( j=0; j<numExpansions; j++ ) {
      for( k=0; k<=j; k++ ) {
        jk = j*j+j+k;
        jks = j*(j+1)/2+k;
        MnmScalar = 0;
        for( n=0; n<=j-abs(k); n++ ) {
          jnk = (j-n)*(j-n)+j-n+k;
          jnks = (j-n)*(j-n+1)/2+k;
          nm = n*n+n;
          cnm = pow(-1.0,n)*anm[nm]*anm[jnk]/anm[jk]*pow(rho,n)*Ynm[nm];
          MnmScalar += MnmVectorB[jnks]*cnm;
        }
        MnmVectorA[jks] = MnmScalar;
      }
    }
    rotation(MnmVectorA,MnmVectorB,Dnm[je+numRelativeBox]);
    for( j=0; j<numCoefficients; j++ ) {
      Mnm[ib][j] += MnmVectorB[j];
    }
  }
}

// m2l
void FmmKernel::m2l(int numBoxIndex, int numLevel) {
  int i,j,ii,ib,ix,iy,iz,ij,jj,jb,jx,jy,jz,je,k,jk,jks,n,nk,nks,jkn,jnk;
  vec3<int> boxIndex3D;
  vec3<double> dist;
  double boxSize,rho,rhoj,rhojk,rhojn;
  std::complex<double> LnmVectorA[numCoefficients],MnmVectorA[numCoefficients];
  std::complex<double> LnmVectorB[numCoefficients],MnmVectorB[numCoefficients];
  std::complex<double> cnm,LnmScalar;

  boxSize = rootBoxSize/(1 << numLevel);
  if( numLevel == 2 ) {
    for( i=0; i<numBoxIndex; i++ ) {
      for( j=0; j<numCoefficients; j++ ) {
        Lnm[i][j] = 0;
      }
    }
  }
  for( ii=0; ii<numBoxIndex; ii++ ) {
    ib = ii+levelOffset[numLevel-1];
    tree.unmorton(boxIndexFull[ib],boxIndex3D);
    ix = boxIndex3D.x;
    iy = boxIndex3D.y;
    iz = boxIndex3D.z;
    for( ij=0; ij<numInteraction[ii]; ij++ ) {
      jj = interactionList[ii][ij];
      jb = jj+levelOffset[numLevel-1];
      for( j=0; j<numCoefficients; j++ ) {
        MnmVectorB[j] = Mnm[jb][j];
      }
      tree.unmorton(boxIndexFull[jb],boxIndex3D);
      jx = boxIndex3D.x;
      jy = boxIndex3D.y;
      jz = boxIndex3D.z;
      dist.x = (ix-jx)*boxSize;
      dist.y = (iy-jy)*boxSize;
      dist.z = (iz-jz)*boxSize;
      boxIndex3D.x = (ix-jx)+3;
      boxIndex3D.y = (iy-jy)+3;
      boxIndex3D.z = (iz-jz)+3;
      tree.morton1(boxIndex3D,je,3);
      rho = sqrt(dist.x*dist.x+dist.y*dist.y+dist.z*dist.z)+eps;
      rotation(MnmVectorB,MnmVectorA,Dnm[je]);
      rhoj = 1;
      for( j=0; j<numExpansions; j++ ) {
        rhojk = rhoj;
        rhoj *= rho;
        for( k=0; k<=j; k++ ) {
          jk = j*j+j+k;
          jks = j*(j+1)/2+k;
          LnmScalar = 0;
          rhojn = rhojk;
          rhojk *= rho;
          for( n=abs(k); n<numExpansions; n++ ) {
            rhojn *= rho;
            nk = n*n+n+k;
            nks = n*(n+1)/2+k;
            jkn = jk*numExpansion2+nk;
            jnk = (j+n)*(j+n)+j+n;
            cnm = Anm[jkn]/rhojn*Ynm[jnk];
            LnmScalar += MnmVectorA[nks]*cnm;
          }
          LnmVectorA[jks] = LnmScalar;
        }
      }
      rotation(LnmVectorA,LnmVectorB,Dnm[je+numRelativeBox]);
      for( j=0; j<numCoefficients; j++ ) {
        Lnm[ii][j] += LnmVectorB[j];
      }
    }
  }
  for( jj=0; jj<numBoxIndex; jj++ ) {
    jb = jj+levelOffset[numLevel-1];
    for( j=0; j<numCoefficients; j++ ) {
      Mnm[jb][j] = 0;
    }
  }
}

// l2l
void FmmKernel::l2l(int numBoxIndex, int numLevel) {
  int numBoxIndexOld,ii,ib,i,nfip,nfic,je,j,k,jk,jks,n,jnk,nk,nks;
  vec3<int> boxIndex3D;
  double boxSize,rho;
  std::complex<double> cnm,LnmScalar;
  std::complex<double> LnmVectorA[numCoefficients],LnmVectorB[numCoefficients];

  boxSize = rootBoxSize/(1 << numLevel);
  numBoxIndexOld = numBoxIndex;
  if( numBoxIndexOld < 8 ) numBoxIndexOld = 8;
  for( ii=0; ii<numBoxIndexOld; ii++ ) {
    for( i=0; i<numCoefficients; i++ ) {
      LnmOld[ii][i] = Lnm[ii][i];
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

  for( ii=0; ii<numBoxIndex; ii++ ) {
    ib = ii+levelOffset[numLevel-1];
    nfip = boxIndexFull[ib]/8;
    nfic = boxIndexFull[ib]%8;
    tree.unmorton(nfic,boxIndex3D);
    boxIndex3D.x = boxIndex3D.x*2+2;
    boxIndex3D.y = boxIndex3D.y*2+2;
    boxIndex3D.z = boxIndex3D.z*2+2;
    tree.morton1(boxIndex3D,je,3);
    rho = boxSize*sqrt(3.0)/2;
    ib = neo[nfip];
    for( i=0; i<numCoefficients; i++ ) {
      LnmVectorA[i] = LnmOld[ib][i];
    }
    rotation(LnmVectorA,LnmVectorB,Dnm[je]);
    for( j=0; j<numExpansions; j++ ) {
      for( k=0; k<=j; k++ ) {
        jk = j*j+j+k;
        jks = j*(j+1)/2+k;
        LnmScalar = 0;
        for( n=j; n<numExpansions; n++ ) {
          jnk = (n-j)*(n-j)+n-j;
          nk = n*n+n+k;
          nks = n*(n+1)/2+k;
          cnm = anm[jnk]*anm[jk]/anm[nk]*pow(rho,n-j)*Ynm[jnk];
          LnmScalar += LnmVectorB[nks]*cnm;
        }
        LnmVectorA[jks] = LnmScalar;
      }
    }
    rotation(LnmVectorA,LnmVectorB,Dnm[je+numRelativeBox]);
    for( i=0; i<numCoefficients; i++ ) {
      Lnm[ii][i] = LnmVectorB[i];
    }
  }
}

// l2p
void FmmKernel::l2p(int numBoxIndex) {
  int ii,i,n,nm,nms,m;
  vec3<int> boxIndex3D;
  vec3<float> boxCenter;
  vec3<double> accel,dist;
  double boxSize,r,theta,phi,accelR,accelTheta,accelPhi;
  double xx,yy,s2,fact,pn,p,p1,p2,rn;
  double YnmReal[numExpansion2],YnmRealTheta[numExpansion2];
  std::complex<double> LnmVector[numCoefficients];
  std::complex<double> rr,rtheta,rphi,I(0.0,1.0),eim;

  boxSize = rootBoxSize/(1 << maxLevel);
  for( ii=0; ii<numBoxIndex; ii++ ) {
    tree.unmorton(boxIndexFull[ii],boxIndex3D);
    boxCenter.x = boxMin.x+(boxIndex3D.x+0.5)*boxSize;
    boxCenter.y = boxMin.y+(boxIndex3D.y+0.5)*boxSize;
    boxCenter.z = boxMin.z+(boxIndex3D.z+0.5)*boxSize;
    for( i=0; i<numCoefficients; i++ ) LnmVector[i] = Lnm[ii][i];
    for( i=particleOffset[0][ii]; i<=particleOffset[1][ii]; i++ ) {
      dist.x = bodyPos[i].x-boxCenter.x;
      dist.y = bodyPos[i].y-boxCenter.y;
      dist.z = bodyPos[i].z-boxCenter.z;
      cart2sph(r,theta,phi,dist.x,dist.y,dist.z);
      xx = cos(theta);
      yy = sin(theta);
      s2 = sqrt((1-xx)*(1+xx));
      fact = 1;
      pn = 1;
      for( m=0; m<numExpansions; m++ ) {
        p = pn;
        nm = m*m+2*m;
        YnmReal[nm] = factorial[nm]*p;
        p1 = p;
        p = xx*(2*m+1)*p;
        YnmRealTheta[nm] = factorial[nm]*(p-(m+1)*xx*p1)/yy;
        for( n=m+1; n<numExpansions; n++ ) {
          nm = n*n+n+m;
          YnmReal[nm] = factorial[nm]*p;
          p2 = p1;
          p1 = p;
          p = (xx*(2*n+1)*p1-(n+m)*p2)/(n-m+1);
          YnmRealTheta[nm] = factorial[nm]*((n-m+1)*p-(n+1)*xx*p1)/yy;
        }
        pn = -pn*fact*s2;
        fact += 2;
      }
      accelR = 0;
      accelTheta = 0;
      accelPhi = 0;
      rn = 1;
      for( n=0; n<numExpansions; n++ ) {
        nm = n*n+n;
        nms = n*(n+1)/2;
        rr = n*rn/r*YnmReal[nm];
        rtheta = rn*YnmRealTheta[nm];
        accelR += real(rr*LnmVector[nms]);
        accelTheta += real(rtheta*LnmVector[nms]);
        for( m=1; m<=n; m++ ) {
          nm = n*n+n+m;
          nms = n*(n+1)/2+m;
          eim = exp(m*phi*I);
          rr = n*rn/r*YnmReal[nm]*eim;
          rtheta = rn*YnmRealTheta[nm]*eim;
          rphi = m*rn*YnmReal[nm]*eim*I;
          accelR += 2*real(rr*LnmVector[nms]);
          accelTheta += 2*real(rtheta*LnmVector[nms]);
          accelPhi += 2*real(rphi*LnmVector[nms]);
        }
        rn *= r;
      }
      accel.x = sin(theta)*cos(phi)*accelR+cos(theta)*cos(phi)/r*accelTheta-sin(phi)/r/sin(theta)*accelPhi;
      accel.y = sin(theta)*sin(phi)*accelR+cos(theta)*sin(phi)/r*accelTheta+cos(phi)/r/sin(theta)*accelPhi;
      accel.z = cos(theta)*accelR-sin(theta)/r*accelTheta;
      bodyAccel[i].x += inv4PI*accel.x;
      bodyAccel[i].y += inv4PI*accel.y;
      bodyAccel[i].z += inv4PI*accel.z;
    }
  }
}

// m2p
void FmmKernel::m2p(int numBoxIndex, int numLevel) {
  int ii,i,ij,jj,jb,j,n,nm,nms,m;
  vec3<int> boxIndex3D;
  vec3<float> boxCenter;
  vec3<double> accel,dist;
  double boxSize,r,theta,phi,rn,accelR,accelTheta,accelPhi;
  double xx,yy,s2,fact,pn,p,p1,p2;
  double YnmReal[numExpansion2],YnmRealTheta[numExpansion2];
  std::complex<double> MnmVector[numCoefficients];
  std::complex<double> rr,rtheta,rphi,I(0.0,1.0),eim;

  boxSize = rootBoxSize/(1 << numLevel);
  for( ii=0; ii<numBoxIndex; ii++ ) {
    for( i=particleOffset[0][ii]; i<=particleOffset[1][ii]; i++ ) {
      for( ij=0; ij<numInteraction[ii]; ij++ ) {
        jj = interactionList[ii][ij];
        jb = jj+levelOffset[numLevel-1];
        for( j=0; j<numCoefficients; j++ ) MnmVector[j] = Mnm[jb][j];
        tree.unmorton(boxIndexFull[jb],boxIndex3D);
        boxCenter.x = boxMin.x+(boxIndex3D.x+0.5)*boxSize;
        boxCenter.y = boxMin.y+(boxIndex3D.y+0.5)*boxSize;
        boxCenter.z = boxMin.z+(boxIndex3D.z+0.5)*boxSize;
        dist.x = bodyPos[i].x-boxCenter.x;
        dist.y = bodyPos[i].y-boxCenter.y;
        dist.z = bodyPos[i].z-boxCenter.z;
        cart2sph(r,theta,phi,dist.x,dist.y,dist.z);
        xx = cos(theta);
        yy = sin(theta);
        s2 = sqrt((1-xx)*(1+xx));
        fact = 1;
        pn = 1;
        for( m=0; m<numExpansions; m++ ) {
          p = pn;
          nm = m*m+2*m;
          YnmReal[nm] = factorial[nm]*p;
          p1 = p;
          p = xx*(2*m+1)*p;
          YnmRealTheta[nm] = factorial[nm]*(p-(m+1)*xx*p1)/yy;
          for( n=m+1; n<numExpansions; n++ ) {
            nm = n*n+n+m;
            YnmReal[nm] = factorial[nm]*p;
            p2 = p1;
            p1 = p;
            p = (xx*(2*n+1)*p1-(n+m)*p2)/(n-m+1);
            YnmRealTheta[nm] = factorial[nm]*((n-m+1)*p-(n+1)*xx*p1)/yy;
          }
          pn = -pn*fact*s2;
          fact += 2;
        }
        accelR = 0;
        accelTheta = 0;
        accelPhi = 0;
        rn = 1/r;
        for( n=0; n<numExpansions; n++ ) {
          rn /= r;
          nm = n*n+n;
          nms = n*(n+1)/2;
          rr = -(n+1)*rn*YnmReal[nm];
          rtheta = rn*r*YnmRealTheta[nm];
          accelR += real(rr*MnmVector[nms]);
          accelTheta += real(rtheta*MnmVector[nms]);
          for( m=1; m<=n; m++ ) {
            nm = n*n+n+m;
            nms = n*(n+1)/2+m;
            eim = exp(m*phi*I);
            rr = -(n+1)*rn*YnmReal[nm]*eim;
            rtheta = rn*r*YnmRealTheta[nm]*eim;
            rphi = m*rn*r*YnmReal[nm]*eim*I;
            accelR += 2*real(rr*MnmVector[nms]);
            accelTheta += 2*real(rtheta*MnmVector[nms]);
            accelPhi += 2*real(rphi*MnmVector[nms]);
          }
        }
        accel.x = sin(theta)*cos(phi)*accelR+cos(theta)*cos(phi)/r*accelTheta-sin(phi)/r/sin(theta)*accelPhi;
        accel.y = sin(theta)*sin(phi)*accelR+cos(theta)*sin(phi)/r*accelTheta+cos(phi)/r/sin(theta)*accelPhi;
        accel.z = cos(theta)*accelR-sin(theta)/r*accelTheta;
        bodyAccel[i].x += inv4PI*accel.x;
        bodyAccel[i].y += inv4PI*accel.y;
        bodyAccel[i].z += inv4PI*accel.z;
      }
    }
  }
}
