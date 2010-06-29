// summation
#define AX      "%xmm0"
#define AY      "%xmm1"
#define AZ      "%xmm2"
#define PHI     "%xmm3"
// j particle
#define XJ      "%xmm4"
#define YJ      "%xmm5"
#define ZJ      "%xmm6"
#define MJ      "%xmm7"
// temporary
#define RINV    "%xmm8"
#define X2      "%xmm9"
#define Y2      "%xmm10"
#define Z2      "%xmm11"
// fixed i particle
#define XI      "%xmm12"
#define YI      "%xmm13"
#define ZI      "%xmm14"
#define R2      "%xmm15"

#define XORPS(a, b) asm("xorps "  a  ","  b );
#define LOADPS(mem, reg) asm("movaps %0, %"reg::"m"(mem));
#define STORPS(reg, mem) asm("movaps %"reg " , %0"::"m"(mem));
#define MOVAPS(src, dst) asm("movaps " src "," dst);
#define MOVQ(src, dst) asm("movq " src "," dst);
#define BCAST0(reg) asm("shufps $0x00, " reg ","  reg);
#define BCAST1(reg) asm("shufps $0x55, " reg ","  reg);
#define BCAST2(reg) asm("shufps $0xaa, " reg ","  reg);
#define BCAST3(reg) asm("shufps $0xff, " reg ","  reg);
#define MULPS(src, dst) asm("mulps " src "," dst);
#define ADDPS(src, dst) asm("addps " src ","  dst);
#define SUBPS(src, dst) asm("subps "  src "," dst);
#define RSQRTPS(src, dst) asm("rsqrtps " src "," dst);
#define MOVHLPS(src, dst) asm("movhlps " src "," dst);
#define DEBUGPS(reg)

#define ALIGN16 __attribute__ ((aligned(16)))

#define NJMAX (1<<24)

typedef double v2df __attribute__ ((vector_size(16)));
typedef float  v4sf __attribute__ ((vector_size(16)));
typedef int    v4si __attribute__ ((vector_size(16)));
typedef short  v8hi __attribute__ ((vector_size(16)));

typedef struct iptdata{
  float x[4];
  float y[4];
  float z[4];
  float eps2[4]; // not used in this implementation
} Ipdata ALIGN16;
typedef struct fodata{
  float ax[4];
  float ay[4];
  float az[4];
  float phi[4];
} Fodata ALIGN16;
typedef struct jpdata{
  float x, y, z, m;
} Jpdata ALIGN16;

void p2p_kernel(Ipdata *ipdata, Fodata *fodata, Jpdata *jpdata, int nj){
  int j;
  assert(((unsigned long)jpdata & 15) == 0);
  assert(((unsigned long)ipdata & 15) == 0);
  assert(((unsigned long)fodata & 15) == 0);

  XORPS(AX, AX);               // AX = 0
  XORPS(AY, AY);               // AY = 0
  XORPS(AZ, AZ);               // AZ = 0
  XORPS(PHI, PHI);             // PHI = 0

  LOADPS(*ipdata->x, XI);      // XI = *ipdata->x
  LOADPS(*ipdata->y, YI);      // YI = *ipdata->y
  LOADPS(*ipdata->z, ZI);      // ZI = *ipdata->z
  LOADPS(*ipdata->eps2, R2);   // R2 = *ipdata->eps2

  LOADPS(jpdata[0], MJ);       // MJ = *jpdata->x,y,z,m
  MOVAPS(MJ, X2);              // X2 = MJ
  MOVAPS(MJ, Y2);              // Y2 = MJ
  MOVAPS(MJ, Z2);              // Z2 = MJ

  BCAST0(X2);                  // X2 = *jpdata->x
  BCAST1(Y2);                  // Y2 = *jpdata->y
  BCAST2(Z2);                  // Z2 = *jpdata->z
  BCAST3(MJ);                  // MJ = *jpdata->m

  SUBPS(XI, X2);               // X2 = X2 - XI
  SUBPS(YI, Y2);               // Y2 = Y2 - YI
  SUBPS(ZI, Z2);               // Z2 = Z2 - ZI

  MOVAPS(X2, XJ);              // XJ = X2
  MOVAPS(Y2, YJ);              // YJ = Y2
  MOVAPS(Z2, ZJ);              // ZJ = Z2

  MULPS(X2, X2);               // X2 = X2 * X2
  MULPS(Y2, Y2);               // Y2 = Y2 * Y2
  MULPS(Z2, Z2);               // Z2 = Z2 * Z2

  ADDPS(X2, R2);               // R2 = R2 + X2
  ADDPS(Y2, R2);               // R2 = R2 + Y2
  ADDPS(Z2, R2);               // R2 = R2 + Z2

  LOADPS(jpdata[1], X2);       // X2 = *jpdata->x,y,z,m
  MOVAPS(X2, Y2);              // Y2 = X2
  MOVAPS(X2, Z2);              // Z2 = X2
  for(j=0;j<nj;j++){
    RSQRTPS(R2, RINV);         // RINV = rsqrt(R2)
    jpdata++;
    LOADPS(*ipdata->eps2, R2); // R2 = *ipdata->eps2
    BCAST0(X2);                // X2 = *jpdata->x
    BCAST1(Y2);                // Y2 = *jpdata->y
    BCAST2(Z2);                // Z2 = *jpdata->z
    SUBPS(XI, X2);             // X2 = X2 - XI
    SUBPS(YI, Y2);             // Y2 = Y2 - YI
    SUBPS(ZI, Z2);             // Z2 = Z2 - ZI

    MULPS(RINV, MJ);           // MJ = MJ * RINV
    SUBPS(MJ, PHI);            // PHI = PHI - MJ
    MULPS(RINV, RINV);         // RINV = RINV * RINV
    MULPS(MJ, RINV);           // RINV = MJ * RINV
    LOADPS(jpdata[0], MJ);     // MJ = *jpdata->x,y,z,m
    BCAST3(MJ);                // MJ = *jpdata->m

    MULPS(RINV, XJ);           // XJ = XJ * RINV
    ADDPS(XJ, AX);             // AX = AX + XJ
    MOVAPS(X2, XJ);            // XJ = X2
    MULPS(X2, X2);             // X2 = X2 * X2
    ADDPS(X2, R2);             // R2 = R2 + X2
    LOADPS(jpdata[1], X2);     // X2 = *jpdata->x,y,z,m

    MULPS(RINV, YJ);           // YJ = YJ * RINV
    ADDPS(YJ, AY);             // AY = AY + YJ
    MOVAPS(Y2, YJ);            // YJ = Y2
    MULPS(Y2, Y2);             // Y2 = Y2 * Y2
    ADDPS(Y2, R2);             // R2 = R2 + Y2
    MOVAPS(X2, Y2);            // Y2 = X2

    MULPS(RINV, ZJ);           // ZJ = ZJ * RINV
    ADDPS(ZJ, AZ);             // AZ = AZ + ZJ
    MOVAPS(Z2, ZJ);            // ZJ = Z2
    MULPS(Z2, Z2);             // Z2 = Z2 * Z2
    ADDPS(Z2, R2);             // R2 = R2 + Z2
    MOVAPS(X2, Z2);            // Z2 = X2
  }
  STORPS(AX, *fodata->ax);     // AX = *fodata->ax
  STORPS(AY, *fodata->ay);     // AY = *fodata->ay
  STORPS(AZ, *fodata->az);     // AZ = *fodata->az
  STORPS(PHI, *fodata->phi);   // PHI = *fodata->phi
}

static inline void v4sf_transpose(
    v4sf *d0, v4sf *d1, v4sf *d2, v4sf *d3,
    v4sf  s0, v4sf  s1, v4sf  s2, v4sf  s3)
{
  *d0 = __builtin_ia32_unpcklps(
        __builtin_ia32_unpcklps(s0, s2),
        __builtin_ia32_unpcklps(s1, s3));
  *d1 = __builtin_ia32_unpckhps(
        __builtin_ia32_unpcklps(s0, s2),
        __builtin_ia32_unpcklps(s1, s3));
  *d2 = __builtin_ia32_unpcklps(
        __builtin_ia32_unpckhps(s0, s2),
        __builtin_ia32_unpckhps(s1, s3));
  *d3 = __builtin_ia32_unpckhps(
        __builtin_ia32_unpckhps(s0, s2),
        __builtin_ia32_unpckhps(s1, s3));
}

static inline void v3sf_store_sp(v4sf vec, float *d0, float *d1, float *d2){
  *d0 = __builtin_ia32_vec_ext_v4sf(vec, 0);
  *d1 = __builtin_ia32_vec_ext_v4sf(vec, 1);
  *d2 = __builtin_ia32_vec_ext_v4sf(vec, 2);
}
