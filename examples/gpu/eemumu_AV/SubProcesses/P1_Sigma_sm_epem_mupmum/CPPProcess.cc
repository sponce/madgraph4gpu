//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.7.3.py3, 2020-06-28
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>

#include "mgOnGpuConfig.h"
using mgOnGpu::dcomplex;
using mgOnGpu::double_v;
using mgOnGpu::dcomplex_v;

namespace MG5_sm
{

#ifndef __CUDACC__
  // Quick and dirty way to share nevt across all computational kernels
  int nevt;
#endif

  using mgOnGpu::nw6;

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  inline const double& pIparIp4Ievt( const double* allmomenta, // input[(npar=4)*(np4=4)*nevt]
                                     const int ipar,
                                     const int ip4,
                                     const int ievt )
  {
    using mgOnGpu::np4;
    using mgOnGpu::npar;
#if defined MGONGPU_LAYOUT_ASA
    using mgOnGpu::nepp;
    const int ipag = ievt/nepp; // #eventpage in this iteration
    const int iepp = ievt%nepp; // #event in the current eventpage in this iteration
    // ASA: allmomenta[npag][npar][np4][nepp]
    return allmomenta[ipag*npar*np4*nepp + ipar*nepp*np4 + ip4*nepp + iepp]; // AOSOA[ipag][ipar][ip4][iepp]
#elif defined MGONGPU_LAYOUT_SOA
#ifdef __CUDACC__
    const int nevt = blockDim.x * gridDim.x;
#else
    using MG5_sm::nevt;
#endif
    // SOA: allmomenta[npar][np4][ndim]
    return allmomenta[ipar*np4*nevt + ip4*nevt + ievt]; // SOA[ipar][ip4][ievt]
#elif defined MGONGPU_LAYOUT_AOS
    // AOS: allmomenta[ndim][npar][np4]
    return allmomenta[ievt*npar*np4 + ipar*np4 + ip4]; // AOS[ievt][ipar][ip4]
#endif
  }

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  void imzxxxM0( const double* allmomenta, // input[(npar=4)*(np4=4)*nevt]
                 //const double fmass,
                 const int nhel,
                 const int nsf,
#ifndef __CUDACC__
                 dcomplex fi[nw6],
                 const int ievt,
#else
#if defined MGONGPU_WFMEM_LOCAL
                 dcomplex fi[nw6],
#else
                 dcomplex* fiv,            // output: fiv[5 * 6 * #threads_in_block]
#endif
#endif
                 const int ipar )          // input: particle# out of npar
  {
#ifndef __CUDACC__
    // ** START LOOP ON IEVT **
    //for (int ievt = 0; ievt < nevt; ++ievt)
#endif
    {
#ifdef __CUDACC__
#if !defined MGONGPU_WFMEM_LOCAL
      const int neib = blockDim.x; // number of events (threads) in block
      const int ieib = threadIdx.x; // index of event (thread) in block
#endif
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
      //printf( "imzxxxM0: ievt=%d ieib=%d\n", ievt, threadIdx.x );
#endif
      const double& pvec0 = pIparIp4Ievt( allmomenta, ipar, 0, ievt );
      const double& pvec1 = pIparIp4Ievt( allmomenta, ipar, 1, ievt );
      const double& pvec2 = pIparIp4Ievt( allmomenta, ipar, 2, ievt );
      const double& pvec3 = pIparIp4Ievt( allmomenta, ipar, 3, ievt );
#if defined __CUDACC__ && !defined MGONGPU_WFMEM_LOCAL
      dcomplex& fi0 = fiv[ipar*nw6*neib + 0*neib + ieib];
      dcomplex& fi1 = fiv[ipar*nw6*neib + 1*neib + ieib];
      dcomplex& fi2 = fiv[ipar*nw6*neib + 2*neib + ieib];
      dcomplex& fi3 = fiv[ipar*nw6*neib + 3*neib + ieib];
      dcomplex& fi4 = fiv[ipar*nw6*neib + 4*neib + ieib];
      dcomplex& fi5 = fiv[ipar*nw6*neib + 5*neib + ieib];
#else
      dcomplex& fi0 = fi[0];
      dcomplex& fi1 = fi[1];
      dcomplex& fi2 = fi[2];
      dcomplex& fi3 = fi[3];
      dcomplex& fi4 = fi[4];
      dcomplex& fi5 = fi[5];
#endif
      fi0 = dcomplex( -pvec0 * nsf, -pvec3 * nsf );
      fi1 = dcomplex( -pvec1 * nsf, -pvec2 * nsf );
      const int nh = nhel * nsf;
      // ASSUMPTIONS FMASS = 0 and
      // (PX = PY = 0 and E = -P3 > 0)
      {
        const dcomplex chi0( 0, 0 );
        const dcomplex chi1( -nhel * sqrt(2 * pvec0), 0 );
        if (nh == 1)
        {
          fi2 = dcomplex( 0, 0 );
          fi3 = dcomplex( 0, 0 );
          fi4 = chi0;
          fi5 = chi1;
        }
        else
        {
          fi2 = chi1;
          fi3 = chi0;
          fi4 = dcomplex( 0, 0 );
          fi5 = dcomplex( 0, 0 );
        }
      }
    }
    // ** END LOOP ON IEVT **
    return;
  }

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  void ixzxxxM0( const double* allmomenta, // input[(npar=4)*(np4=4)*nevt]
                 //const double fmass,
                 const int nhel,
                 const int nsf,
#ifndef __CUDACC__
                 dcomplex fi[nw6],
                 const int ievt,
#else
#if defined MGONGPU_WFMEM_LOCAL
                 dcomplex fi[nw6],
#else
                 dcomplex* fiv,            // output: fiv[5 * 6 * #threads_in_block]
#endif
#endif
                 const int ipar )          // input: particle# out of npar
  {
#ifndef __CUDACC__
    // ** START LOOP ON IEVT **
    //for (int ievt = 0; ievt < nevt; ++ievt)
#endif
    {
#ifdef __CUDACC__
#if !defined MGONGPU_WFMEM_LOCAL
      const int neib = blockDim.x; // number of events (threads) in block
      const int ieib = threadIdx.x; // index of event (thread) in block
#endif
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
      //printf( "ixzxxxM0: ievt=%d ieib=%d\n", ievt, threadIdx.x );
#endif
      const double& pvec0 = pIparIp4Ievt( allmomenta, ipar, 0, ievt );
      const double& pvec1 = pIparIp4Ievt( allmomenta, ipar, 1, ievt );
      const double& pvec2 = pIparIp4Ievt( allmomenta, ipar, 2, ievt );
      const double& pvec3 = pIparIp4Ievt( allmomenta, ipar, 3, ievt );
#if defined __CUDACC__ && !defined MGONGPU_WFMEM_LOCAL
      dcomplex& fi0 = fiv[ipar*nw6*neib + 0*neib + ieib];
      dcomplex& fi1 = fiv[ipar*nw6*neib + 1*neib + ieib];
      dcomplex& fi2 = fiv[ipar*nw6*neib + 2*neib + ieib];
      dcomplex& fi3 = fiv[ipar*nw6*neib + 3*neib + ieib];
      dcomplex& fi4 = fiv[ipar*nw6*neib + 4*neib + ieib];
      dcomplex& fi5 = fiv[ipar*nw6*neib + 5*neib + ieib];
#else
      dcomplex& fi0 = fi[0];
      dcomplex& fi1 = fi[1];
      dcomplex& fi2 = fi[2];
      dcomplex& fi3 = fi[3];
      dcomplex& fi4 = fi[4];
      dcomplex& fi5 = fi[5];
#endif
      fi0 = dcomplex( -pvec0 * nsf, -pvec3 * nsf );
      fi1 = dcomplex( -pvec1 * nsf, -pvec2 * nsf );
      const int nh = nhel * nsf;
      // ASSUMPTIONS FMASS = 0 and
      // (PX and PY are not 0)
      {
        const double sqp0p3 = sqrt( pvec0 + pvec3 ) * nsf;
        const dcomplex chi0( sqp0p3, 0 );
        const dcomplex chi1( nh * pvec1 / sqp0p3, pvec2 / sqp0p3 );
        if ( nh == 1 )
        {
          fi2 = dcomplex( 0, 0 );
          fi3 = dcomplex( 0, 0 );
          fi4 = chi0;
          fi5 = chi1;
        }
        else
        {
          fi2 = chi1;
          fi3 = chi0;
          fi4 = dcomplex( 0, 0 );
          fi5 = dcomplex( 0, 0 );
        }
      }
    }
    // ** END LOOP ON IEVT **
    return;
  }

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  void oxzxxxM0( const double* allmomenta, // input[(npar=4)*(np4=4)*nevt]
                 //const double fmass,
                 const int nhel,
                 const int nsf,
#ifndef __CUDACC__
                 dcomplex fo[nw6],
                 const int ievt,
#else
#if defined MGONGPU_WFMEM_LOCAL
                 dcomplex fo[nw6],
#else
                 dcomplex* fov,            // output: fov[5 * 6 * #threads_in_block]
#endif
#endif
                 const int ipar )          // input: particle# out of npar
  {
#ifndef __CUDACC__
    // ** START LOOP ON IEVT **
    //for (int ievt = 0; ievt < nevt; ++ievt)
#endif
    {
#ifdef __CUDACC__
#if !defined MGONGPU_WFMEM_LOCAL
      const int neib = blockDim.x; // number of events (threads) in block
      const int ieib = threadIdx.x; // index of event (thread) in block
#endif
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
      //printf( "oxzxxxM0: ievt=%d ieib=%d\n", ievt, threadIdx.x );
#endif
      const double& pvec0 = pIparIp4Ievt( allmomenta, ipar, 0, ievt );
      const double& pvec1 = pIparIp4Ievt( allmomenta, ipar, 1, ievt );
      const double& pvec2 = pIparIp4Ievt( allmomenta, ipar, 2, ievt );
      const double& pvec3 = pIparIp4Ievt( allmomenta, ipar, 3, ievt );
#if defined __CUDACC__ && !defined MGONGPU_WFMEM_LOCAL
      dcomplex& fo0 = fov[ipar*nw6*neib + 0*neib + ieib];
      dcomplex& fo1 = fov[ipar*nw6*neib + 1*neib + ieib];
      dcomplex& fo2 = fov[ipar*nw6*neib + 2*neib + ieib];
      dcomplex& fo3 = fov[ipar*nw6*neib + 3*neib + ieib];
      dcomplex& fo4 = fov[ipar*nw6*neib + 4*neib + ieib];
      dcomplex& fo5 = fov[ipar*nw6*neib + 5*neib + ieib];
#else
      dcomplex& fo0 = fo[0];
      dcomplex& fo1 = fo[1];
      dcomplex& fo2 = fo[2];
      dcomplex& fo3 = fo[3];
      dcomplex& fo4 = fo[4];
      dcomplex& fo5 = fo[5];
#endif
      fo0 = dcomplex( pvec0 * nsf, pvec3 * nsf );
      fo1 = dcomplex( pvec1 * nsf, pvec2 * nsf );
      const int nh = nhel * nsf;
      // ASSUMPTIONS FMASS = 0 and
      // EITHER (Px and Py are not zero)
      // OR (PX = PY = 0 and E = P3 > 0)
      {
        const double sqp0p3 = sqrt( pvec0 + pvec3 ) * nsf;
        const dcomplex chi0( sqp0p3, 0 );
        const dcomplex chi1( nh * pvec1 / sqp0p3, -pvec2 / sqp0p3 );
        if( nh == 1 )
        {
          fo2 = chi0;
          fo3 = chi1;
          fo4 = dcomplex( 0, 0 );
          fo5 = dcomplex( 0, 0 );
        }
        else
        {
          fo2 = dcomplex( 0, 0 );
          fo3 = dcomplex( 0, 0 );
          fo4 = chi1;
          fo5 = chi0;
        }
      }
    }
    // ** END LOOP ON IEVT **
    return;
  }

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  void FFV1_0(const dcomplex F1[],
              const dcomplex F2[],
              const dcomplex V3[],
              const dcomplex COUP,
              dcomplex * vertex)
  {
    const dcomplex cI = dcomplex (0., 1.);
    const dcomplex TMP4 =
      (F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) +
       (F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])) +
        (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) +
         F1[5] * (F2[2] * (-V3[3] + cI * (V3[4])) + F2[3] * (V3[2] + V3[5])))));
    (*vertex) = COUP * - cI * TMP4;
  }

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  void FFV1P0_3(const dcomplex F1[],
                const dcomplex F2[],
                const dcomplex COUP,
                const double M3,
                const double W3,
                dcomplex V3[])
  {
    const dcomplex cI = dcomplex (0., 1.);
    V3[0] = +F1[0] + F2[0];
    V3[1] = +F1[1] + F2[1];
    const double P3[4] = { -V3[0].real(),
                           -V3[1].real(),
                           -V3[1].imag(),
                           -V3[0].imag() };
    const dcomplex denom =
      COUP/((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) - (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
    V3[2] = denom * (-cI) * (F1[2] * F2[4] + F1[3] * F2[5] + F1[4] * F2[2] + F1[5] * F2[3]);
    V3[3] = denom * (-cI) * (-F1[2] * F2[5] - F1[3] * F2[4] + F1[4] * F2[3] + F1[5] * F2[2]);
    V3[4] = denom * (-cI) * (-cI * (F1[2] * F2[5] + F1[5] * F2[2]) + cI * (F1[3] * F2[4] + F1[4] * F2[3]));
    V3[5] = denom * (-cI) * (-F1[2] * F2[4] - F1[5] * F2[3] + F1[3] * F2[5] + F1[4] * F2[2]);
  }

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  void FFV2_4_0(const dcomplex F1[],
                const dcomplex F2[],
                const dcomplex V3[],
                const dcomplex COUP1,
                const dcomplex COUP2,
                dcomplex * vertex)
  {
    const dcomplex cI = dcomplex (0., 1.);
    const dcomplex TMP2 =
      (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) +
       F1[5] * (F2[2] * (-V3[3] + cI * (V3[4])) + F2[3] * (V3[2] + V3[5])));
    const dcomplex TMP0 =
      (F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) +
       F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])));
    (*vertex) = (-1.) * (COUP2 * (+cI * (TMP0) + 2. * cI * (TMP2)) + cI * (TMP0 * COUP1));
  }

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  __device__
#endif
  void FFV2_4_3(const dcomplex F1[],
                const dcomplex F2[],
                const dcomplex COUP1,
                const dcomplex COUP2,
                const double M3,
                const double W3,
                dcomplex V3[])
  {
    const dcomplex cI = dcomplex (0., 1.);
    double OM3 = 0.;
    if (M3 != 0.) OM3 = 1./(M3 * M3);
    V3[0] = +F1[0] + F2[0];
    V3[1] = +F1[1] + F2[1];
    const double P3[4] = { -V3[0].real(),
                           -V3[1].real(),
                           -V3[1].imag(),
                           -V3[0].imag() };
    const dcomplex TMP1 =
      (F1[2] * (F2[4] * (P3[0] + P3[3]) + F2[5] * (P3[1] + cI * (P3[2]))) +
       F1[3] * (F2[4] * (P3[1] - cI * (P3[2])) + F2[5] * (P3[0] - P3[3])));
    const dcomplex TMP3 =
      (F1[4] * (F2[2] * (P3[0] - P3[3]) - F2[3] * (P3[1] + cI * (P3[2]))) +
       F1[5] * (F2[2] * (-P3[1] + cI * (P3[2])) + F2[3] * (P3[0] + P3[3])));
    const dcomplex denom =
      1./((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
          (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
    V3[2] = denom * (-2. * cI) *
      (COUP2 * (OM3 * - 1./2. * P3[0] * (TMP1 + 2. * (TMP3))
                + (+1./2. * (F1[2] * F2[4] + F1[3] * F2[5]) + F1[4] * F2[2] + F1[5] * F2[3]))
       + 1./2. * (COUP1 * (F1[2] * F2[4] + F1[3] * F2[5] - P3[0] * OM3 * TMP1)));
    V3[3] = denom * (-2. * cI) *
      (COUP2 * (OM3 * - 1./2. * P3[1] * (TMP1 + 2. * (TMP3))
                + (-1./2. * (F1[2] * F2[5] + F1[3] * F2[4]) + F1[4] * F2[3] + F1[5] * F2[2]))
       - 1./2. * (COUP1 * (F1[2] * F2[5] + F1[3] * F2[4] + P3[1] * OM3 * TMP1)));
    V3[4] = denom * cI *
      (COUP2 * (OM3 * P3[2] * (TMP1 + 2. * (TMP3))
                + (+cI * (F1[2] * F2[5]) - cI * (F1[3] * F2[4])
                   - 2. * cI * (F1[4] * F2[3])
                   + 2. * cI * (F1[5] * F2[2])))
       + COUP1 * (+cI * (F1[2] * F2[5]) - cI * (F1[3] * F2[4]) + P3[2] * OM3 * TMP1));
    V3[5] = denom * 2. * cI *
      (COUP2 * (OM3 * 1./2. * P3[3] * (TMP1 + 2. * (TMP3)) +
                (+1./2. * (F1[2] * F2[4]) - 1./2. * (F1[3] * F2[5]) - F1[4] * F2[2] + F1[5] * F2[3]))
       + 1./2. * (COUP1 * (F1[2] * F2[4] + P3[3] * OM3 * TMP1 - F1[3] * F2[5])));
  }


}  // end namespace $(namespace)s_sm


//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.7.3.py3, 2020-06-28
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include <algorithm>
#include <iostream>

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: e+ e- > mu+ mu- WEIGHTED<=4 @1

#ifdef __CUDACC__
namespace gProc
#else
namespace Proc
#endif
{
  using mgOnGpu::np4;
  using mgOnGpu::npar;
  const int ncomb = 16; // #helicity combinations is hardcoded for this process (eemumu: ncomb=16)

#ifdef __CUDACC__
  __device__ __constant__ int cHel[ncomb][npar];
  __device__ __constant__ double cIPC[6];  // coupling ?
  __device__ __constant__ double cIPD[2];
#else
  static int cHel[ncomb][npar];
  static double cIPC[6];  // coupling ?
  static double cIPD[2];
#endif

#ifdef __CUDACC__
  __device__ unsigned long long sigmakin_itry = 0; // first iteration over nevt events
  __device__ bool sigmakin_goodhel[ncomb] = { false };
#endif

  //--------------------------------------------------------------------------

  using mgOnGpu::nwf;
  using mgOnGpu::nw6;

#ifdef __CUDACC__
#if !defined MGONGPU_WFMEM_LOCAL

  using mgOnGpu::nbpgMAX;
  // Allocate global or shared memory for the wavefunctions of all (external and internal) particles
  // See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#allocation-persisting-kernel-launches
  __device__ dcomplex* dwf[nbpgMAX]; // device wf[#blocks][5 * 6 * #threads_in_block]

#if defined MGONGPU_WFMEM_GLOBAL
  __global__
#elif defined MGONGPU_WFMEM_SHARED
  __device__
#endif
  void sigmakin_alloc()
  {
    // Wavefunctions for this block: bwf[5 * 6 * #threads_in_block]
    dcomplex*& bwf = dwf[blockIdx.x]; 
#if defined MGONGPU_WFMEM_SHARED
    __shared__ dcomplex* sbwf;
#endif

    // Only the first thread in the block does the allocation (we need one allocation per block)
    if ( threadIdx.x == 0 )
    {
#if defined MGONGPU_WFMEM_GLOBAL
      bwf = (dcomplex*)malloc( nwf * nw6 * blockDim.x * sizeof(dcomplex) ); // dcomplex bwf[5 * 6 * #threads_in_block]
#elif defined MGONGPU_WFMEM_SHARED
      sbwf = (dcomplex*)malloc( nwf * nw6 * blockDim.x * sizeof(dcomplex) ); // dcomplex bwf[5 * 6 * #threads_in_block]
      bwf = sbwf;
#endif
      if ( bwf == NULL )
      {
        printf( "ERROR in sigmakin_alloc (block #%4d): malloc failed\n", blockIdx.x );
        assert( bwf != NULL );
      }
      //else printf( "INFO in sigmakin_alloc (block #%4d): malloc successful\n", blockIdx.x );
    }    
    __syncthreads();

    // All threads in the block should see the allocation by now
    assert( bwf != NULL );
  }
  
#if defined MGONGPU_WFMEM_GLOBAL
  __global__
#elif defined MGONGPU_WFMEM_SHARED
  __device__
#endif
  void sigmakin_free()
  {
#if defined MGONGPU_WFMEM_SHARED
    __syncthreads();
#endif
    // Only free from one thread!
    // [NB: if this free is missing, cuda-memcheck fails to detect it]
    // [NB: but if free is called twice, cuda-memcheck does detect it]
    dcomplex* bwf = dwf[blockIdx.x]; 
    if ( threadIdx.x == 0 ) free( bwf );
  }

#endif
#endif

  //--------------------------------------------------------------------------

  // Evaluate |M|^2 for each subprocess
#ifdef __CUDACC__
  __device__
#endif
  // ** NB: allmomenta can have three different layouts
  // ASA: allmomenta[npag][npar][np4][nepp] where ndim=npag*nepp
  // SOA: allmomenta[npar][np4][ndim]
  // AOS: allmomenta[ndim][npar][np4]
  void calculate_wavefunctions( int ihel,
                                const double* allmomenta, // input[(npar=4)*(np4=4)*nevt]
                                double &matrix
#ifndef __CUDACC__
                                , const int ievt
#endif
                                )
  {
#ifdef __CUDACC__
#if !defined MGONGPU_WFMEM_LOCAL
    const int iblk = blockIdx.x; // index of block in grid
    const int neib = blockDim.x; // number of events (threads) in block
    const int ieib = threadIdx.x; // index of event (thread) in block
#endif
#else
    //printf( "calculate_wavefunctions: ievt %d\n", ievt );
#endif

    dcomplex amp[2];
    dcomplex w[nwf][nw6]; // w[5][6]    
#ifdef __CUDACC__ 
#if !defined MGONGPU_WFMEM_LOCAL
    // eventually move to same AOSOA everywhere, blocks and threads
    dcomplex* bwf = dwf[iblk]; 
    MG5_sm::oxzxxxM0( allmomenta, cHel[ihel][0], -1, bwf, 0 );
    MG5_sm::imzxxxM0( allmomenta, cHel[ihel][1], +1, bwf, 1 );
    MG5_sm::ixzxxxM0( allmomenta, cHel[ihel][2], -1, bwf, 2 );
    MG5_sm::oxzxxxM0( allmomenta, cHel[ihel][3], +1, bwf, 3 );
    for ( int iwf=0; iwf<4; iwf++ ) // only copy the first 4 out of 5 
      for ( int iw6=0; iw6<nw6; iw6++ ) 
        w[iwf][iw6] = bwf[iwf*nw6*neib + iw6*neib + ieib];
#else
    MG5_sm::oxzxxxM0( allmomenta, cHel[ihel][0], -1, w[0], 0 );
    MG5_sm::imzxxxM0( allmomenta, cHel[ihel][1], +1, w[1], 1 );
    MG5_sm::ixzxxxM0( allmomenta, cHel[ihel][2], -1, w[2], 2 );
    MG5_sm::oxzxxxM0( allmomenta, cHel[ihel][3], +1, w[3], 3 );
#endif
#else
    MG5_sm::oxzxxxM0( allmomenta, cHel[ihel][0], -1, w[0], ievt, 0 );
    MG5_sm::imzxxxM0( allmomenta, cHel[ihel][1], +1, w[1], ievt, 1 );
    MG5_sm::ixzxxxM0( allmomenta, cHel[ihel][2], -1, w[2], ievt, 2 );
    MG5_sm::oxzxxxM0( allmomenta, cHel[ihel][3], +1, w[3], ievt, 3 );
#endif

    // Diagram 1
    MG5_sm::FFV1P0_3(w[1], w[0], dcomplex (cIPC[0], cIPC[1]), 0., 0., w[4]);
    MG5_sm::FFV1_0(w[2], w[3], w[4], dcomplex (cIPC[0], cIPC[1]), &amp[0]);

    // Diagram 2
    MG5_sm::FFV2_4_3(w[1], w[0], dcomplex (cIPC[2], cIPC[3]), dcomplex (cIPC[4], cIPC[5]), cIPD[0], cIPD[1], w[4]);
    MG5_sm::FFV2_4_0(w[2], w[3], w[4], dcomplex (cIPC[2], cIPC[3]), dcomplex (cIPC[4], cIPC[5]), &amp[1]);

    const int ncolor = 1;
    dcomplex ztemp;
    dcomplex jamp[ncolor];

    // The color matrix;
    static const double denom[ncolor] = {1};
    static const double cf[ncolor][ncolor] = {{1}};

    // Calculate color flows
    jamp[0] = -amp[0] - amp[1];

    // Sum and square the color flows to get the matrix element
    for(int icol = 0; icol < ncolor; icol++ )
    {
      ztemp = 0.;
      for(int jcol = 0; jcol < ncolor; jcol++ )
        ztemp = ztemp + cf[icol][jcol] * jamp[jcol];
      matrix = matrix + (ztemp * conj(jamp[icol])).real()/denom[icol];
    }

    // Store the leading color flows for choice of color
    // for(i=0;i < ncolor; i++)
    // jamp2[0][i] += real(jamp[i]*conj(jamp[i]));

  }

  //--------------------------------------------------------------------------

  CPPProcess::CPPProcess(int numiterations,
                         int gpublocks,
                         int gputhreads,
                         bool verbose,
                         bool debug)
    : m_numiterations(numiterations)
    , gpu_nblocks(gpublocks)
    , gpu_nthreads(gputhreads)
    , dim(gpu_nblocks * gpu_nthreads)
    , m_verbose(verbose)
    , m_debug(debug)
  {
    // Helicities for the process - nodim
    static const int tHel[ncomb][nexternal] =
      { {-1, -1, -1, -1}, {-1, -1, -1, +1}, {-1, -1, +1, -1}, {-1, -1, +1, +1},
        {-1, +1, -1, -1}, {-1, +1, -1, +1}, {-1, +1, +1, -1}, {-1, +1, +1, +1},
        {+1, -1, -1, -1}, {+1, -1, -1, +1}, {+1, -1, +1, -1}, {+1, -1, +1, +1},
        {+1, +1, -1, -1}, {+1, +1, -1, +1}, {+1, +1, +1, -1}, {+1, +1, +1, +1} };
#ifdef __CUDACC__
    checkCuda( cudaMemcpyToSymbol( cHel, tHel, ncomb * nexternal * sizeof(int) ) );
#else
    memcpy( cHel, tHel, ncomb * nexternal * sizeof(int) );
#endif
    // SANITY CHECK: GPU shared memory usage is based on casts of double[2] to complex
    assert( sizeof(dcomplex) == 2*sizeof(double) );
  }

  //--------------------------------------------------------------------------

  CPPProcess::~CPPProcess() {}

  //--------------------------------------------------------------------------

  const std::vector<double> &CPPProcess::getMasses() const {return mME;}

  //--------------------------------------------------------------------------
  // Initialize process.

  void CPPProcess::initProc(std::string param_card_name)
  {
    // Instantiate the model class and set parameters that stay fixed during run
    pars = Parameters_sm::getInstance();
    SLHAReader slha(param_card_name, m_verbose);
    pars->setIndependentParameters(slha);
    pars->setIndependentCouplings();
    if (m_verbose) {
      pars->printIndependentParameters();
      pars->printIndependentCouplings();
    }
    pars->setDependentParameters();
    pars->setDependentCouplings();
    // Set external particle masses for this matrix element
    mME.push_back(pars->ZERO);
    mME.push_back(pars->ZERO);
    mME.push_back(pars->ZERO);
    mME.push_back(pars->ZERO);
    static dcomplex tIPC[3] = {pars->GC_3, pars->GC_50,
                               pars->GC_59};
    static double tIPD[2] = {pars->mdl_MZ, pars->mdl_WZ};

#ifdef __CUDACC__
    checkCuda( cudaMemcpyToSymbol( cIPC, tIPC, 3 * sizeof(dcomplex ) ) );
    checkCuda( cudaMemcpyToSymbol( cIPD, tIPD, 2 * sizeof(double) ) );
#else
    memcpy( cIPC, tIPC, 3 * sizeof(dcomplex ) );
    memcpy( cIPD, tIPD, 2 * sizeof(double) );
#endif

  }

  //--------------------------------------------------------------------------
  // Evaluate |M|^2, part independent of incoming flavour.

  // ** NB: allmomenta can have three different layouts
  // ASA: allmomenta[npag][npar][np4][nepp] where ndim=npag*nepp
  // SOA: allmomenta[npar][np4][ndim]
  // AOS: allmomenta[ndim][npar][np4]
#ifdef __CUDACC__
  __global__
#endif
  void sigmaKin( const double* allmomenta, // input[(npar=4)*(np4=4)*nevt]
                 double* output            // output[nevt]
#ifdef __CUDACC__
                 // NB: nevt == ndim=gpublocks*gputhreads in CUDA
#else
                 , const int nevt          // input: #events
#endif
                 )
  {
    // Set the parameters which change event by event
    // Need to discuss this with Stefan
    // pars->setDependentParameters();
    // pars->setDependentCouplings();
    // Reset color flows
    const int maxtry = 10;
#ifndef __CUDACC__
    static unsigned long long sigmakin_itry = 0; // first iteration over nevt events
    static bool sigmakin_goodhel[ncomb] = { false };
#endif

#ifndef __CUDACC__
    MG5_sm::nevt = nevt;
    // ** START LOOP ON IEVT **
    for (int ievt = 0; ievt < nevt; ++ievt)
#endif
    {
#ifdef __CUDACC__
      const int idim = blockDim.x * blockIdx.x + threadIdx.x; // event# == threadid (previously was: tid)
      const int ievt = idim;
      //printf( "sigmakin: ievt %d\n", ievt );
#endif

      // Denominators: spins, colors and identical particles
      const int nprocesses = 1;
      const int denominators[nprocesses] = {4};

      // Reset the matrix elements
      double matrix_element[nprocesses];
      for(int iproc = 0; iproc < nprocesses; iproc++ )
      {
        matrix_element[iproc] = 0.;
      }

#ifdef __CUDACC__
#if defined MGONGPU_WFMEM_SHARED
      sigmakin_alloc();
#endif
#endif
      double melast = matrix_element[0];
      for (int ihel = 0; ihel < ncomb; ihel++ )
      {
        if ( sigmakin_itry>maxtry && !sigmakin_goodhel[ihel] ) continue;
#ifdef __CUDACC__
        calculate_wavefunctions(ihel, allmomenta, matrix_element[0]); // adds ME for ihel to matrix_element[0]
#else
        calculate_wavefunctions(ihel, allmomenta, matrix_element[0], ievt); // adds ME for ihel to matrix_element[0]
#endif
        if ( sigmakin_itry<=maxtry )
        {
          if ( !sigmakin_goodhel[ihel] && matrix_element[0]>melast ) sigmakin_goodhel[ihel] = true;
          melast = matrix_element[0];
        }
      }
#ifdef __CUDACC__
#if defined MGONGPU_WFMEM_SHARED
      sigmakin_free();
#endif
#endif

      for (int iproc = 0; iproc < nprocesses; ++iproc)
      {
        matrix_element[iproc] /= denominators[iproc];
      }

      for (int iproc = 0; iproc < nprocesses; ++iproc)
      {
        output[iproc*nprocesses + ievt] = matrix_element[iproc];
      }

#ifndef __CUDACC__
      //if ( sigmakin_itry == maxtry )
      //  for (int ihel = 0; ihel < ncomb; ihel++ )
      //    printf( "sigmakin: ihelgood %2d %d\n", ihel, sigmakin_goodhel[ihel] );
      if ( sigmakin_itry <= maxtry )
        sigmakin_itry++;
#else
      if ( sigmakin_itry <= maxtry )
        atomicAdd(&sigmakin_itry, 1);
#endif

    }
    // ** END LOOP ON IEVT **

  }

  //--------------------------------------------------------------------------

}
