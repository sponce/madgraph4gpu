// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, A. Valassi, Z. Wettersten (2020-2023) for the MG5aMC CUDACPP plugin.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
// MadGraph5_aMC@NLO v. 3.5.1_lo_vect, 2023-08-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "CPPProcess.h"

#include "mgOnGpuConfig.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <cassert>

// Test ncu metrics for CUDA thread divergence
#undef MGONGPU_TEST_DIVERGENCE
#define ALWAYS_INLINE __attribute__( ( always_inline ) )

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: e+ e- > mu+ mu- WEIGHTED<=4 @1
namespace mg5amcCpu {

  using cxtype_sv6 = std::array<cxtype_sv, 6>;

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  inline cxtype_sv6 ALWAYS_INLINE
  myopzxxx( const fptype_v momenta[], // input: momenta
            const int nhel ) {      // input: -1 or +1 (helicity of fermion)
    const fptype_sv& pvec3 = momenta[3];
    const fptype_v csqp0p3 = fpsqrt( 2. * pvec3 ) * (fptype)-1;
    std::array<cxtype_sv, CPPProcess::nw6> fo{
      cxtype_sv{pvec3 * (fptype)-1, pvec3 * (fptype)-1}, 0., 0., 0., 0., 0. 
    };
    ( ( nhel == -1 ) ? fo[2].real() : fo[5].real() ) = csqp0p3;
    return fo;
  }

  inline cxtype_sv6 ALWAYS_INLINE
  myimzxxx( const fptype_v momenta[], // input: momenta
            const int nhel ) {      // input: -1 or +1 (helicity of fermion)
    const fptype_sv& pvec3 = momenta[CPPProcess::np4 + 3];
    const fptype_v chi = -(fptype)nhel * fpsqrt( -2. * pvec3 );
    std::array<cxtype_sv, CPPProcess::nw6> fi{
      cxtype_sv{ pvec3, -pvec3 }, 0., 0., 0., 0., 0. 
    };
    ( ( nhel == 1 ) ? fi[5].real() : fi[2].real() ) = chi;
    return fi;
  }

  inline cxtype_sv6 ALWAYS_INLINE
  myixzxxx( const fptype_v momenta[], // input: momenta
            const int nhel ) {      // input: -1 or +1 (helicity of fermion)
    const fptype_sv& pvec0 = momenta[2 * CPPProcess::np4 + 0];
    const fptype_sv& pvec1 = momenta[2 * CPPProcess::np4 + 1];
    const fptype_sv& pvec2 = momenta[2 * CPPProcess::np4 + 2];
    const fptype_sv& pvec3 = momenta[2 * CPPProcess::np4 + 3];
    std::array<cxtype_sv, CPPProcess::nw6> fi{
      cxtype_sv{ -pvec0 * (fptype)-1, -pvec3 * (fptype)-1 }, // AV: BUG FIX
      cxtype_sv{ -pvec1 * (fptype)-1, -pvec2 * (fptype)-1 }, // AV: BUG FIX
      0., 0., 0., 0.
    };
    const fptype_sv sqp0p3 = fpsqrt( pvec0 + pvec3 ) * (fptype)-1;
    const cxtype_sv chi0 = { sqp0p3, fptype_sv{} };
    const cxtype_sv chi1 = { (fptype)nhel * -1 * pvec1 / sqp0p3, pvec2 / sqp0p3 };
    if( nhel == -1 ) {
      fi[2] = {};
      fi[3] = {};
      fi[4] = chi0;
      fi[5] = chi1;
    } else {
      fi[2] = chi1;
      fi[3] = chi0;
      fi[4] = {};
      fi[5] = {};
    }
    return fi;
  }

  inline cxtype_sv6 ALWAYS_INLINE
  myoxzxxx( const fptype_v momenta[], // input: momenta
            const int nhel ) {      // input: -1 or +1 (helicity of fermion)
    const fptype_sv& pvec0 = momenta[3 * CPPProcess::np4 + 0];
    const fptype_sv& pvec1 = momenta[3 * CPPProcess::np4 + 1];
    const fptype_sv& pvec2 = momenta[3 * CPPProcess::np4 + 2];
    const fptype_sv& pvec3 = momenta[3 * CPPProcess::np4 + 3];
    std::array<cxtype_sv, CPPProcess::nw6> fo{
      cxtype_sv{ pvec0, pvec3 },
      cxtype_sv{ pvec1, pvec2 },
      0., 0., 0., 0.
    };
    const fptype_sv sqp0p3 = fpsqrt( pvec0 + pvec3 );
    const cxtype_sv chi0 = { sqp0p3, fptype_sv{} };
    const cxtype_sv chi1 = { (fptype)nhel * pvec1 / sqp0p3, -pvec2 / sqp0p3 };
    if( nhel == 1 ) {
      fo[2] = chi0;
      fo[3] = chi1;
      fo[4] = {};
      fo[5] = {};
    } else {
      fo[2] = {};
      fo[3] = {};
      fo[4] = chi1;
      fo[5] = chi0;
    }
    return fo;
  }

  inline cxtype_sv6 ALWAYS_INLINE
  myFFV1P0_3( const cxtype_sv F1[],
              const cxtype_sv F2[],
              const cxtype& allCOUP) {
    const cxtype_sv COUP = cxtype_sv{ fptype_v{allCOUP.real()}, fptype_v{allCOUP.imag()} };
    const cxtype cI = { 0., 1. };
    auto V30 = +F1[0] + F2[0];
    auto V31 = +F1[1] + F2[1];
    const fptype_sv P3[4] = { -cxreal( V30 ), -cxreal( V31 ), -cximag( V31 ), -cximag( V30 ) };
    const cxtype_sv denom = COUP / ( ( P3[0] * P3[0] ) - ( P3[1] * P3[1] ) - ( P3[2] * P3[2] ) - ( P3[3] * P3[3] ) );
    return {
      V30, V31, 
      denom * ( -cI ) * ( F1[2] * F2[4] + F1[3] * F2[5] + F1[4] * F2[2] + F1[5] * F2[3] ),
      denom * ( -cI ) * ( -F1[2] * F2[5] - F1[3] * F2[4] + F1[4] * F2[3] + F1[5] * F2[2] ),
      denom * ( -cI ) * ( -cI * ( F1[2] * F2[5] + F1[5] * F2[2] ) + cI * ( F1[3] * F2[4] + F1[4] * F2[3] ) ),
      denom * ( -cI ) * ( -F1[2] * F2[4] - F1[5] * F2[3] + F1[3] * F2[5] + F1[4] * F2[2] )
      };
  }

  inline cxtype_sv ALWAYS_INLINE
  myFFV1_0( const cxtype_sv6& F1,
            const cxtype_sv6& F2,
            const cxtype_sv6& V3,
            const cxtype& allCOUP ) {
    const cxtype_sv COUP = cxtype_sv{ fptype_v{allCOUP.real()}, fptype_v{allCOUP.imag()} };
    const cxtype cI = { 0., 1. };
    const cxtype_sv TMP0 = ( F1[2] * ( F2[4] * ( V3[2] + V3[5] ) + F2[5] * ( V3[3] + cI * V3[4] ) ) + ( F1[3] * ( F2[4] * ( V3[3] - cI * V3[4] ) + F2[5] * ( V3[2] - V3[5] ) ) + ( F1[4] * ( F2[2] * ( V3[2] - V3[5] ) - F2[3] * ( V3[3] + cI * V3[4] ) ) + F1[5] * ( F2[2] * ( -V3[3] + cI * V3[4] ) + F2[3] * ( V3[2] + V3[5] ) ) ) ) );
    return COUP * -cI * TMP0;
  }

  // Compute the output amplitude 'vertex' from the input wavefunctions F1[6], F2[6], V3[6]
  inline cxtype_sv ALWAYS_INLINE
  myFFV2_4_0( const cxtype_sv6& F1,
              const cxtype_sv6& F2,
              const cxtype_sv6& V3,
              const cxtype& allCOUP1,
              const cxtype& allCOUP2 ) {
    const cxtype_sv COUP1 = cxtype_sv{ fptype_v{allCOUP1.real()}, fptype_v{allCOUP1.imag()} };
    const cxtype_sv COUP2 = cxtype_sv{ fptype_v{allCOUP2.real()}, fptype_v{allCOUP2.imag()} };
    const cxtype cI = { 0., 1. };
    constexpr fptype one( 1. );
    constexpr fptype two( 2. );
    const cxtype_sv TMP1 = ( F1[2] * ( F2[4] * ( V3[2] + V3[5] ) + F2[5] * ( V3[3] + cI * V3[4] ) ) + F1[3] * ( F2[4] * ( V3[3] - cI * V3[4] ) + F2[5] * ( V3[2] - V3[5] ) ) );
    const cxtype_sv TMP3 = ( F1[4] * ( F2[2] * ( V3[2] - V3[5] ) - F2[3] * ( V3[3] + cI * V3[4] ) ) + F1[5] * ( F2[2] * ( -V3[3] + cI * V3[4] ) + F2[3] * ( V3[2] + V3[5] ) ) );
    return ( -one ) * ( COUP2 * ( +cI * TMP1 + ( two * cI ) * TMP3 ) + cI * ( TMP1 * COUP1 ) );
  }
  
  // Compute the output wavefunction 'V3[6]' from the input wavefunctions F1[6], F2[6]
  inline cxtype_sv6 ALWAYS_INLINE
  myFFV2_4_3( const cxtype_sv F1[],
              const cxtype_sv F2[],
              const cxtype& allCOUP1,
              const cxtype& allCOUP2,
              const fptype M3,
              const fptype W3 ) {
    const cxtype_sv COUP1 = cxtype_sv{ fptype_v{allCOUP1.real()}, fptype_v{allCOUP1.imag()} };
    const cxtype_sv COUP2 = cxtype_sv{ fptype_v{allCOUP2.real()}, fptype_v{allCOUP2.imag()} };
    const cxtype cI = { 0., 1. };
    const fptype OM3 = ( M3 != 0. ? 1. / ( M3 * M3 ) : 0. );
    auto V30 = +F1[0] + F2[0];
    auto V31 = +F1[1] + F2[1];
    const fptype_sv P3[4] = { -cxreal( V30 ), -cxreal( V31 ), -cximag( V31 ), -cximag( V30 ) };
    constexpr fptype one( 1. );
    constexpr fptype two( 2. );
    constexpr fptype half( 1. / 2. );
    const cxtype_sv TMP2 = ( F1[2] * ( F2[4] * ( P3[0] + P3[3] ) + F2[5] * ( P3[1] + cI * P3[2] ) ) + F1[3] * ( F2[4] * ( P3[1] - cI * P3[2] ) + F2[5] * ( P3[0] - P3[3] ) ) );
    const cxtype_sv TMP4 = ( F1[4] * ( F2[2] * ( P3[0] - P3[3] ) - F2[3] * ( P3[1] + cI * P3[2] ) ) + F1[5] * ( F2[2] * ( -P3[1] + cI * P3[2] ) + F2[3] * ( P3[0] + P3[3] ) ) );
    const cxtype_sv denom = one / ( ( P3[0] * P3[0] ) - ( P3[1] * P3[1] ) - ( P3[2] * P3[2] ) - ( P3[3] * P3[3] ) - M3 * ( M3 - cI * W3 ) );
    return {
      V30, V31, 
      denom * ( -two * cI ) * ( COUP2 * ( OM3 * -half * P3[0] * ( TMP2 + two * TMP4 ) + ( +half * ( F1[2] * F2[4] + F1[3] * F2[5] ) + F1[4] * F2[2] + F1[5] * F2[3] ) ) + half * ( COUP1 * ( F1[2] * F2[4] + F1[3] * F2[5] - P3[0] * OM3 * TMP2 ) ) ),
      denom * ( -two * cI ) * ( COUP2 * ( OM3 * -half * P3[1] * ( TMP2 + two * TMP4 ) + ( -half * ( F1[2] * F2[5] + F1[3] * F2[4] ) + F1[4] * F2[3] + F1[5] * F2[2] ) ) - half * ( COUP1 * ( F1[2] * F2[5] + F1[3] * F2[4] + P3[1] * OM3 * TMP2 ) ) ),
      denom * cI * ( COUP2 * ( OM3 * P3[2] * ( TMP2 + two * TMP4 ) + ( +cI * ( F1[2] * F2[5] ) - cI * ( F1[3] * F2[4] ) + ( -two * cI ) * ( F1[4] * F2[3] ) + ( two * cI ) * ( F1[5] * F2[2] ) ) ) + COUP1 * ( +cI * ( F1[2] * F2[5] ) - cI * ( F1[3] * F2[4] ) + P3[2] * OM3 * TMP2 ) ),
      denom * ( two * cI ) * ( COUP2 * ( OM3 * half * P3[3] * ( TMP2 + two * TMP4 ) + ( +half * ( F1[2] * F2[4] ) - half * ( F1[3] * F2[5] ) - F1[4] * F2[2] + F1[5] * F2[3] ) ) + half * ( COUP1 * ( F1[2] * F2[4] + P3[3] * OM3 * TMP2 - F1[3] * F2[5] ) ) )
    };
  }

}

namespace mg5amcCpu {
  using Parameters_sm_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // Physics parameters (masses, coupling, etc...)
  static fptype cIPD[2];
  static cxtype cIPC[3];

  // Helicity combinations (and filtering of "good" helicity combinations)
  static short cHel[CPPProcess::ncomb][CPPProcess::npar];
  static int cNGoodHel;
  static int cGoodHel[CPPProcess::ncomb];

  // Evaluate |M|^2 for each subprocess
  // NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event(s)
  // (similarly, it also ADDS the numerator and denominator for a given ihel to their running sums over helicities)
  // In C++, this function computes the ME for a single event "page" or SIMD vector
  inline void /* clang-format off */
  calculate_wavefunctions( int ihel,
                           const fptype_v* allmomenta,      // input: momenta[nevt*npar*4]
                           const fptype* allcouplings,    // input: couplings[nevt*ndcoup*2]
                           fptype_v* allMEs,                // output: allMEs[nevt], |M|^2 running_sum_over_helicities
                           const int ievt0                // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)
                           ) {
    using namespace mg5amcCpu;

    const fptype_v* momenta = &allmomenta[ievt0 * CPPProcess::np4];

    auto w_sv0 = myopzxxx( momenta, cHel[ihel][0] ); // NB: opzxxx only uses pz
    auto w_sv1 = myimzxxx( momenta, cHel[ihel][1] ); // NB: imzxxx only uses pz
    auto w_sv2 = myixzxxx( momenta, cHel[ihel][2] );
    auto w_sv3 = myoxzxxx( momenta, cHel[ihel][3] );
    auto w_sv4 = myFFV1P0_3( w_sv1.data(), w_sv0.data(), cIPC[0] );
    cxtype_sv jamp_sv = -myFFV1_0( w_sv2, w_sv3, w_sv4, cIPC[0] );

    w_sv4 = myFFV2_4_3( w_sv1.data(), w_sv0.data(), cIPC[1], cIPC[2], cIPD[0], cIPD[1] );
    jamp_sv -= myFFV2_4_0( w_sv2, w_sv3, w_sv4, cIPC[1], cIPC[2] );

    allMEs[ievt0] += cxabs2( jamp_sv );
  }

  CPPProcess::CPPProcess( bool verbose )
    : m_verbose( verbose ), m_pars( 0 ), m_masses() {
    // Helicities for the process [NB do keep 'static' for this constexpr array, see issue #283]
    // *** NB There is no automatic check yet that these are in the same order as Fortran! #569 ***
    static constexpr short tHel[ncomb][npar] = {
      { -1, 1, 1, -1 },
      { -1, 1, 1, 1 },
      { -1, 1, -1, -1 },
      { -1, 1, -1, 1 },
      { -1, -1, 1, -1 },
      { -1, -1, 1, 1 },
      { -1, -1, -1, -1 },
      { -1, -1, -1, 1 },
      { 1, 1, 1, -1 },
      { 1, 1, 1, 1 },
      { 1, 1, -1, -1 },
      { 1, 1, -1, 1 },
      { 1, -1, 1, -1 },
      { 1, -1, 1, 1 },
      { 1, -1, -1, -1 },
      { 1, -1, -1, 1 } };
    memcpy( cHel, tHel, ncomb * npar * sizeof( short ) );
  }

  CPPProcess::~CPPProcess() {}

  // Initialize process (with parameters read from user cards)
  void
  CPPProcess::initProc( const std::string& param_card_name )
  {
    // Instantiate the model class and set parameters that stay fixed during run
    m_pars = Parameters_sm::getInstance();
    SLHAReader slha( param_card_name, m_verbose );
    m_pars->setIndependentParameters( slha );
    m_pars->setIndependentCouplings();
    if( m_verbose )
    {
      m_pars->printIndependentParameters();
      m_pars->printIndependentCouplings();
    }
    // Set external particle masses for this matrix element
    m_masses.push_back( m_pars->ZERO );
    m_masses.push_back( m_pars->ZERO );
    m_masses.push_back( m_pars->ZERO );
    m_masses.push_back( m_pars->ZERO );
    // Read physics parameters like masses and couplings from user configuration files (static: initialize once)
    // Then copy them to CUDA constant memory (issue #39) or its C++ emulation in file-scope static memory
    cIPD[0] = (fptype)m_pars->mdl_MZ;
    cIPD[1] = (fptype)m_pars->mdl_WZ;
    cIPC[0] = m_pars->GC_3;
    cIPC[1] = m_pars->GC_50;
    cIPC[2] = m_pars->GC_59;
  }

  __global__ void /* clang-format off */
  computeDependentCouplings( const fptype* allgs, // input: Gs[nevt]
                             fptype* allcouplings // output: couplings[nevt*ndcoup*2]
                             , const int nevt     // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
                             ) { /* clang-format on */
    using namespace mg5amcCpu;
    for( int ievt0 = 0; ievt0 < nevt; ievt0 += neppV ) {
      const fptype* gs = &( allgs[ievt0] );
      fptype* couplings = allcouplings;
      const fptype_sv& gs_sv = *reinterpret_cast<const fptype_sv*>( gs );
      Parameters_sm_dependentCouplings::computeDependentCouplings_fromG( gs_sv );
    }
  }

  void
  sigmaKin_getGoodHel( const fptype_v* allmomenta,   // input: momenta[nevt*npar*4]
                       fptype_v* allMEs,             // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       bool* isGoodHel,            // output: isGoodHel[ncomb] - host array (C++ implementation)
                       const int nevt )            // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    // Allocate arrays at build time to contain at least 16 events (or at least neppV events if neppV>16, e.g. in future VPUs)
    constexpr int maxtry0 = std::max( 16, neppV ); // 16, but at least neppV (otherwise the npagV loop does not even start)
    // Loop over only nevt events if nevt is < 16 (note that nevt is always >= neppV)
    assert( nevt >= neppV );
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)

    // HELICITY LOOP: CALCULATE WAVEFUNCTIONS
    const int npagV = maxtry / neppV;
    const int npagV2 = npagV;            // loop on one SIMD page (neppV events) at a time
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 ) {
      for( int ihel = 0; ihel < CPPProcess::ncomb; ihel++ ) {
        // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
        allMEs[ipagV2] = fptype_v{};
        calculate_wavefunctions( ihel, allmomenta, nullptr, allMEs, ipagV2 );
        fptype_v& me = allMEs[ipagV2];
        for(int i=0; i<4; i++) {
          if (me[i] != 0) {
            isGoodHel[ihel] = true;
          }
        }
      }
    }
  }

  // output: nGoodHel (the number of good helicity combinations out of ncomb)
  // input: isGoodHel[ncomb] - host array (CUDA and C++)
  int sigmaKin_setGoodHel( const bool* isGoodHel ) {
    int nGoodHel = 0;
    int goodHel[CPPProcess::ncomb] = { 0 };
    for( int ihel = 0; ihel < CPPProcess::ncomb; ihel++ ) {
      if( isGoodHel[ihel] ) {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < CPPProcess::ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  // Evaluate |M|^2, part independent of incoming flavour
  __global__ void /* clang-format off */
  sigmaKin( const fptype_v* allmomenta,      // input: momenta[nevt*npar*4]
            const fptype* allrndhel,       // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,       // input: random numbers[nevt] for color selection
            fptype_v* allMEs,                // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                // output: helicity selection[nevt]
            const int nevt               // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
            ) { /* clang-format on */
    // === PART 0 - INITIALISATION (before calculate_wavefunctions) ===
    const int nevt1 = (nevt+neppV-1)/neppV;
    memset(allMEs, 0, nevt1*neppV*sizeof(fptype));

    // === PART 1 - HELICITY LOOP: CALCULATE WAVEFUNCTIONS ===
    // (in both CUDA and C++, using precomputed good helicities)

    // *** START OF PART 1b - C++ (loop on event pages)
    for( int ievt0 = 0; ievt0 < nevt/neppV; ievt0++ ) {
      // Running sum of partial amplitudes squared for event by event color selection (#402)
      fptype toto{};
      fptype_sv MEs_ighel[CPPProcess::ncomb] = { 0 };    // sum of MEs for all good helicities up to ighel (for the first - and/or only - neppV page)
      for( int ighel = 0; ighel < cNGoodHel; ighel++ ) {
        calculate_wavefunctions( cGoodHel[ighel], allmomenta, &toto, allMEs, ievt0 );
        MEs_ighel[ighel] = allMEs[ievt0];
      }
      // Event-by-event random choice of helicity #403
      for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
        const int ievt = ievt0*neppV + ieppV;
        for( int ighel = 0; ighel < cNGoodHel; ighel++ ) {
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel][ieppV] / MEs_ighel[cNGoodHel - 1][ieppV] );
          if( okhel ) {
            const int ihelF = cGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt] = ihelF;
            break;
          }
        }
        if (allrndcol[ievt] < 1) break;
      }
    }
    // *** END OF PART 1b - C++ (loop on event pages)

    // === PART 2 - FINALISATION (after calculate_wavefunctions) ===
    // Get the final |M|^2 as an average over helicities/colors of the running sum of |M|^2 over helicities for the given event
    // [NB 'sum over final spins, average over initial spins', eg see
    // https://www.uzh.ch/cmsssl/physik/dam/jcr:2e24b7b1-f4d7-4160-817e-47b13dbf1d7c/Handout_4_2016-UZH.pdf]
    for( int ievt0 = 0; ievt0 < nevt/neppV; ievt0++ ) {
      allMEs[ievt0] /= 4;
    }
  }

} // end namespace
