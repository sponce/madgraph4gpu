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

#include "CudaRuntime.h"
#include "HelAmps_sm.h"
#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessGs.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessWavefunctions.h"

#include "coloramps.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>

// Test ncu metrics for CUDA thread divergence
#undef MGONGPU_TEST_DIVERGENCE

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: e+ e- > mu+ mu- WEIGHTED<=4 @1

namespace mg5amcCpu {
  using Parameters_sm_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_sm_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // Physics parameters (masses, coupling, etc...)
  static fptype cIPD[2];
  static fptype cIPC[6];

  // Helicity combinations (and filtering of "good" helicity combinations)
  static short cHel[CPPProcess::ncomb][CPPProcess::npar];
  static int cNGoodHel;
  static int cGoodHel[CPPProcess::ncomb];

  // Evaluate |M|^2 for each subprocess
  // NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event(s)
  // (similarly, it also ADDS the numerator and denominator for a given ihel to their running sums over helicities)
  // In C++, this function computes the ME for a single event "page" or SIMD vector
  __device__ INLINE void /* clang-format off */
  calculate_wavefunctions( int ihel,
                           const fptype* allmomenta,      // input: momenta[nevt*npar*4]
                           const fptype* allcouplings,    // input: couplings[nevt*ndcoup*2]
                           fptype* allMEs,                // output: allMEs[nevt], |M|^2 running_sum_over_helicities
                           const unsigned int channelId,  // input: multichannel channel id (1 to #diagrams); 0 to disable channel enhancement
                           fptype* allNumerators,         // output: multichannel numerators[nevt], running_sum_over_helicities
                           fptype* allDenominators,       // output: multichannel denominators[nevt], running_sum_over_helicities
                           fptype_sv* jamp2_sv            // output: jamp2[1][1][neppV] for color choice (nullptr if disabled)
                           , const int ievt0             // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)
                           ) {
    assert(ievt0 % neppV == 0);
    using namespace mg5amcCpu;

    // The variable nwf (which is specific to each P1 subdirectory, #644) is only used here
    // It is hardcoded here because various attempts to hardcode it in CPPProcess.h
    // at generation time gave the wrong result...
    static const int nwf = 5;

    // Local TEMPORARY variables for a subset of Feynman diagrams in the given C++ event page (ipagV)
    // these variables are reused several times (and re-initialised each time) within the event page
    // in other words, amplitudes and wavefunctions still have TRIVIAL ACCESS:
    // there is currently no need to have large memory structures for wavefunctions/amplitudes
    // in all events (no kernel splitting yet)!

    // particle wavefunctions within Feynman diagrams
    // (nw6 is often 6, the dimension of spin 1/2 or spin 1 particles)
    cxtype_sv w_sv[nwf][CPPProcess::nw6];
    cxtype_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram

    // Proof of concept for using fptype* in the interface
    fptype* w_fp[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) w_fp[iwf] = reinterpret_cast<fptype*>( w_sv[iwf] );
    fptype* amp_fp;
    amp_fp = reinterpret_cast<fptype*>( amp_sv );

    // Local variables for the given C++ event page (ipagV)
    // [jamp: sum (for one event or event page) of the invariant amplitudes for all Feynman diagrams in a given color combination]
    cxtype_sv jamp_sv{}; // all zeros

    // Calculate wavefunctions and amplitudes for all diagrams in all processes
    // for one SIMD event pages
    constexpr size_t nxcoup = ndcoup + nicoup; // both dependent and independent couplings
    const fptype* allCOUPs[nxcoup];
    for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ ) {
      // dependent couplings, vary event-by-event
      allCOUPs[idcoup] = MemoryAccessCouplingsBase::idcoupAccessBuffer( allcouplings, idcoup );
    }
    for( size_t iicoup = 0; iicoup < nicoup; iicoup++ ) {
      // independent couplings, fixed for all events
      allCOUPs[ndcoup + iicoup] = HostAccessCouplingsFixed::iicoupAccessBufferConst( cIPC, iicoup );
    }
    // C++ kernels take input/output buffers with momenta/MEs for one specific event
    // (the first in the current event page)
    const fptype* momenta = &( allmomenta[ievt0 * CPPProcess::npar * CPPProcess::np4] );
    const fptype* COUPs[nxcoup];
    for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ ) {
       // dependent couplings, vary event-by-event
      COUPs[idcoup] = KernelAccessCouplings<false>::ieventAccessRecordConst( allCOUPs[idcoup], ievt0 );
    }
    for( size_t iicoup = 0; iicoup < nicoup; iicoup++ ) {
      COUPs[ndcoup + iicoup] = allCOUPs[ndcoup + iicoup]; // independent couplings, fixed for all events
    }
    fptype* MEs = &( allMEs[ievt0] );

    // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
    jamp_sv = cxzero_sv();

    // Numerators and denominators for the current SIMD event page (C++)
    fptype_sv& numerators_sv = *reinterpret_cast<fptype_v*>( &( allNumerators[ievt0] ) );
    fptype_sv& denominators_sv = *reinterpret_cast<fptype_v*>( &( allDenominators[ievt0] ) );

    // *** DIAGRAM 1 OF 2 ***

    // Wavefunction(s) for diagram number 1
    opzxxx<KernelAccessMomenta<false>, HostAccessWavefunctions>( momenta, cHel[ihel][0], -1, w_fp[0], 0 ); // NB: opzxxx only uses pz
    imzxxx<KernelAccessMomenta<false>, HostAccessWavefunctions>( momenta, cHel[ihel][1], +1, w_fp[1], 1 ); // NB: imzxxx only uses pz
    ixzxxx<KernelAccessMomenta<false>, HostAccessWavefunctions>( momenta, cHel[ihel][2], -1, w_fp[2], 2 );
    oxzxxx<KernelAccessMomenta<false>, HostAccessWavefunctions>( momenta, cHel[ihel][3], +1, w_fp[3], 3 );
    FFV1P0_3<HostAccessWavefunctions, HostAccessCouplingsFixed>( w_fp[1], w_fp[0], COUPs[0], 0., 0., w_fp[4] );

    // Amplitude(s) for diagram number 1
    FFV1_0<HostAccessWavefunctions, HostAccessAmplitudes, HostAccessCouplingsFixed>( w_fp[2], w_fp[3], w_fp[4], COUPs[0], &amp_fp[0] );
    if( channelId == 1 ) numerators_sv += cxabs2( amp_sv[0] );
    if( channelId != 0 ) denominators_sv += cxabs2( amp_sv[0] );
    jamp_sv -= amp_sv[0];

    // *** DIAGRAM 2 OF 2 ***

    // Wavefunction(s) for diagram number 2
    FFV2_4_3<HostAccessWavefunctions, HostAccessCouplingsFixed>( w_fp[1], w_fp[0], COUPs[1], COUPs[2], cIPD[0], cIPD[1], w_fp[4] );
    // Amplitude(s) for diagram number 2
    FFV2_4_0<HostAccessWavefunctions, HostAccessAmplitudes, HostAccessCouplingsFixed>( w_fp[2], w_fp[3], w_fp[4], COUPs[1], COUPs[2], &amp_fp[0] );
    if( channelId == 2 ) numerators_sv += cxabs2( amp_sv[0] );
    if( channelId != 0 ) denominators_sv += cxabs2( amp_sv[0] );
    jamp_sv -= amp_sv[0];

    // *** COLOR CHOICE BELOW ***
    // Store the leading color flows for choice of color
    if( jamp2_sv ) // disable color choice if nullptr
      *jamp2_sv += cxabs2( jamp_sv );

    // *** COLOR MATRIX BELOW ***

    // Sum and square the color flows to get the matrix element
    // (compute |M|^2 by squaring |M|, taking into account colours)
    fptype_sv deltaMEs = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes

    // Use the property that M is a real matrix (see #475):
    // we can rewrite the quadratic form (A-iB)(M)(A+iB) as AMA - iBMA + iBMA + BMB = AMA + BMB
    // In addition, on C++ use the property that M is symmetric (see #475),
    // and also use constexpr to compute "2*" and "/1" once and for all at compile time:
    // we gain (not a factor 2...) in speed here as we only loop over the up diagonal part of the matrix.
    // Strangely, CUDA is slower instead, so keep the old implementation for the moment.
    // Diagonal terms
    fptype2_sv jampRi_sv = (fptype2_sv)( cxreal( jamp_sv ) );
    fptype2_sv jampIi_sv = (fptype2_sv)( cximag( jamp_sv ) );
    fptype2_sv deltaMEs2 = ( jampRi_sv * jampRi_sv + jampIi_sv * jampIi_sv );
    deltaMEs += deltaMEs2;
      
    // *** STORE THE RESULTS ***

    // NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event(s)
    fptype_sv& MEs_sv = *reinterpret_cast<fptype_v*>( MEs );
    MEs_sv += deltaMEs; // fix #435
    return;
  }

  CPPProcess::CPPProcess( bool verbose,
                          bool debug )
    : m_verbose( verbose ), m_debug( debug ), m_pars( 0 ), m_masses() {
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
    const fptype tIPD[2] = { (fptype)m_pars->mdl_MZ, (fptype)m_pars->mdl_WZ };
    const cxtype tIPC[3] = { cxmake( m_pars->GC_3 ), cxmake( m_pars->GC_50 ), cxmake( m_pars->GC_59 ) };
    memcpy( cIPD, tIPD, 2 * sizeof( fptype ) );
    memcpy( cIPC, tIPC, 3 * sizeof( cxtype ) );
  }

  __global__ void /* clang-format off */
  computeDependentCouplings( const fptype* allgs, // input: Gs[nevt]
                             fptype* allcouplings // output: couplings[nevt*ndcoup*2]
                             , const int nevt     // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
                             ) { /* clang-format on */
    using namespace mg5amcCpu;
    for( int ievt0 = 0; ievt0 < nevt; ievt0 += neppV ) {
      const fptype* gs = &( allgs[ievt0] );
      fptype* couplings = MemoryAccessCouplings::ieventAccessRecord( allcouplings, ievt0 );
      G2COUP<HostAccessGs, KernelAccessCouplings<false>>( gs, couplings );
    }
  }

  void
  sigmaKin_getGoodHel( const fptype* allmomenta,   // input: momenta[nevt*npar*4]
                       const fptype* allcouplings, // input: couplings[nevt*ndcoup*2]
                       fptype* allMEs,             // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype* allNumerators,      // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype* allDenominators,    // output: multichannel denominators[nevt], running_sum_over_helicities
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
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
      for( int ihel = 0; ihel < CPPProcess::ncomb; ihel++ ) {
        // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
        for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
          const int ievt = ievt00 + ieppV;
          allMEs[ievt] = 0;
        }
        constexpr fptype_sv* jamp2_sv = nullptr; // no need for color selection during helicity filtering
        constexpr unsigned int channelId = 0; // disable single-diagram channel enhancement
        calculate_wavefunctions( ihel, allmomenta, allcouplings, allMEs, channelId, allNumerators, allDenominators, jamp2_sv, ievt00 );
        for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
          const int ievt = ievt00 + ieppV;
          if( allMEs[ievt] != 0 ) {// NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
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
  sigmaKin( const fptype* allmomenta,      // input: momenta[nevt*npar*4]
            const fptype* allcouplings,    // input: couplings[nevt*ndcoup*2]
            const fptype* allrndhel,       // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,       // input: random numbers[nevt] for color selection
            fptype* allMEs,                // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            const unsigned int channelId,  // input: multichannel channel id (1 to #diagrams); 0 to disable channel enhancement
            fptype* allNumerators,         // output: multichannel numerators[nevt], running_sum_over_helicities
            fptype* allDenominators,       // output: multichannel denominators[nevt], running_sum_over_helicities
            int* allselhel,                // output: helicity selection[nevt]
            int* allselcol                 // output: helicity selection[nevt]
            , const int nevt               // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
            ) { /* clang-format on */
    // === PART 0 - INITIALISATION (before calculate_wavefunctions) ===
    const int nevt1 = (nevt+neppV-1)/neppV;
    memset(allMEs, 0, nevt1*neppV*sizeof(fptype));
    memset(allNumerators, 0, nevt1*neppV*sizeof(fptype));
    memset(allDenominators, 0, nevt1*neppV*sizeof(fptype));

    // === PART 1 - HELICITY LOOP: CALCULATE WAVEFUNCTIONS ===
    // (in both CUDA and C++, using precomputed good helicities)

    // *** START OF PART 1b - C++ (loop on event pages)
    const int npagV = nevt / neppV;
#ifdef _OPENMP
#pragma omp parallel for default( none ) shared( allcouplings, allMEs, allmomenta, allrndcol, allrndhel, allselcol, allselhel, cGoodHel, cNGoodHel, npagV, allDenominators, allNumerators, channelId, mgOnGpu::icolamp )
#endif // _OPENMP
    for( int ipagV = 0; ipagV < npagV; ++ipagV ) {
      // Running sum of partial amplitudes squared for event by event color selection (#402)
      fptype_sv jamp2_sv = { 0 };
      fptype_sv MEs_ighel[CPPProcess::ncomb] = { 0 };    // sum of MEs for all good helicities up to ighel (for the first - and/or only - neppV page)
      const int ievt0 = ipagV * neppV; // loop on one SIMD page (neppV events) at a time
      for( int ighel = 0; ighel < cNGoodHel; ighel++ ) {
        const int ihel = cGoodHel[ighel];
        calculate_wavefunctions( ihel, allmomenta, allcouplings, allMEs, channelId, allNumerators, allDenominators, &jamp2_sv, ievt0 );
        MEs_ighel[ighel] = *reinterpret_cast<fptype_v*>( &( allMEs[ievt0] ) );
      }
      // Event-by-event random choice of helicity #403
      for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
        const int ievt = ievt0 + ieppV;
        for( int ighel = 0; ighel < cNGoodHel; ighel++ ) {
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel][ieppV] / MEs_ighel[cNGoodHel - 1][ieppV] );
          if( okhel ) {
            const int ihelF = cGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt] = ihelF;
            break;
          }
        }
      }
      const int channelIdC = channelId - 1; // coloramps.h uses the C array indexing starting at 0
      // Event-by-event random choice of color #402
      fptype_sv targetamp{ 0 };
      if( mgOnGpu::icolamp[channelIdC][0] ) targetamp += jamp2_sv;
      for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
        const int ievt = ievt0 + ieppV;
        const bool okcol = allrndcol[ievt] < ( targetamp[ieppV] / targetamp[ieppV] );
        if( okcol ) {
          allselcol[ievt] = 1; // NB Fortran [1,1], cudacpp [0,0]
          break;
        }
      }
    }
    // *** END OF PART 1b - C++ (loop on event pages)

    // === PART 2 - FINALISATION (after calculate_wavefunctions) ===
    // Get the final |M|^2 as an average over helicities/colors of the running sum of |M|^2 over helicities for the given event
    // [NB 'sum over final spins, average over initial spins', eg see
    // https://www.uzh.ch/cmsssl/physik/dam/jcr:2e24b7b1-f4d7-4160-817e-47b13dbf1d7c/Handout_4_2016-UZH.pdf]
    for( int ievt0 = 0; ievt0 < nevt; ievt0 += neppV ) {
      fptype_sv& MEs_sv = *reinterpret_cast<fptype_v*>( &( allMEs[ievt0] ) );
      MEs_sv /= 4;
      if( channelId > 0 ) {
        fptype_sv& numerators_sv = *reinterpret_cast<fptype_v*>( &( allNumerators[ievt0] ) );
        fptype_sv& denominators_sv = *reinterpret_cast<fptype_v*>( &( allDenominators[ievt0] ) );
        MEs_sv *= numerators_sv / denominators_sv;
      }
    }
  }

} // end namespace
