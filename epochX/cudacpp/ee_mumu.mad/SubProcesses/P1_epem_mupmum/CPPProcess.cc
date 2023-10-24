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

#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
#include "MemoryAccessDenominators.h"
#include "MemoryAccessNumerators.h"
#include "coloramps.h"
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>

// Test ncu metrics for CUDA thread divergence
#undef MGONGPU_TEST_DIVERGENCE
//#define MGONGPU_TEST_DIVERGENCE 1

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: e+ e- > mu+ mu- WEIGHTED<=4 @1

namespace mg5amcCpu
{
  constexpr int nw6 = CPPProcess::nw6;     // dimensions of each wavefunction (HELAS KEK 91-11): e.g. 6 for e+ e- -> mu+ mu- (fermions and vectors)
  constexpr int npar = CPPProcess::npar;   // #particles in total (external = initial + final): e.g. 4 for e+ e- -> mu+ mu-
  constexpr int ncomb = CPPProcess::ncomb; // #helicity combinations: e.g. 16 for e+ e- -> mu+ mu- (2**4 = fermion spin up/down ** npar)

  // [NB: I am currently unable to get the right value of nwf in CPPProcess.h - will hardcode it in CPPProcess.cc instead (#644)]
  //using CPPProcess::nwf; // #wavefunctions = #external (npar) + #internal: e.g. 5 for e+ e- -> mu+ mu- (1 internal is gamma or Z)

  using Parameters_sm_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_sm_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // The number of colors
  constexpr int ncolor = 1;

  // The number of SIMD vectors of events processed by calculate_wavefunction
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  constexpr int nParity = 2;
#else
  constexpr int nParity = 1;
#endif

  // Physics parameters (masses, coupling, etc...)
  // For CUDA performance, hardcoded constexpr's would be better: fewer registers and a tiny throughput increase
  // However, physics parameters are user-defined through card files: use CUDA constant memory instead (issue #39)
  // [NB if hardcoded parameters are used, it's better to define them here to avoid silent shadowing (issue #263)]
#ifdef MGONGPU_HARDCODE_PARAM
  __device__ const fptype cIPD[2] = { (fptype)Parameters_sm::mdl_MZ, (fptype)Parameters_sm::mdl_WZ };
  __device__ const fptype cIPC[6] = { (fptype)Parameters_sm::GC_3.real(), (fptype)Parameters_sm::GC_3.imag(), (fptype)Parameters_sm::GC_50.real(), (fptype)Parameters_sm::GC_50.imag(), (fptype)Parameters_sm::GC_59.real(), (fptype)Parameters_sm::GC_59.imag() };
#else
  static fptype cIPD[2];
  static fptype cIPC[6];
#endif

  // Helicity combinations (and filtering of "good" helicity combinations)
  static short cHel[ncomb][npar];
  static int cNGoodHel;
  static int cGoodHel[ncomb];

  //--------------------------------------------------------------------------

  // Evaluate |M|^2 for each subprocess
  // NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event(s)
  // (similarly, it also ADDS the numerator and denominator for a given ihel to their running sums over helicities)
  // In CUDA, this device function computes the ME for a single event
  // In C++, this function computes the ME for a single event "page" or SIMD vector (or for two in "mixed" precision mode, nParity=2)
  __device__ INLINE void /* clang-format off */
  calculate_wavefunctions( int ihel,
                           const fptype* allmomenta,      // input: momenta[nevt*npar*4]
                           const fptype* allcouplings,    // input: couplings[nevt*ndcoup*2]
                           fptype* allMEs,                // output: allMEs[nevt], |M|^2 running_sum_over_helicities
#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
                           const unsigned int channelId,  // input: multichannel channel id (1 to #diagrams); 0 to disable channel enhancement
                           fptype* allNumerators,         // output: multichannel numerators[nevt], running_sum_over_helicities
                           fptype* allDenominators,       // output: multichannel denominators[nevt], running_sum_over_helicities
#endif
                           fptype_sv* jamp2_sv            // output: jamp2[nParity][ncolor][neppV] for color choice (nullptr if disabled)
                           , const int ievt00             // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)
                           )
  //ALWAYS_INLINE // attributes are not permitted in a function definition
  {
    using namespace mg5amcCpu;
    using M_ACCESS = HostAccessMomenta;         // non-trivial access: buffer includes all events
    using E_ACCESS = HostAccessMatrixElements;  // non-trivial access: buffer includes all events
    using W_ACCESS = HostAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = HostAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = HostAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = HostAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
    using NUM_ACCESS = HostAccessNumerators;    // non-trivial access: buffer includes all events
    using DEN_ACCESS = HostAccessDenominators;  // non-trivial access: buffer includes all events
#endif
    mgDebug( 0, __FUNCTION__ );

    // The variable nwf (which is specific to each P1 subdirectory, #644) is only used here
    // It is hardcoded here because various attempts to hardcode it in CPPProcess.h at generation time gave the wrong result...
    static const int nwf = 5; // #wavefunctions = #external (npar) + #internal: e.g. 5 for e+ e- -> mu+ mu- (1 internal is gamma or Z)

    // Local TEMPORARY variables for a subset of Feynman diagrams in the given CUDA event (ievt) or C++ event page (ipagV)
    // [NB these variables are reused several times (and re-initialised each time) within the same event or event page]
    // ** NB: in other words, amplitudes and wavefunctions still have TRIVIAL ACCESS: there is currently no need
    // ** NB: to have large memory structurs for wavefunctions/amplitudes in all events (no kernel splitting yet)!
    //MemoryBufferWavefunctions w_buffer[nwf]{ neppV };
    cxtype_sv w_sv[nwf][nw6]; // particle wavefunctions within Feynman diagrams (nw6 is often 6, the dimension of spin 1/2 or spin 1 particles)
    cxtype_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram

    // Proof of concept for using fptype* in the interface
    fptype* w_fp[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) w_fp[iwf] = reinterpret_cast<fptype*>( w_sv[iwf] );
    fptype* amp_fp;
    amp_fp = reinterpret_cast<fptype*>( amp_sv );

    // Local variables for the given CUDA event (ievt) or C++ event page (ipagV)
    // [jamp: sum (for one event or event page) of the invariant amplitudes for all Feynman diagrams in a given color combination]
    cxtype_sv jamp_sv[ncolor] = {}; // all zeros (NB: vector cxtype_v IS initialized to 0, but scalar cxtype is NOT, if "= {}" is missing!)

    // === Calculate wavefunctions and amplitudes for all diagrams in all processes         ===
    // === (for one event in CUDA, for one - or two in mixed mode - SIMD event pages in C++ ===
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    cxtype_sv jamp_sv_previous[ncolor] = {};
    fptype* MEs_previous = 0;
#endif
    for( int iParity = 0; iParity < nParity; ++iParity )
    { // START LOOP ON IPARITY
      const int ievt0 = ievt00 + iParity * neppV;
      constexpr size_t nxcoup = ndcoup + nicoup; // both dependent and independent couplings
      const fptype* allCOUPs[nxcoup];
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        allCOUPs[idcoup] = CD_ACCESS::idcoupAccessBufferConst( allcouplings, idcoup ); // dependent couplings, vary event-by-event
      for( size_t iicoup = 0; iicoup < nicoup; iicoup++ )
        allCOUPs[ndcoup + iicoup] = CI_ACCESS::iicoupAccessBufferConst( cIPC, iicoup ); // independent couplings, fixed for all events
      // C++ kernels take input/output buffers with momenta/MEs for one specific event (the first in the current event page)
      const fptype* momenta = M_ACCESS::ieventAccessRecordConst( allmomenta, ievt0 );
      const fptype* COUPs[nxcoup];
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        COUPs[idcoup] = CD_ACCESS::ieventAccessRecordConst( allCOUPs[idcoup], ievt0 ); // dependent couplings, vary event-by-event
      for( size_t iicoup = 0; iicoup < nicoup; iicoup++ )
        COUPs[ndcoup + iicoup] = allCOUPs[ndcoup + iicoup]; // independent couplings, fixed for all events
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
      fptype* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 );
      fptype* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
#endif

      // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
      for( int i = 0; i < ncolor; i++ ) { jamp_sv[i] = cxzero_sv(); }

#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
      // Numerators and denominators for the current event (CUDA) or SIMD event page (C++)
      fptype_sv& numerators_sv = NUM_ACCESS::kernelAccess( numerators );
      fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
#endif

      // *** DIAGRAM 1 OF 2 ***

      // Wavefunction(s) for diagram number 1
      opzxxx<M_ACCESS, W_ACCESS>( momenta, cHel[ihel][0], -1, w_fp[0], 0 ); // NB: opzxxx only uses pz

      imzxxx<M_ACCESS, W_ACCESS>( momenta, cHel[ihel][1], +1, w_fp[1], 1 ); // NB: imzxxx only uses pz

      ixzxxx<M_ACCESS, W_ACCESS>( momenta, cHel[ihel][2], -1, w_fp[2], 2 );

      oxzxxx<M_ACCESS, W_ACCESS>( momenta, cHel[ihel][3], +1, w_fp[3], 3 );

      FFV1P0_3<W_ACCESS, CI_ACCESS>( w_fp[1], w_fp[0], COUPs[0], 0., 0., w_fp[4] );

      // Amplitude(s) for diagram number 1
      FFV1_0<W_ACCESS, A_ACCESS, CI_ACCESS>( w_fp[2], w_fp[3], w_fp[4], COUPs[0], &amp_fp[0] );
#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
      if( channelId == 1 ) numerators_sv += cxabs2( amp_sv[0] );
      if( channelId != 0 ) denominators_sv += cxabs2( amp_sv[0] );
#endif
      jamp_sv[0] -= amp_sv[0];

      // *** DIAGRAM 2 OF 2 ***

      // Wavefunction(s) for diagram number 2
      FFV2_4_3<W_ACCESS, CI_ACCESS>( w_fp[1], w_fp[0], COUPs[1], COUPs[2], cIPD[0], cIPD[1], w_fp[4] );

      // Amplitude(s) for diagram number 2
      FFV2_4_0<W_ACCESS, A_ACCESS, CI_ACCESS>( w_fp[2], w_fp[3], w_fp[4], COUPs[1], COUPs[2], &amp_fp[0] );
#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
      if( channelId == 2 ) numerators_sv += cxabs2( amp_sv[0] );
      if( channelId != 0 ) denominators_sv += cxabs2( amp_sv[0] );
#endif
      jamp_sv[0] -= amp_sv[0];

      // *** COLOR CHOICE BELOW ***
      // Store the leading color flows for choice of color
      if( jamp2_sv ) // disable color choice if nullptr
        for( int icolC = 0; icolC < ncolor; icolC++ )
          jamp2_sv[ncolor * iParity + icolC] += cxabs2( jamp_sv[icolC] );

      // *** COLOR MATRIX BELOW ***
      // (This method used to be called CPPProcess::matrix_1_epem_mupmum()?)

      // The color denominators (initialize all array elements, with ncolor=1)
      // [NB do keep 'static' for these constexpr arrays, see issue #283]
      static constexpr fptype2 denom[ncolor] = { 1 }; // 1-D array[1]

      // The color matrix (initialize all array elements, with ncolor=1)
      // [NB do keep 'static' for these constexpr arrays, see issue #283]
      static constexpr fptype2 cf[ncolor][ncolor] = { { 1 } }; // 2-D array[1][1]

      // Pre-compute a constexpr triangular color matrix properly normalized #475
      struct TriangularNormalizedColorMatrix
      {
        // See https://stackoverflow.com/a/34465458
        __host__ __device__ constexpr TriangularNormalizedColorMatrix()
          : value()
        {
          for( int icol = 0; icol < ncolor; icol++ )
          {
            // Diagonal terms
            value[icol][icol] = cf[icol][icol] / denom[icol];
            // Off-diagonal terms
            for( int jcol = icol + 1; jcol < ncolor; jcol++ )
              value[icol][jcol] = 2 * cf[icol][jcol] / denom[icol];
          }
        }
        fptype2 value[ncolor][ncolor];
      };
      static constexpr auto cf2 = TriangularNormalizedColorMatrix();

#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      if( iParity == 0 ) // NB: first page is 0! skip even pages, compute on odd pages
      {
        // Mixed fptypes: delay color algebra and ME updates to next (odd) ipagV
        for( int icol = 0; icol < ncolor; icol++ )
          jamp_sv_previous[icol] = jamp_sv[icol];
        MEs_previous = MEs;
        continue; // go to next iParity in the loop: skip color algebra and ME update on odd pages
      }
      fptype_sv deltaMEs_previous = { 0 };
#endif

      // Sum and square the color flows to get the matrix element
      // (compute |M|^2 by squaring |M|, taking into account colours)
      // Sum and square the color flows to get the matrix element
      // (compute |M|^2 by squaring |M|, taking into account colours)
      fptype_sv deltaMEs = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes

      // Use the property that M is a real matrix (see #475):
      // we can rewrite the quadratic form (A-iB)(M)(A+iB) as AMA - iBMA + iBMA + BMB = AMA + BMB
      // In addition, on C++ use the property that M is symmetric (see #475),
      // and also use constexpr to compute "2*" and "/denom[icol]" once and for all at compile time:
      // we gain (not a factor 2...) in speed here as we only loop over the up diagonal part of the matrix.
      // Strangely, CUDA is slower instead, so keep the old implementation for the moment.
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      fptype2_sv jampR_sv[ncolor] = { 0 };
      fptype2_sv jampI_sv[ncolor] = { 0 };
      for( int icol = 0; icol < ncolor; icol++ )
      {
        jampR_sv[icol] = fpvmerge( cxreal( jamp_sv_previous[icol] ), cxreal( jamp_sv[icol] ) );
        jampI_sv[icol] = fpvmerge( cximag( jamp_sv_previous[icol] ), cximag( jamp_sv[icol] ) );
      }
#endif
      for( int icol = 0; icol < ncolor; icol++ )
      {
        // === C++ START ===
        // Diagonal terms
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        fptype2_sv& jampRi_sv = jampR_sv[icol];
        fptype2_sv& jampIi_sv = jampI_sv[icol];
#else
        fptype2_sv jampRi_sv = (fptype2_sv)( cxreal( jamp_sv[icol] ) );
        fptype2_sv jampIi_sv = (fptype2_sv)( cximag( jamp_sv[icol] ) );
#endif
        fptype2_sv ztempR_sv = cf2.value[icol][icol] * jampRi_sv;
        fptype2_sv ztempI_sv = cf2.value[icol][icol] * jampIi_sv;
        // Off-diagonal terms
        for( int jcol = icol + 1; jcol < ncolor; jcol++ )
        {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          fptype2_sv& jampRj_sv = jampR_sv[jcol];
          fptype2_sv& jampIj_sv = jampI_sv[jcol];
#else
          fptype2_sv jampRj_sv = (fptype2_sv)( cxreal( jamp_sv[jcol] ) );
          fptype2_sv jampIj_sv = (fptype2_sv)( cximag( jamp_sv[jcol] ) );
#endif
          ztempR_sv += cf2.value[icol][jcol] * jampRj_sv;
          ztempI_sv += cf2.value[icol][jcol] * jampIj_sv;
        }
        fptype2_sv deltaMEs2 = ( jampRi_sv * ztempR_sv + jampIi_sv * ztempI_sv );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        deltaMEs_previous += fpvsplit0( deltaMEs2 );
        deltaMEs += fpvsplit1( deltaMEs2 );
#else
        deltaMEs += deltaMEs2;
#endif
        // === C++ END ===
      }

      // *** STORE THE RESULTS ***

      // NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event(s)
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv += deltaMEs; // fix #435
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      fptype_sv& MEs_sv_previous = E_ACCESS::kernelAccess( MEs_previous );
      MEs_sv_previous += deltaMEs_previous;
#endif
    } // END LOOP ON IPARITY
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  CPPProcess::CPPProcess( bool verbose,
                          bool debug )
    : m_verbose( verbose )
    , m_debug( debug )
#ifndef MGONGPU_HARDCODE_PARAM
    , m_pars( 0 )
#endif
    , m_masses()
  {
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

  //--------------------------------------------------------------------------

  CPPProcess::~CPPProcess() {}

  //--------------------------------------------------------------------------

#ifndef MGONGPU_HARDCODE_PARAM
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
#else
  // Initialize process (with hardcoded parameters)
  void
  CPPProcess::initProc( const std::string& /*param_card_name*/ )
  {
    // Use hardcoded physics parameters
    if( m_verbose )
    {
      Parameters_sm::printIndependentParameters();
      Parameters_sm::printIndependentCouplings();
      //Parameters_sm::printDependentParameters(); // now computed event-by-event (running alphas #373)
      //Parameters_sm::printDependentCouplings(); // now computed event-by-event (running alphas #373)
    }
    // Set external particle masses for this matrix element
    m_masses.push_back( Parameters_sm::ZERO );
    m_masses.push_back( Parameters_sm::ZERO );
    m_masses.push_back( Parameters_sm::ZERO );
    m_masses.push_back( Parameters_sm::ZERO );
  }
#endif

  __global__ void /* clang-format off */
  computeDependentCouplings( const fptype* allgs, // input: Gs[nevt]
                             fptype* allcouplings // output: couplings[nevt*ndcoup*2]
                             , const int nevt     // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  ) /* clang-format on */
  {
    using namespace mg5amcCpu;
    using G_ACCESS = HostAccessGs;
    using C_ACCESS = HostAccessCouplings;
    for( int ipagV = 0; ipagV < nevt / neppV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      const fptype* gs = MemoryAccessGs::ieventAccessRecordConst( allgs, ievt0 );
      fptype* couplings = MemoryAccessCouplings::ieventAccessRecord( allcouplings, ievt0 );
      G2COUP<G_ACCESS, C_ACCESS>( gs, couplings );
    }
  }

  //--------------------------------------------------------------------------

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
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    assert( npagV % 2 == 0 );     // SANITY CHECK for mixed fptypes: two neppV-pages are merged to one 2*neppV-page
    const int npagV2 = npagV / 2; // loop on two SIMD pages (neppV events) at a time
#else
    const int npagV2 = npagV;            // loop on one SIMD page (neppV events) at a time
#endif
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 ) {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      const int ievt00 = ipagV2 * neppV * 2; // loop on two SIMD pages (neppV events) at a time
#else
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
#endif
      for( int ihel = 0; ihel < ncomb; ihel++ ) {
        // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
        for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
          const int ievt = ievt00 + ieppV;
          allMEs[ievt] = 0;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          allMEs[ievt2] = 0;
#endif
        }
        constexpr fptype_sv* jamp2_sv = nullptr; // no need for color selection during helicity filtering
        constexpr unsigned int channelId = 0; // disable single-diagram channel enhancement
        calculate_wavefunctions( ihel, allmomenta, allcouplings, allMEs, channelId, allNumerators, allDenominators, jamp2_sv, ievt00 );
        for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
          const int ievt = ievt00 + ieppV;
          if( allMEs[ievt] != 0 ) {// NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
            isGoodHel[ihel] = true;
          }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          if( allMEs[ievt2] != 0 ) { // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
            isGoodHel[ihel] = true;
          }
#endif
        }
      }
    }
  }

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) { // input: isGoodHel[ncomb] - host array (CUDA and C++)
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes
    for( int ihel = 0; ihel < ncomb; ihel++ ) {
      if( isGoodHel[ihel] ) {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  //--------------------------------------------------------------------------
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
            ) /* clang-format on */
  {
    mgDebugInitialise();

    // SANITY CHECKS for cudacpp code generation (see issues #272 and #343 and PRs #619, #626, #360 and #396)
    // These variable are not used anywhere else in the code and their scope is limited to this sanity check
    {
      // nprocesses>1 was last observed for "mirror processes" in uux_ttx in the 270 branch (see issue #343 and PRs #360 and #396)
      constexpr int nprocesses = 1;
      static_assert( nprocesses == 1, "Assume nprocesses == 1" );
      // process_id corresponds to the index of DSIG1 Fortran functions (must be 1 because cudacpp is unable to handle DSIG2)
      constexpr int process_id = 1; // code generation source: madevent + cudacpp exporter
      static_assert( process_id == 1, "Assume process_id == 1" );
    }

    // Denominators: spins, colors and identical particles
    constexpr int helcolDenominators[1] = { 4 }; // assume nprocesses == 1 (#272 and #343)

    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;   // non-trivial access: buffer includes all events
    using DEN_ACCESS = HostAccessDenominators; // non-trivial access: buffer includes all events

    // Start sigmaKin_lines
    // === PART 0 - INITIALISATION (before calculate_wavefunctions) ===
    // Reset the "matrix elements" - running sums of |M|^2 over helicities for the given event
    const int npagV = nevt / neppV;
    for( int ipagV = 0; ipagV < npagV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv = fptype_sv{ 0 };
      fptype* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 );
      fptype* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
      fptype_sv& numerators_sv = NUM_ACCESS::kernelAccess( numerators );
      fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
      numerators_sv = fptype_sv{ 0 };
      denominators_sv = fptype_sv{ 0 };
    }

    // === PART 1 - HELICITY LOOP: CALCULATE WAVEFUNCTIONS ===
    // (in both CUDA and C++, using precomputed good helicities)

    // *** START OF PART 1b - C++ (loop on event pages)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    assert( npagV % 2 == 0 );     // SANITY CHECK for mixed fptypes: two neppV-pages are merged to one 2*neppV-page
    const int npagV2 = npagV / 2; // loop on two SIMD pages (neppV events) at a time
#else
    const int npagV2 = npagV;            // loop on one SIMD page (neppV events) at a time
#endif
#ifdef _OPENMP
    // OMP multithreading #575 (NB: tested only with gcc11 so far)
    // See https://www.openmp.org/specifications/
    // - default(none): no variables are shared by default
    // - shared: as the name says
    // - private: give each thread its own copy, without initialising
    // - firstprivate: give each thread its own copy, and initialise with value from outside
#define _OMPLIST0 allcouplings, allMEs, allmomenta, allrndcol, allrndhel, allselcol, allselhel, cGoodHel, cNGoodHel, npagV2
#define _OMPLIST1 , allDenominators, allNumerators, channelId, mgOnGpu::icolamp
#pragma omp parallel for default( none ) shared( _OMPLIST0 _OMPLIST1 )
#undef _OMPLIST0
#undef _OMPLIST1
#endif // _OPENMP
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 ) {
      // Running sum of partial amplitudes squared for event by event color selection (#402)
      // (jamp2[nParity][ncolor][neppV] for the SIMD vector - or the two SIMD vectors - of events processed in calculate_wavefunctions)
      fptype_sv jamp2_sv[nParity * ncolor] = { 0 };
      fptype_sv MEs_ighel[ncomb] = { 0 };    // sum of MEs for all good helicities up to ighel (for the first - and/or only - neppV page)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      fptype_sv MEs_ighel2[ncomb] = { 0 };   // sum of MEs for all good helicities up to ighel (for the second neppV page)
      const int ievt00 = ipagV2 * neppV * 2; // loop on two SIMD pages (neppV events) at a time
#else
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
#endif
      for( int ighel = 0; ighel < cNGoodHel; ighel++ ) {
        const int ihel = cGoodHel[ighel];
        calculate_wavefunctions( ihel, allmomenta, allcouplings, allMEs, channelId, allNumerators, allDenominators, jamp2_sv, ievt00 );
        MEs_ighel[ighel] = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 ) );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        MEs_ighel2[ighel] = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 + neppV ) );
#endif
      }
      // Event-by-event random choice of helicity #403
      for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
        const int ievt = ievt00 + ieppV;
        //printf( "sigmaKin: ievt=%4d rndhel=%f\n", ievt, allrndhel[ievt] );
        for( int ighel = 0; ighel < cNGoodHel; ighel++ ) {
#if defined MGONGPU_CPPSIMD
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel][ieppV] / MEs_ighel[cNGoodHel - 1][ieppV] );
#else
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel] / MEs_ighel[cNGoodHel - 1] );
#endif
          if( okhel ) {
            const int ihelF = cGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt] = ihelF;
            //printf( "sigmaKin: ievt=%4d ihel=%4d\n", ievt, ihelF );
            break;
          }
        }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        const int ievt2 = ievt00 + ieppV + neppV;
        //printf( "sigmaKin: ievt=%4d rndhel=%f\n", ievt2, allrndhel[ievt2] );
        for( int ighel = 0; ighel < cNGoodHel; ighel++ ) {
          if( allrndhel[ievt2] < ( MEs_ighel2[ighel][ieppV] / MEs_ighel2[cNGoodHel - 1][ieppV] ) ) {
            const int ihelF = cGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt2] = ihelF;
            //printf( "sigmaKin: ievt=%4d ihel=%4d\n", ievt, ihelF );
            break;
          }
        }
#endif
      }
      const int channelIdC = channelId - 1; // coloramps.h uses the C array indexing starting at 0
      // Event-by-event random choice of color #402
      fptype_sv targetamp[ncolor] = { 0 };
      for( int icolC = 0; icolC < ncolor; icolC++ ) {
        if( icolC == 0 )
          targetamp[icolC] = fptype_sv{ 0 };
        else
          targetamp[icolC] = targetamp[icolC - 1];
        if( mgOnGpu::icolamp[channelIdC][icolC] ) targetamp[icolC] += jamp2_sv[icolC];
      }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      fptype_sv targetamp2[ncolor] = { 0 };
      for( int icolC = 0; icolC < ncolor; icolC++ ) {
        if( icolC == 0 )
          targetamp2[icolC] = fptype_sv{ 0 };
        else
          targetamp2[icolC] = targetamp2[icolC - 1];
        if( mgOnGpu::icolamp[channelIdC][icolC] ) targetamp2[icolC] += jamp2_sv[ncolor + icolC];
      }
#endif
      for( int ieppV = 0; ieppV < neppV; ++ieppV ) {
        const int ievt = ievt00 + ieppV;
        //printf( "sigmaKin: ievt=%4d rndcol=%f\n", ievt, allrndcol[ievt] );
        for( int icolC = 0; icolC < ncolor; icolC++ ) {
#if defined MGONGPU_CPPSIMD
          const bool okcol = allrndcol[ievt] < ( targetamp[icolC][ieppV] / targetamp[ncolor - 1][ieppV] );
#else
          const bool okcol = allrndcol[ievt] < ( targetamp[icolC] / targetamp[ncolor - 1] );
#endif
          if( okcol ) {
            allselcol[ievt] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
            break;
          }
        }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        const int ievt2 = ievt00 + ieppV + neppV;
        //printf( "sigmaKin: ievt=%4d rndcol=%f\n", ievt2, allrndcol[ievt2] );
        for( int icolC = 0; icolC < ncolor; icolC++ ) {
          if( allrndcol[ievt2] < ( targetamp2[icolC][ieppV] / targetamp2[ncolor - 1][ieppV] ) ) {
            allselcol[ievt2] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
            break;
          }
        }
#endif
      }
    }
    // *** END OF PART 1b - C++ (loop on event pages)

    // === PART 2 - FINALISATION (after calculate_wavefunctions) ===
    // Get the final |M|^2 as an average over helicities/colors of the running sum of |M|^2 over helicities for the given event
    // [NB 'sum over final spins, average over initial spins', eg see
    // https://www.uzh.ch/cmsssl/physik/dam/jcr:2e24b7b1-f4d7-4160-817e-47b13dbf1d7c/Handout_4_2016-UZH.pdf]
    for( int ipagV = 0; ipagV < npagV; ++ipagV ) {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv /= helcolDenominators[0];
      if( channelId > 0 ) {
        fptype* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 );
        fptype* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
        fptype_sv& numerators_sv = NUM_ACCESS::kernelAccess( numerators );
        fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
        MEs_sv *= numerators_sv / denominators_sv;
      }
    }
    mgDebugFinalise();
  }

} // end namespace
