// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: S. Roiser (Nov 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Roiser, A. Valassi (2021-2023) for the MG5aMC CUDACPP plugin.

#pragma once

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"           // for CPPProcess
#include "CrossSectionKernels.h"  // for flagAbnormalMEs
#include "MatrixElementKernels.h" // for MatrixElementKernelHost, MatrixElementKernelDevice
#include "MemoryAccessMomenta.h"  // for MemoryAccessMomenta::neppV
#include "MemoryBuffers.h"        // for HostBufferMomenta, DeviceBufferMomenta etc

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <type_traits>

namespace mg5amcCpu {
  //--------------------------------------------------------------------------
  /**
   * A base class for a class whose pointer is passed between Fortran and C++.
   * This is not really necessary, but it allows minimal type checks on all such pointers.
   */
  struct CppObjectInFortran
  {
    CppObjectInFortran() {}
    virtual ~CppObjectInFortran() {}
  };

  //--------------------------------------------------------------------------
  /**
   * A templated class for calling the CUDA/C++ matrix element calculations of the event generation workflow.
   * The FORTRANFPTYPE template parameter indicates the precision of the Fortran momenta from MadEvent (float or double).
   * The precision of the matrix element calculation is hardcoded in the fptype typedef in CUDA/C++.
   *
   * The Fortran momenta passed in are in the form of
   *   DOUBLE PRECISION P_MULTI(0:3, NEXTERNAL, VECSIZE_USED)
   * where the dimensions are <np4F(#momenta)>, <nparF(#particles)>, <nevtF(#events)>.
   * In memory, this is stored in a way that C reads as an array P_MULTI[nevtF][nparF][np4F].
   * The CUDA/C++ momenta are stored as an array[npagM][npar][np4][neppV] with nevt=npagM*neppV.
   * The Bridge is configured to store nevt==nevtF events in CUDA/C++.
   * It also checks that Fortran and C++ parameters match, nparF==npar and np4F==np4.
   *
   * The cpu/gpu sequences take FORTRANFPTYPE* (not fptype*) momenta/MEs.
   * This allows mixing double in MadEvent Fortran with float in CUDA/C++ sigmaKin.
   * In the fcheck_sa.f test, Fortran uses double while CUDA/C++ may use double or float.
   * In the check_sa "--bridge" test, everything is implemented in fptype (double or float).
   */
  template<typename FORTRANFPTYPE>
  class Bridge final : public CppObjectInFortran
  {
  public:
    /**
     * Constructor
     *
     * @param nevtF (VECSIZE_USED, vector.inc) number of events in Fortran array loops (VECSIZE_USED <= VECSIZE_MEMMAX)
     * @param nparF (NEXTERNAL, nexternal.inc) number of external particles in Fortran arrays (KEPT FOR SANITY CHECKS ONLY)
     * @param np4F number of momenta components, usually 4, in Fortran arrays (KEPT FOR SANITY CHECKS ONLY)
     */
    Bridge( unsigned int nevtF, unsigned int nparF, unsigned int np4F );

    /**
     * Destructor
     */
    virtual ~Bridge() {}

    // Delete copy/move constructors and assignment operators
    Bridge( const Bridge& ) = delete;
    Bridge( Bridge&& ) = delete;
    Bridge& operator=( const Bridge& ) = delete;
    Bridge& operator=( Bridge&& ) = delete;

    /**
     * Sequence to be executed for the vectorized CPU matrix element calculation
     *
     * @param momenta the pointer to the input 4-momenta
     * @param gs the pointer to the input Gs (running QCD coupling constant alphas)
     * @param rndhel the pointer to the input random numbers for helicity selection
     * @param rndcol the pointer to the input random numbers for color selection
     * @param channelId the Feynman diagram to enhance in multi-channel mode if 1 to n (disable multi-channel if 0)
     * @param mes the pointer to the output matrix elements
     * @param selhel the pointer to the output selected helicities
     * @param selcol the pointer to the output selected colors
     * @param goodHelOnly quit after computing good helicities?
     */
    void cpu_sequence( const FORTRANFPTYPE* momenta,
                       const FORTRANFPTYPE* gs,
                       const FORTRANFPTYPE* rndhel,
                       const FORTRANFPTYPE* rndcol,
                       FORTRANFPTYPE* mes,
                       int* selhel,
                       int* selcol,
                       const bool goodHelOnly = false );

    // Return the number of good helicities (-1 initially when they have not yet been calculated)
    int nGoodHel() const { return m_nGoodHel; }

    // Return the total number of helicities (expose cudacpp ncomb in the Bridge interface to Fortran)
    constexpr int nTotHel() const { return CPPProcess::ncomb; }

  private:
    unsigned int m_nevt; // number of events
    int m_nGoodHel;      // the number of good helicities (-1 initially when they have not yet been calculated)
    HostBufferMomenta m_hstMomentaC;
    HostBufferGs m_hstGs;
    HostBufferRndNumHelicity m_hstRndHel;
    HostBufferRndNumColor m_hstRndCol;
    HostBufferMatrixElements m_hstMEs;
    HostBufferSelectedHelicity m_hstSelHel;
    HostBufferSelectedColor m_hstSelCol;
    MatrixElementKernelHost m_pmek;
  };

  //--------------------------------------------------------------------------
  //
  // Forward declare transposition methods
  //

  template<typename Tin, typename Tout>
  void hst_transposeMomentaF2C( const Tin* in, Tout* out, const unsigned int nevt );

  template<typename Tin, typename Tout>
  void hst_transposeMomentaC2F( const Tin* in, Tout* out, const unsigned int nevt );

  //--------------------------------------------------------------------------
  //
  // Implementations of member functions of class Bridge
  //

  template<typename FORTRANFPTYPE>
  Bridge<FORTRANFPTYPE>::Bridge( unsigned int nevtF, unsigned int nparF, unsigned int np4F )
    : m_nevt( nevtF )
    , m_nGoodHel( -1 )
    , m_hstMomentaC( m_nevt )
    , m_hstGs( m_nevt )
    , m_hstRndHel( m_nevt )
    , m_hstRndCol( m_nevt )
    , m_hstMEs( m_nevt )
    , m_hstSelHel( m_nevt )
    , m_hstSelCol( m_nevt )
    , m_pmek( m_hstMomentaC, m_hstGs, m_hstRndHel, m_hstRndCol, m_hstMEs, m_hstSelHel, m_hstSelCol, m_nevt )
  {
    if( nparF != CPPProcess::npar ) throw std::runtime_error( "Bridge constructor: npar mismatch" );
    if( np4F != CPPProcess::np4 ) throw std::runtime_error( "Bridge constructor: np4 mismatch" );
    std::cout << "WARNING! Instantiate host Bridge (nevt=" << m_nevt << ")" << std::endl;
    // Create a process object, read param card and set parameters
    // FIXME: the process instance can happily go out of scope because it is only needed to read parameters?
    // FIXME: the CPPProcess should really be a singleton? what if fbridgecreate is called from several Fortran threads?
    CPPProcess process( /*verbose=*/false );
    std::string paramCard = "../../Cards/param_card.dat";
    if( !std::filesystem::exists( paramCard ) )
    {
      paramCard = "../" + paramCard;
    }
    process.initProc( paramCard );
  }

  template<typename FORTRANFPTYPE>
  void Bridge<FORTRANFPTYPE>::cpu_sequence( const FORTRANFPTYPE* momenta,
                                            const FORTRANFPTYPE* gs,
                                            const FORTRANFPTYPE* rndhel,
                                            const FORTRANFPTYPE* rndcol,
                                            FORTRANFPTYPE* mes,
                                            int* selhel,
                                            int* selcol,
                                            const bool goodHelOnly )
  {
    hst_transposeMomentaF2C( momenta, m_hstMomentaC.data(), m_nevt );
    if constexpr( std::is_same_v<FORTRANFPTYPE, fptype> )
    {
      memcpy( m_hstGs.data(), gs, m_nevt * sizeof( FORTRANFPTYPE ) );
      memcpy( m_hstRndHel.data(), rndhel, m_nevt * sizeof( FORTRANFPTYPE ) );
      memcpy( m_hstRndCol.data(), rndcol, m_nevt * sizeof( FORTRANFPTYPE ) );
    }
    else
    {
      std::copy( gs, gs + m_nevt, m_hstGs.data() );
      std::copy( rndhel, rndhel + m_nevt, m_hstRndHel.data() );
      std::copy( rndcol, rndcol + m_nevt, m_hstRndCol.data() );
    }
    if( m_nGoodHel < 0 )
    {
      m_nGoodHel = m_pmek.computeGoodHelicities();
      if( m_nGoodHel < 0 ) throw std::runtime_error( "Bridge cpu_sequence: computeGoodHelicities returned nGoodHel<0" );
    }
    if( goodHelOnly ) return;
    m_pmek.computeMatrixElements();
    flagAbnormalMEs( m_hstMEs.data(), m_nevt );
    if constexpr( std::is_same_v<FORTRANFPTYPE, fptype> )
    {
      memcpy( mes, m_hstMEs.data(), m_hstMEs.bytes() );
      memcpy( selhel, m_hstSelHel.data(), m_hstSelHel.bytes() );
      memcpy( selcol, m_hstSelCol.data(), m_hstSelCol.bytes() );
    }
    else
    {
      std::copy( m_hstMEs.data(), m_hstMEs.data() + m_nevt, mes );
      std::copy( m_hstSelHel.data(), m_hstSelHel.data() + m_nevt, selhel );
      std::copy( m_hstSelCol.data(), m_hstSelCol.data() + m_nevt, selcol );
    }
  }

  //--------------------------------------------------------------------------
  //
  // Implementations of transposition methods
  // - FORTRAN arrays: P_MULTI(0:3, NEXTERNAL, VECSIZE_USED) ==> p_multi[nevtF][nparF][np4F] in C++ (AOS)
  // - C++ array: momenta[npagM][npar][np4][neppV] with nevt=npagM*neppV (AOSOA)
  //


  template<typename Tin, typename Tout, bool F2C>
  void hst_transposeMomenta( const Tin* in, Tout* out, const unsigned int nevt )
  {
    constexpr bool oldImplementation = false; // default: use new implementation
    if constexpr( oldImplementation )
    {
      // SR initial implementation
      constexpr unsigned int part = CPPProcess::npar;
      constexpr unsigned int mome = CPPProcess::np4;
      constexpr unsigned int strd = neppV;
      unsigned int arrlen = nevt * part * mome;
      for( unsigned int pos = 0; pos < arrlen; ++pos )
      {
        unsigned int page_i = pos / ( strd * mome * part );
        unsigned int rest_1 = pos % ( strd * mome * part );
        unsigned int part_i = rest_1 / ( strd * mome );
        unsigned int rest_2 = rest_1 % ( strd * mome );
        unsigned int mome_i = rest_2 / strd;
        unsigned int strd_i = rest_2 % strd;
        unsigned int inpos =
          ( page_i * strd + strd_i ) // event number
            * ( part * mome )        // event size (pos of event)
          + part_i * mome            // particle inside event
          + mome_i;                  // momentum inside particle
        if constexpr( F2C )          // needs c++17 and cuda >=11.2 (#333)
          out[pos] = in[inpos];      // F2C (Fortran to C)
        else
          out[inpos] = in[pos]; // C2F (C to Fortran)
      }
    }
    else
    {
      // AV attempt another implementation: this is slightly faster (better c++ pipelining?)
      // [NB! this is not a transposition, it is an AOS to AOSOA conversion: if neppV=1, a memcpy is enough]
      // F-style: AOS[nevtF][nparF][np4F]
      // C-style: AOSOA[npagM][npar][np4][neppV] with nevt=npagM*neppV
      constexpr unsigned int npar = CPPProcess::npar;
      constexpr unsigned int np4 = CPPProcess::np4;
      if constexpr( neppV == 1 && std::is_same_v<Tin, Tout> )
      {
        memcpy( out, in, nevt * npar * np4 * sizeof( Tin ) );
      }
      else
      {
        const unsigned int npagM = nevt / neppV;
        assert( nevt % neppV == 0 ); // number of events is not a multiple of neppV???
        for( unsigned int ipagM = 0; ipagM < npagM; ipagM++ )
          for( unsigned int ip4 = 0; ip4 < np4; ip4++ )
            for( unsigned int ipar = 0; ipar < npar; ipar++ )
              for( unsigned int ieppM = 0; ieppM < neppV; ieppM++ )
              {
                unsigned int ievt = ipagM * neppV + ieppM;
                unsigned int cpos = ipagM * npar * np4 * neppV + ipar * np4 * neppV + ip4 * neppV + ieppM;
                unsigned int fpos = ievt * npar * np4 + ipar * np4 + ip4;
                if constexpr( F2C )
                  out[cpos] = in[fpos]; // F2C (Fortran to C)
                else
                  out[fpos] = in[cpos]; // C2F (C to Fortran)
              }
      }
    }
  }

  template<typename Tin, typename Tout>
  void hst_transposeMomentaF2C( const Tin* in, Tout* out, const unsigned int nevt )
  {
    constexpr bool F2C = true;
    hst_transposeMomenta<Tin, Tout, F2C>( in, out, nevt );
  }

  template<typename Tin, typename Tout>
  void hst_transposeMomentaC2F( const Tin* in, Tout* out, const unsigned int nevt )
  {
    constexpr bool F2C = false;
    hst_transposeMomenta<Tin, Tout, F2C>( in, out, nevt );
  }

  //--------------------------------------------------------------------------
}
