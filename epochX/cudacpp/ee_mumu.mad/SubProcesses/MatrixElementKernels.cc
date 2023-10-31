// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2022-2023) for the MG5aMC CUDACPP plugin.

#include "MatrixElementKernels.h"

#include "CPPProcess.h"
#include "CudaRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"

#include <sstream>

//============================================================================

namespace mg5amcCpu
{

  //--------------------------------------------------------------------------

  MatrixElementKernelHost::MatrixElementKernelHost( const fptype_v* momenta,         // input: momenta
                                                    const BufferGs& gs,                   // input: gs for alphaS
                                                    const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                                                    const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                                                    BufferMatrixElements& matrixElements, // output: matrix elements
                                                    BufferSelectedHelicity& selhel,       // output: helicity selection
                                                    BufferSelectedColor& selcol,          // output: color selection
                                                    const size_t nevt )
    : MatrixElementKernelBase( momenta, gs, rndhel, rndcol, matrixElements, selhel, selcol )
    , NumberOfEvents( nevt )
    , m_couplings( nevt )
  {
    if( m_matrixElements.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelHost: matrixElements must be a host array" );
    if( this->nevt() != m_matrixElements.nevt() ) throw std::runtime_error( "MatrixElementKernelHost: nevt mismatch with matrixElements" );
    // Sanity checks for memory access (momenta buffer)
    static_assert( ispoweroftwo( neppV ), "neppV is not a power of 2" );
    if( nevt % neppV != 0 )
    {
      std::ostringstream sstr;
      sstr << "MatrixElementKernelHost: nevt should be a multiple of neppV=" << neppV;
      throw std::runtime_error( sstr.str() );
    }
    // Fail gently and avoid "Illegal instruction (core dumped)" if the host does not support the SIMD used in the ME calculation
    // Note: this prevents a crash on pmpe04 but not on some github CI nodes?
    // [NB: SIMD vectorization in mg5amc C++ code is only used in the ME calculation below MatrixElementKernelHost!]
    if( !MatrixElementKernelHost::hostSupportsSIMD() )
      throw std::runtime_error( "Host does not support the SIMD implementation of MatrixElementKernelsHost" );
  }

  //--------------------------------------------------------------------------

  int MatrixElementKernelHost::computeGoodHelicities()
  {
    constexpr int ncomb = CPPProcess::ncomb; // the number of helicity combinations
    HostBufferHelicityMask hstIsGoodHel( ncomb );
    // ... 0d1. Compute good helicity mask on the host
    computeDependentCouplings( m_gs.data(), m_couplings.data(), m_gs.size() );
    sigmaKin_getGoodHel( m_momenta, m_couplings.data(), m_matrixElements.data(), hstIsGoodHel.data(), nevt() );
    // ... 0d2. Copy back good helicity list to static memory on the host
    // [FIXME! REMOVE THIS STATIC THAT BREAKS MULTITHREADING?]
    return sigmaKin_setGoodHel( hstIsGoodHel.data() );
  }

  //--------------------------------------------------------------------------

  void MatrixElementKernelHost::computeMatrixElements()
  {
    computeDependentCouplings( m_gs.data(), m_couplings.data(), m_gs.size() );
    sigmaKin( m_momenta, m_couplings.data(), m_rndhel.data(), m_rndcol.data(), m_matrixElements.data(), m_selhel.data(), m_selcol.data(), nevt() );
  }

  //--------------------------------------------------------------------------

  // Does this host system support the SIMD used in the matrix element calculation?
  bool MatrixElementKernelHost::hostSupportsSIMD( const bool verbose )
  {
#if defined __AVX512VL__
    bool known = true;
    bool ok = __builtin_cpu_supports( "avx512vl" );
    const std::string tag = "skylake-avx512 (AVX512VL)";
#elif defined __AVX2__
    bool known = true;
    bool ok = __builtin_cpu_supports( "avx2" );
    const std::string tag = "haswell (AVX2)";
#elif defined __SSE4_2__
    bool known = true;
    bool ok = __builtin_cpu_supports( "sse4.2" );
    const std::string tag = "nehalem (SSE4.2)";
#else
    bool known = true;
    bool ok = true;
    const std::string tag = "none";
#endif
    if( verbose )
    {
      if( tag == "none" )
        std::cout << "INFO: The application does not require the host to support any AVX feature" << std::endl;
      else if( ok && known )
        std::cout << "INFO: The application is built for " << tag << " and the host supports it" << std::endl;
      else if( ok )
        std::cout << "WARNING: The application is built for " << tag << " but it is unknown if the host supports it" << std::endl;
      else
        std::cout << "ERROR! The application is built for " << tag << " but the host does not support it" << std::endl;
    }
    return ok;
  }

}
