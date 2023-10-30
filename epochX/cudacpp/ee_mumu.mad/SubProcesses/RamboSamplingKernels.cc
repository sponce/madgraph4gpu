// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2021-2023) for the MG5aMC CUDACPP plugin.

#include "RamboSamplingKernels.h"

#include "CudaRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessRandomNumbers.h"
#include "MemoryAccessWeights.h"
#include "MemoryBuffers.h"
#include "rambo.h" // inline implementation of RAMBO algorithms and kernels

#include <sstream>
#include <cstring>

#include <immintrin.h>

namespace mg5amcCpu {

  RamboSamplingKernelHost::RamboSamplingKernelHost( const fptype energy,               // input: energy
                                                    const BufferRndNumMomenta& rndmom, // input: random numbers in [0,1]
                                                    BufferMomenta& momenta,            // output: momenta
                                                    BufferWeights& weights,            // output: weights
                                                    const size_t nevt )
    : NumberOfEvents( nevt )
    , m_energy( energy )
    , m_rndmom( rndmom )
    , m_momenta( momenta )
    , m_weights( weights ) {
    if( this->nevt() != m_rndmom.nevt() ) throw std::runtime_error( "RamboSamplingKernelHost: nevt mismatch with rndmom" );
    if( this->nevt() != m_momenta.nevt() ) throw std::runtime_error( "RamboSamplingKernelHost: nevt mismatch with momenta" );
    if( this->nevt() != m_weights.nevt() ) throw std::runtime_error( "RamboSamplingKernelHost: nevt mismatch with weights" );
    // Sanity checks for memory access (momenta buffer)
    if( nevt % neppV != 0 ) {
      std::ostringstream sstr;
      sstr << "RamboSamplingKernelHost: nevt should be a multiple of neppV=" << neppV;
      throw std::runtime_error( sstr.str() );
    }
    // Sanity checks for memory access (random number buffer)
    constexpr int neppR = MemoryAccessRandomNumbers::neppR; // AOSOA layout
    static_assert( ispoweroftwo( neppR ), "neppR is not a power of 2" );
    if( nevt % neppR != 0 ) {
      std::ostringstream sstr;
      sstr << "RamboSamplingKernelHost: nevt should be a multiple of neppR=" << neppR;
      throw std::runtime_error( sstr.str() );
    }
  }

  void RamboSamplingKernelHost::getMomentaInitial() {
    const fptype_v mom{m_energy / 2};
    const fptype_v zero{0};
    memset(m_momenta.data(), 0, nevt()*neppV*8);
    for( size_t ievt = 0; ievt < nevt(); ievt += neppV ) {
      fptype_v* momenta = reinterpret_cast<fptype_v*>(&m_momenta.data()[ievt * 8]);
      momenta[0] = mom;
      momenta[3] = mom;
      momenta[4] = mom;
      momenta[7] = -mom;
    }
  }

  inline fptype_v __attribute__( ( always_inline ) ) sqrt_v(fptype_v in) {
    return fptype_v{ sqrt(in[0]), sqrt(in[1]), sqrt(in[2]), sqrt(in[3]) };
  }
  inline fptype_v __attribute__( ( always_inline ) ) rsqrt_v(fptype_v in) {
    return 1./sqrt_v(in);
  }
  inline fptype_v __attribute__( ( always_inline ) ) log_v(fptype_v in) {
    return fptype_v{ log(in[0]), log(in[1]), log(in[2]), log(in[3]) };
  }
  inline std::pair<fptype_v, fptype_v> __attribute__( ( always_inline ) ) sincos_v(fptype_v in) {
    return {
      fptype_v{ sin(in[0]), sin(in[1]), sin(in[2]), sin(in[3]) },
      fptype_v{ cos(in[0]), cos(in[1]), cos(in[2]), cos(in[3]) }
    };
  }

  void RamboSamplingKernelHost::getMomentaFinal() {
    const fptype twopi = 8. * atan( 1. );
    
    for( size_t ievt = 0; ievt < nevt(); ievt+=neppV ) {
      const fptype_v* rndmom = reinterpret_cast<const fptype_v*>(&( m_rndmom.data()[ievt * nparf * np4] ));
      fptype_v* momenta = reinterpret_cast<fptype_v*>(&( m_momenta.data()[ievt * npar * np4] ));

      // generate n massless momenta in infinite phase space
      fptype_v q[nparf][np4];
      for( int iparf = 0; iparf < nparf; iparf++ ) {
        const fptype_v r1 = rndmom[(iparf * 4    ) * 8 / neppV];
        const fptype_v r2 = rndmom[(iparf * 4 + 1) * 8 / neppV];
        const fptype_v r3 = rndmom[(iparf * 4 + 2) * 8 / neppV];
        const fptype_v r4 = rndmom[(iparf * 4 + 3) * 8 / neppV];
        const fptype_v c = 2. * r1 - 1.;
        const fptype_v s = sqrt_v( 1. - c * c );
        const fptype_v f = twopi * r2;
        q[iparf][0] = -log_v( r3 * r4 );
        q[iparf][3] = q[iparf][0] * c;
        auto [ss, cc] = sincos_v(f);
        q[iparf][2] = q[iparf][0] * s * cc;
        q[iparf][1] = q[iparf][0] * s * ss;
      }

      // calculate the parameters of the conformal transformation
      fptype_v r[np4] = {q[0][0] + q[1][0],
                         q[0][1] + q[1][1],
                         q[0][2] + q[1][2],
                         q[0][3] + q[1][3] };
      fptype_v b[np4 - 1];
      const fptype_v rmas = rsqrt_v( r[0]*r[0] - r[3]*r[3] - r[2]*r[2] - r[1]*r[1] );
      for( int i4 = 1; i4 < np4; i4++ ) b[i4 - 1] = -r[i4] * rmas;
      const fptype_v g = r[0] * rmas;
      const fptype_v a = 1. / ( 1. + g );
      const fptype_v x0 = m_energy * rmas;

      // transform the q's conformally into the p's (i.e. the 'momenta')
      for( int iparf = 0; iparf < nparf; iparf++ ) {
        fptype_v bq = b[0] * q[iparf][1] + b[1] * q[iparf][2] + b[2] * q[iparf][3];
        for( int i4 = 1; i4 < np4; i4++ ) {
          momenta[(iparf + npari) * np4 + i4] = x0 * ( q[iparf][i4] + b[i4 - 1] * ( q[iparf][0] + a * bq ) );
        }
        momenta[(iparf + npari) * np4] = x0 * ( g * q[iparf][0] + bq );
      }

    }

    // probably auto vectorized
    const fptype po2log = std::log( twopi / 4. );
    for( size_t n = 0; n < nevt(); n++ ) {
      m_weights.data()[n] = po2log;
    }
  }
}
