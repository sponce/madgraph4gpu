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

  void RamboSamplingKernelHost::getMomentaFinal() {
    for( size_t ievt = 0; ievt < nevt(); ievt+=neppV ) {
      for( size_t ieppM = 0; ieppM < neppV; ++ieppM ) {
        // AOSOA[ipagM][0][0][ieppM]
        const fptype* rndmom =  &( m_rndmom.data()[ievt * nparf * np4 + ieppM] );
        fptype* momenta = &( m_momenta.data()[ievt * npar * np4 + ieppM] );
        fptype& wt = m_weights.data()[ievt+ieppM];

        // initialization step: factorials for the phase space weight
        const fptype twopi = 8. * atan( 1. );
        const fptype po2log = log( twopi / 4. );
        fptype z[nparf];
        z[1] = po2log;
        for( int kpar = 2; kpar < nparf; kpar++ )
          z[kpar] = z[kpar - 1] + po2log - 2. * log( fptype( kpar - 1 ) );
        for( int kpar = 2; kpar < nparf; kpar++ )
          z[kpar] = ( z[kpar] - log( fptype( kpar ) ) );

        // generate n massless momenta in infinite phase space
        fptype q[nparf][np4];
        for( int iparf = 0; iparf < nparf; iparf++ ) {
          const fptype r1 = rndmom[(iparf * 4    ) * 8];
          const fptype r2 = rndmom[(iparf * 4 + 1) * 8];
          const fptype r3 = rndmom[(iparf * 4 + 2) * 8];
          const fptype r4 = rndmom[(iparf * 4 + 3) * 8];
          const fptype c = 2. * r1 - 1.;
          const fptype s = sqrt( 1. - c * c );
          const fptype f = twopi * r2;
          q[iparf][0] = -log( r3 * r4 );
          q[iparf][3] = q[iparf][0] * c;
          q[iparf][2] = q[iparf][0] * s * cos( f );
          q[iparf][1] = q[iparf][0] * s * sin( f );
        }

        // calculate the parameters of the conformal transformation
        fptype r[np4];
        for( int i4 = 0; i4 < np4; i4++ ) r[i4] = 0.;
        for( int iparf = 0; iparf < nparf; iparf++ ) {
          for( int i4 = 0; i4 < np4; i4++ ) r[i4] = r[i4] + q[iparf][i4];
        }
        fptype b[np4 - 1];
        const fptype rmas = sqrt( pow( r[0], 2 ) - pow( r[3], 2 ) - pow( r[2], 2 ) - pow( r[1], 2 ) );
        for( int i4 = 1; i4 < np4; i4++ ) b[i4 - 1] = -r[i4] / rmas;
        const fptype g = r[0] / rmas;
        const fptype a = 1. / ( 1. + g );
        const fptype x0 = m_energy / rmas;

        // transform the q's conformally into the p's (i.e. the 'momenta')
        for( int iparf = 0; iparf < nparf; iparf++ ) {
          fptype bq = b[0] * q[iparf][1] + b[1] * q[iparf][2] + b[2] * q[iparf][3];
          for( int i4 = 1; i4 < np4; i4++ ) {
            momenta[(iparf + npari) * np4 * neppV + i4 * neppV] = x0 * ( q[iparf][i4] + b[i4 - 1] * ( q[iparf][0] + a * bq ) );
          }
          momenta[(iparf + npari) * np4 * neppV] = x0 * ( g * q[iparf][0] + bq );
        }

        // calculate weight (NB return log of weight)
        wt = po2log;
        if( nparf != 2 ) wt = ( 2. * nparf - 4. ) * log( m_energy ) + z[nparf - 1];

        // issue warnings if weight is too small or too large
        static int iwarn[5] = { 0, 0, 0, 0, 0 };
        if( wt < -180. ) {
          if( iwarn[0] <= 5 ) std::cout << "Too small wt, risk for underflow: " << wt << std::endl;
          iwarn[0] = iwarn[0] + 1;
        }
        if( wt > 174. ) {
          if( iwarn[1] <= 5 ) std::cout << "Too large wt, risk for overflow: " << wt << std::endl;
          iwarn[1] = iwarn[1] + 1;
        }
      }
    }
  }
}
