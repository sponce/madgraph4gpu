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
    for( size_t ievt = 0; ievt < nevt(); ++ievt ) {
      // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
      const int ipagM = ievt / neppV; // #event "M-page"
      const int ieppM = ievt % neppV; // #event in the current event M-page
      const fptype* ievtRndmom =  &( m_rndmom.data()[ipagM * nparf * np4 * neppV + ieppM] ); // AOSOA[ipagR][0][0][ieppR]
      fptype* ievtMomenta = &( m_momenta.data()[ipagM * npar * np4 * neppV + ieppM] ); // AOSOA[ipagM][0][0][ieppM]      
      fptype* ievtWeights = &( m_weights.data()[ievt] );
      ramboGetMomentaFinal( m_energy, ievtRndmom, ievtMomenta, ievtWeights );
    }
  }

}
