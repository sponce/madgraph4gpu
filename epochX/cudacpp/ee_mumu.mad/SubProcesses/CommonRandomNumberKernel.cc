// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2021-2023) for the MG5aMC CUDACPP plugin.

#include "CommonRandomNumbers.h"
#include "MemoryBuffers.h"
#include "RandomNumberKernels.h"

#include <cassert>

namespace mg5amcCpu {

  CommonRandomNumberKernel::CommonRandomNumberKernel( BufferRndNumMomenta& rnarray )
    : m_rnarray( rnarray ) , m_seed( 20211220 ) {
    if( m_rnarray.isOnDevice() )
      throw std::runtime_error( "CommonRandomNumberKernel on host with a device random number array" );
  }

  void CommonRandomNumberKernel::generateRnarray() {
    // NB: generate as double (HARDCODED)
    std::vector<double> rnd = CommonRandomNumbers::generate<double>( m_rnarray.size(), m_seed );
    // NB: copy may imply a double-to-float conversion
    std::copy( rnd.begin(), rnd.end(), m_rnarray.data() );
  }

}
