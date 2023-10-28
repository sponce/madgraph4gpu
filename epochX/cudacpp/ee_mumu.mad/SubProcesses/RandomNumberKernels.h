// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2021-2023) for the MG5aMC CUDACPP plugin.

#pragma once

#include "mgOnGpuConfig.h"
#include "MemoryBuffers.h"

namespace mg5amcCpu {

  // A class encapsulating common random number generation on a CPU host
  class CommonRandomNumberKernel final {
  public:
    // Constructor from an existing output buffer
    CommonRandomNumberKernel( BufferRndNumMomenta& rnarray );
    // Seed the random number generator
    void seedGenerator( const unsigned int seed ) { m_seed = seed; };
    // Generate the random number array
    void generateRnarray();

  private:
    // The buffer for the output random numbers
    BufferRndNumMomenta& m_rnarray;
    // The generator seed
    unsigned int m_seed;
  };
}
