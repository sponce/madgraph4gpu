// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2021-2023) for the MG5aMC CUDACPP plugin.
#pragma once

#include "mgOnGpuConfig.h"
#include "MemoryBuffers.h"

namespace mg5amcCpu {

  // A class encapsulating RAMBO phase space sampling on a CPU host
  struct RamboSamplingKernelHost final : NumberOfEvents {
    // Constructor from existing input and output buffers
    RamboSamplingKernelHost( const fptype energy,               // input: energy
                             const BufferRndNumMomenta& rndmom, // input: random numbers in [0,1]
                             BufferMomenta& momenta,            // output: momenta
                             BufferWeights& weights,            // output: weights
                             const size_t nevt );
    // Get momenta of initial state particles
    void getMomentaInitial();
    // Get momenta of final state particles and weights
    void getMomentaFinal();
    
    // The energy
    const fptype m_energy;
    // The buffer for the input random numbers
    alignas(32) const BufferRndNumMomenta& m_rndmom;
    // The buffer for the output momenta
    alignas(32) BufferMomenta& m_momenta;
    // The buffer for the output weights
    alignas(32) BufferWeights& m_weights;
  };

}
