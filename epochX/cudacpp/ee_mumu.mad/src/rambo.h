// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, A. Valassi (2020-2023) for the MG5aMC CUDACPP plugin.
//==========================================================================

#include "mgOnGpuConfig.h"

#include "mgOnGpuFptypes.h"

#include "CPPProcess.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

// Simplified rambo version for 2 to N (with N>=2) processes with massless particles
namespace mg5amcCpu {
  constexpr int np4 = CPPProcess::np4;     // dimensions of 4-momenta (E,px,py,pz)
  constexpr int npari = CPPProcess::npari; // #particles in the initial state (incoming): e.g. 2 (e+ e-) for e+ e- -> mu+ mu-
  constexpr int nparf = CPPProcess::nparf; // #particles in the final state (outgoing): e.g. 2 (mu+ mu-) for e+ e- -> mu+ mu-
  constexpr int npar = CPPProcess::npar;   // #particles in total (external = initial + final): e.g. 4 for e+ e- -> mu+ mu-

  inline __attribute__( ( always_inline ) ) fptype& kernelAccessIp4Ipar( fptype* buffer,
                                                                        const int ip4,
                                                                        const int ipar ) {
    return buffer[ipar * np4 * neppV + ip4 * neppV]; // AOSOA[0][ipar][ip4][0]
  }

  // Fill in the momenta of the initial particles
  // [NB: the output buffer includes both initial and final momenta, but only initial momenta are filled in]
  inline void __attribute__( ( always_inline ) )
  ramboGetMomentaInitial( const fptype energy, // input: energy
                          fptype* momenta ) {  // output: momenta for one event or for a set of events
    const fptype mom = energy / 2;
    momenta[0] = mom;
    momenta[neppV] = 0;
    momenta[2 * neppV] = 0;
    momenta[3 * neppV] = mom;
    momenta[4 * neppV] = mom;
    momenta[5 * neppV] = 0;
    momenta[6 * neppV] = 0;
    momenta[7 * neppV] = -mom;
  }
}
