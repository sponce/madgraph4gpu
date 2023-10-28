// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2021-2023) for the MG5aMC CUDACPP plugin.

#ifndef MemoryAccessMomenta_H
#define MemoryAccessMomenta_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "MemoryAccessHelpers.h"
#include "MemoryAccessVectors.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
namespace mg5amcCpu {

  // A class describing the internal layout of memory buffers for momenta
  // This implementation uses an AOSOA[npagM][npar][np4][neppV] where nevt=npagM*neppV
  // [If many implementations are used, a suffix _AOSOAv1 should be appended to the class name]
  class MemoryAccessMomentaBase { //_AOSOAv1 {

    friend class MemoryAccessHelper<MemoryAccessMomentaBase>;
    friend class KernelAccessHelper<MemoryAccessMomentaBase, true>;
    friend class KernelAccessHelper<MemoryAccessMomentaBase, false>;

    // The number of components of a 4-momentum
    static constexpr int np4 = CPPProcess::np4;

    // The number of particles in this physics process
    static constexpr int npar = CPPProcess::npar;

    //--------------------------------------------------------------------------
    // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
    // (in other words: first locate the event record for a given event, then locate an element in that record)
    //--------------------------------------------------------------------------

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
    static __host__ __device__ inline fptype*
    ieventAccessRecord( fptype* buffer, const int ievt ) {
      const int ipagM = ievt / neppV; // #event "M-page"
      const int ieppM = ievt % neppV; // #event in the current event M-page
      return &( buffer[ipagM * npar * np4 * neppV + ieppM] ); // AOSOA[ipagM][0][0][ieppM]
    }

    //--------------------------------------------------------------------------

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, Ts... args ) <===]
    // [NB: expand variadic template "Ts... args" to "const int ip4, const int ipar" and rename "Field" as "Ip4Ipar"]
    static __host__ __device__ inline fptype&
    decodeRecord( fptype* buffer,
                  const int ip4,
                  const int ipar ) {
      return buffer[ipar * np4 * neppV + ip4 * neppV]; // AOSOA[0][ipar][ip4][0]
    }
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on explicit event numbers
  // Its methods use the MemoryAccessHelper templates - note the use of the template keyword in template function instantiations
  class MemoryAccessMomenta : public MemoryAccessMomentaBase
  {
  public:

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessMomentaBase>::ieventAccessRecord;

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (const) ===> const fptype* ieventAccessRecordConst( const fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessMomentaBase>::ieventAccessRecordConst;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto decodeRecordIp4Ipar = MemoryAccessHelper<MemoryAccessMomentaBase>::decodeRecord;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (const) ===> const fptype& decodeRecordConst( const fptype* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto decodeRecordIp4IparConst =
      MemoryAccessHelper<MemoryAccessMomentaBase>::template decodeRecordConst<int, int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (non-const) ===> fptype& ieventAccessIp4Ipar( fptype* buffer, const ievt, const int ipar, const int ipar ) <===]
    static constexpr auto ieventAccessIp4Ipar =
      MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessField<int, int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype& ieventAccessIp4IparConst( const fptype* buffer, const ievt, const int ipar, const int ipar ) <===]
    // DEFAULT VERSION
    static constexpr auto ieventAccessIp4IparConst =
      MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessFieldConst<int, int>;
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  // Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
  template<bool onDevice>
  class KernelAccessMomenta
  {
  public:

    // Expose selected functions from MemoryAccessMomenta
    static constexpr auto ieventAccessRecordConst = MemoryAccessMomenta::ieventAccessRecordConst;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (non-const, SCALAR) ===> fptype& kernelAccessIp4Ipar( fptype* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto kernelAccessIp4Ipar =
      KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessField<int, int>;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR) ===> const fptype& kernelAccessIp4IparConst( const fptype* buffer, const int ipar, const int ipar ) <===]
    // DEFAULT VERSION
    static constexpr auto kernelAccessIp4IparConst_s =
      KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessFieldConst<int, int>;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR OR VECTOR) ===> fptype_sv kernelAccessIp4IparConst( const fptype* buffer, const int ipar, const int ipar ) <===]
    // FIXME? Eventually return by const reference and support aligned arrays only?
    // FIXME? Currently return by value to support also unaligned and arbitrary arrays
    static __host__ __device__ inline fptype_sv
    kernelAccessIp4IparConst( const fptype* buffer,
                              const int ip4,
                              const int ipar ) {
      const fptype& out = buffer[ipar * CPPProcess::np4 * neppV + ip4 * neppV];
      if( (size_t)( buffer ) % mgOnGpu::cppAlign == 0 ) {
        return mg5amcCpu::fptypevFromAlignedArray( out );
      } else {
        return mg5amcCpu::fptypevFromUnalignedArray( out );
      }
    }

    // Is this a HostAccess or DeviceAccess class?
    // [this is only needed for a warning printout in rambo.h for nparf==1 #358]
    static __host__ __device__ inline constexpr bool
    isOnDevice()
    {
      return onDevice;
    }
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessMomenta<false> HostAccessMomenta;
  typedef KernelAccessMomenta<true> DeviceAccessMomenta;

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessMomenta_H
