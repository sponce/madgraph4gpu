#ifndef MemoryAccessMomenta_H
#define MemoryAccessMomenta_H 1

#include "mgOnGpuConfig.h"
#include "mgOnGpuTypes.h"
#include "mgOnGpuVectors.h"

#include "MemoryAccessHelpers.h"
#include "MemoryAccessVectors.h"

//----------------------------------------------------------------------------

// A class describing the internal layout of memory buffers for momenta
// This implementation uses an AOSOA[npagM][npar][np4][neppM] where nevt=npagM*neppM
// [If many implementations are used, a suffix _AOSOAv1 should be appended to the class name]
class MemoryAccessMomentaBase//_AOSOAv1
{
public:

  // Number of Events Per Page in the momenta AOSOA memory layout
  // (these are all best kept as a compile-time constants: see issue #23)
#ifdef __CUDACC__
  // -----------------------------------------------------------------------------------------------
  // --- GPUs: neppM is best set to a power of 2 times the number of fptype's in a 32-byte cacheline
  // --- This is relevant to ensure coalesced access to momenta in global memory
  // --- Note that neppR is hardcoded and may differ from neppM and neppV on some platforms
  // -----------------------------------------------------------------------------------------------
  //static constexpr int neppM = 64/sizeof(fptype); // 2x 32-byte GPU cache lines (512 bits): 8 (DOUBLE) or 16 (FLOAT)
  static constexpr int neppM = 32/sizeof(fptype); // (DEFAULT) 32-byte GPU cache line (256 bits): 4 (DOUBLE) or 8 (FLOAT)
  //static constexpr int neppM = 1;  // *** NB: this is equivalent to AOS ***
#else
  // -----------------------------------------------------------------------------------------------
  // --- CPUs: neppM is best set equal to the number of fptype's (neppV) in a vector register
  // --- This is relevant to ensure faster access to momenta from C++ memory cache lines
  // --- However, neppM is now decoupled from neppV (issue #176) and can be separately hardcoded
  // --- In practice, neppR, neppM and neppV could now (in principle) all be different
  // -----------------------------------------------------------------------------------------------
#ifdef MGONGPU_CPPSIMD
  static constexpr int neppM = MGONGPU_CPPSIMD; // (DEFAULT) neppM=neppV for optimal performance
  //static constexpr int neppM = 64/sizeof(fptype); // maximum CPU vector width (512 bits): 8 (DOUBLE) or 16 (FLOAT)
  //static constexpr int neppM = 32/sizeof(fptype); // lower CPU vector width (256 bits): 4 (DOUBLE) or 8 (FLOAT)
  //static constexpr int neppM = 1; // *** NB: this is equivalent to AOS ***
  //static constexpr int neppM = MGONGPU_CPPSIMD*2; // FOR TESTS
#else
  static constexpr int neppM = 1; // (DEFAULT) neppM=neppV for optimal performance (NB: this is equivalent to AOS)
#endif
#endif

  // SANITY CHECK: check that neppM is a power of two
  static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );
  
private:

  friend class MemoryAccessHelper<MemoryAccessMomentaBase>;
  friend class KernelAccessHelper<MemoryAccessMomentaBase, true>;
  friend class KernelAccessHelper<MemoryAccessMomentaBase, false>;
  
  // The number of components of a 4-momentum
  static constexpr int np4 = mgOnGpu::np4;

  // The number of particles in this physics process
  static constexpr int npar = mgOnGpu::npar;

  //--------------------------------------------------------------------------
  // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
  // (in other words: first locate the event record for a given event, then locate an element in that record)
  //--------------------------------------------------------------------------

  // Locate an event record (output) in a memory buffer (input) from an explicit event number (input)
  // (Non-const memory access to event record from ievent)
  static
  __host__ __device__ inline
  fptype* ieventAccessRecord( fptype* buffer,
                              const int ievt )
  {
    constexpr int ip4 = 0;
    constexpr int ipar = 0;
    const int ipagM = ievt/neppM; // #event "M-page"
    const int ieppM = ievt%neppM; // #event in the current event M-page
    return &( buffer[ipagM*npar*np4*neppM + ipar*np4*neppM + ip4*neppM + ieppM] ); // AOSOA[ipagM][ipar][ip4][ieppM]
  }

  //--------------------------------------------------------------------------

  // Locate a field (output) of an event record (input) from the given field indexes (input)
  // (Non-const memory access to field in an event record)
  static
  __host__ __device__ inline
  fptype& decodeRecord( fptype* buffer,
                        const int ip4,
                        const int ipar )
  {
    constexpr int ipagM = 0;
    constexpr int ieppM = 0;
    return buffer[ipagM*npar*np4*neppM + ipar*np4*neppM + ip4*neppM + ieppM]; // AOSOA[ipagM][ipar][ip4][ieppM]
  }

};

//----------------------------------------------------------------------------

// A class providing access to memory buffers for a given event, based on explicit event numbers
class MemoryAccessMomenta : public MemoryAccessMomentaBase
{
public:

  // (Non-const memory access to event record from ievent)
  static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessMomentaBase>::ieventAccessRecord;

  // (Const memory access to event record from ievent)
  static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessMomentaBase>::ieventAccessRecordConst;

  // (Non-const memory access to field in an event record)
  static constexpr auto decodeRecordIp4Ipar = MemoryAccessHelper<MemoryAccessMomentaBase>::decodeRecord;

  // [NOTE THE USE OF THE TEMPLATE KEYWORD IN ALL OF THE FOLLOWING TEMPLATE FUNCTION INSTANTIATIONS]
  // (Const memory access to field in an event record)
  static constexpr auto decodeRecordIp4IparConst =
    MemoryAccessHelper<MemoryAccessMomentaBase>::template decodeRecordConst<int, int>;

  // (Non-const memory access to field from ievent)
  static constexpr auto ieventAccessIp4Ipar =
    MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessField<int, int>;

  // (Const memory access to field from ievent)
  static constexpr auto ieventAccessIp4IparConst =
    MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessFieldConst<int, int>;
  /*
  // (Const memory access to field from ievent - DEBUG version with printouts)
  static
  __host__ __device__ inline
  const fptype& ieventAccessIp4IparConst( const fptype* buffer,
                                          const int ievt,
                                          const int ip4,
                                          const int ipar )
  {
    const fptype& out = MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessFieldConst<int, int>( buffer, ievt, ip4, ipar );
    printf( "ipar=%2d ip4=%2d ievt=%8d out=%8.3f\n", ipar, ip4, ievt, out );
    return out;
  }
  */

};

//----------------------------------------------------------------------------

// A class providing access to memory buffers for a given event, based on implicit kernel rules
template<bool onDevice>
class KernelAccessMomenta
{
public:

  // (Non-const memory access to field from kernel)
  static constexpr auto kernelAccessIp4Ipar =
    KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessField<int, int>;

  // (Const memory access to field from kernel, scalar)
  static constexpr auto kernelAccessIp4IparConst_s =
    KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessFieldConst<int, int>;
  /*
  // (Const memory access to field from kernel, scalar - DEBUG version with printouts)
  static
  __host__ __device__ inline
  const fptype& kernelAccessIp4IparConst_s( const fptype* buffer,
                                            const int ip4,
                                            const int ipar )
  {
    const fptype& out = KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessFieldConst<int, int>( buffer, ip4, ipar );
    printf( "ipar=%2d ip4=%2d ievt=  kernel out=%8.3f\n", ipar, ip4, out );
    return out;
  }
  */

  // (Const memory access to field from kernel, scalar or vector)
  // [FIXME? Eventually return by reference and support aligned arrays only?]
  // [Currently return by value to support also unaligned and arbitrary arrays]
  static
  __host__ __device__ inline
  fptype_sv kernelAccessIp4IparConst( const fptype* buffer,
                                      const int ip4,
                                      const int ipar )
  {
    const fptype& out = kernelAccessIp4IparConst_s( buffer, ip4, ipar );
#ifndef MGONGPU_CPPSIMD
    return out;
#else
    constexpr int neppM = MemoryAccessMomentaBase::neppM;
    constexpr bool useContiguousEventsIfPossible = true; // DEFAULT
    //constexpr bool useContiguousEventsIfPossible = false; // FOR PERFORMANCE TESTS (treat as arbitrary array even if it is an AOSOA)
    // Use c++17 "if constexpr": compile-time branching
    if constexpr ( useContiguousEventsIfPossible && ( neppM >= neppV ) && ( neppM%neppV == 0 ) )
    {
      //constexpr bool skipAlignmentCheck = true; // FASTEST (SEGFAULTS IF MISALIGNED ACCESS, NEEDS A SANITY CHECK ELSEWHERE!)
      constexpr bool skipAlignmentCheck = false; // NEW DEFAULT: A BIT SLOWER BUT SAFER [ALLOWS MISALIGNED ACCESS]
      if constexpr ( skipAlignmentCheck )
      {
        //static bool first=true; if( first ){ std::cout << "WARNING! assume aligned AOSOA, skip check" << std::endl; first=false; } // SLOWS DOWN...
        // Fastest (5.09E6 in eemumu 512y)
        // This assumes alignment for momenta1d without checking - causes segmentation fault in reinterpret_cast if not aligned!
        return mg5amcCpu::fptypevFromAlignedArray( out ); // use reinterpret_cast
      }
      else if ( (size_t)(buffer) % mgOnGpu::cppAlign == 0 )
      {
        //static bool first=true; if( first ){ std::cout << "WARNING! aligned AOSOA, reinterpret cast" << std::endl; first=false; } // SLOWS DOWN...
        // A tiny bit (<1%) slower because of the alignment check (5.07E6 in eemumu 512y)
        // This explicitly checks buffer alignment to avoid segmentation faults in reinterpret_cast
        return mg5amcCpu::fptypevFromAlignedArray( out ); // use reinterpret_cast
      }
      else
      {
        //static bool first=true; if( first ){ std::cout << "WARNING! AOSOA but no reinterpret cast" << std::endl; first=false; } // SLOWS DOWN...
        // A bit (1%) slower (5.05E6 in eemumu 512y)
        // This does not require buffer alignment, but it requires AOSOA with neppM>=neppV and neppM%neppV==0
        return mg5amcCpu::fptypevFromUnalignedArray( out ); // do not use reinterpret_cast
      }
    }
    else
    {
      //static bool first=true; if( first ){ std::cout << "WARNING! arbitrary array" << std::endl; first=false; } // SLOWS DOWN...
      // Much (7-12%) slower (4.30E6 for AOSOA, 4.53E6 for AOS in eemumu 512y)
      //... to do? implementation based on fptypevFromArbitraryArray ...
      std::cout << "ERROR! useContiguousEventsIfPossible=" << useContiguousEventsIfPossible
                << ", neppM=" << neppM << ", neppV=" << neppV << std::endl;
      throw std::logic_error( "MemoryAccessMomenta requires an AOSOA and does not support arbitrary arrays" ); // no path to this statement
    }
#endif
  }

};

//----------------------------------------------------------------------------

typedef KernelAccessMomenta<false> HostAccessMomenta;
typedef KernelAccessMomenta<true> DeviceAccessMomenta;

//----------------------------------------------------------------------------

#endif // MemoryAccessMomenta_H
