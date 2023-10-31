// Copyright (C) 2020-2023 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: A. Valassi (Dec 2021, based on earlier work by S. Hageboeck) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Roiser, A. Valassi (2021-2023) for the MG5aMC CUDACPP plugin.

#pragma once

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"

#include "CPPProcess.h"
#include "CudaRuntime.h"
#include "Parameters_sm.h"

#include <sstream>

namespace mg5amcCpu {

  namespace MemoryBuffers
  {
    // Process-independent compile-time constants
    static constexpr size_t np4 = CPPProcess::np4;
    static constexpr size_t nw6 = CPPProcess::nw6;
    static constexpr size_t nx2 = mgOnGpu::nx2;
    // Process-dependent compile-time constants
    static constexpr size_t nparf = CPPProcess::nparf;
    static constexpr size_t npar = CPPProcess::npar;
    static constexpr size_t ndcoup = Parameters_sm_dependentCouplings::ndcoup;
  }

  //--------------------------------------------------------------------------

  // An abstract interface encapsulating a given number of events
  class INumberOfEvents
  {
  public:
    virtual ~INumberOfEvents() {}
    virtual size_t nevt() const = 0;
  };

  //--------------------------------------------------------------------------

  // A class encapsulating a given number of events
  class NumberOfEvents : virtual public INumberOfEvents
  {
  public:
    NumberOfEvents( const size_t nevt )
      : m_nevt( nevt ) {}
    virtual ~NumberOfEvents() {}
    virtual size_t nevt() const override { return m_nevt; }
  private:
    const size_t m_nevt;
  };

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer (not necessarily an event buffer)
  template<typename T>
  class BufferBase : virtual public INumberOfEvents
  {
  public:
    BufferBase( const size_t size )
      : m_size( size ), m_data( nullptr ) {}
    virtual ~BufferBase() {}
  public:
    T* data() { return m_data; }
    const T* data() const { return m_data; }
    T& operator[]( const size_t index ) { return m_data[index]; }
    const T& operator[]( const size_t index ) const { return m_data[index]; }
    size_t size() const { return m_size; }
    size_t bytes() const { return m_size * sizeof( T ); }
    bool isOnDevice() const { return false; }
    virtual size_t nevt() const override { throw std::runtime_error( "This BufferBase is not an event buffer" ); }
  protected:
    const size_t m_size;
    T* m_data;
  };

  //--------------------------------------------------------------------------

  constexpr bool HostBufferALIGNED = false;   // ismisaligned=false
  constexpr bool HostBufferMISALIGNED = true; // ismisaligned=true

  // A class encapsulating a C++ host buffer
  template<typename T>
  class HostBufferBase : public BufferBase<T>
  {
  public:
    HostBufferBase( const size_t size )
      : BufferBase<T>( size ) {
        this->m_data = new( std::align_val_t( cppAlign ) ) T[size]();
    }
    virtual ~HostBufferBase() {
        ::operator delete[]( this->m_data, std::align_val_t( cppAlign ) );
    }
    static constexpr bool isaligned() { return true; }
  public:
    static constexpr size_t cppAlign = mgOnGpu::cppAlign;
  };

  // A class encapsulating a C++ host buffer for a given number of events
  template<typename T, size_t sizePerEvent>
  class HostBuffer : public HostBufferBase<T>, virtual private NumberOfEvents
  {
  public:
    HostBuffer( const size_t nevt )
      : NumberOfEvents( nevt )
      , HostBufferBase<T>( sizePerEvent * nevt ) {}
    virtual ~HostBuffer() {}
    virtual size_t nevt() const override final { return NumberOfEvents::nevt(); }
  };

  // A base class encapsulating a memory buffer for momenta random numbers
  typedef BufferBase<fptype> BufferRndNumMomenta;

  // The size (number of elements) per event in a memory buffer for momenta random numbers
  constexpr size_t sizePerEventRndNumMomenta = MemoryBuffers::np4 * MemoryBuffers::nparf;

  // A class encapsulating a C++ host buffer for momenta random numbers
  typedef HostBuffer<fptype, sizePerEventRndNumMomenta> HostBufferRndNumMomenta;

  // A base class encapsulating a memory buffer for Gs (related to the event-by-event strength of running coupling constant alphas QCD)
  typedef BufferBase<fptype> BufferGs;

  // The size (number of elements) per event in a memory buffer for Gs
  constexpr size_t sizePerEventGs = 1;

  // A class encapsulating a C++ host buffer for gs
  typedef HostBuffer<fptype, sizePerEventGs> HostBufferGs;

#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
  // A base class encapsulating a memory buffer for numerators (of the multichannel single-diagram enhancement factors)
  typedef BufferBase<fptype> BufferNumerators;

  // The size (number of elements) per event in a memory buffer for numerators
  constexpr size_t sizePerEventNumerators = 1;

  // A class encapsulating a C++ host buffer for gs
  typedef HostBuffer<fptype, sizePerEventNumerators> HostBufferNumerators;
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPU_SUPPORTS_MULTICHANNEL
  // A base class encapsulating a memory buffer for denominators (of the multichannel single-diagram enhancement factors)
  typedef BufferBase<fptype> BufferDenominators;

  // The size (number of elements) per event in a memory buffer for denominators
  constexpr size_t sizePerEventDenominators = 1;

  // A class encapsulating a C++ host buffer for gs
  typedef HostBuffer<fptype, sizePerEventDenominators> HostBufferDenominators;
#endif

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for couplings that depend on the event-by-event running coupling constant alphas QCD
  typedef BufferBase<fptype> BufferCouplings;

  // The size (number of elements) per event in a memory buffer for random numbers
  constexpr size_t sizePerEventCouplings = MemoryBuffers::ndcoup * MemoryBuffers::nx2;

  // A class encapsulating a C++ host buffer for gs
  typedef HostBuffer<fptype, sizePerEventCouplings> HostBufferCouplings;

  // A base class encapsulating a memory buffer for momenta
  typedef BufferBase<fptype> BufferMomenta;

  // The size (number of elements) per event in a memory buffer for momenta
  constexpr size_t sizePerEventMomenta = MemoryBuffers::np4 * MemoryBuffers::npar;

  // A class encapsulating a C++ host buffer for momenta
  typedef HostBuffer<fptype, sizePerEventMomenta> HostBufferMomenta;

  // A base class encapsulating a memory buffer for sampling weights
  typedef BufferBase<fptype> BufferWeights;

  // The size (number of elements) per event in a memory buffer for sampling weights
  constexpr size_t sizePerEventWeights = 1;

  // A class encapsulating a C++ host buffer for sampling weights
  typedef HostBuffer<fptype, sizePerEventWeights> HostBufferWeights;

  // A base class encapsulating a memory buffer for matrix elements
  typedef BufferBase<fptype> BufferMatrixElements;

  // The size (number of elements) per event in a memory buffer for matrix elements
  constexpr size_t sizePerEventMatrixElements = 1;

  // A class encapsulating a C++ host buffer for matrix elements
  typedef HostBuffer<fptype, sizePerEventMatrixElements> HostBufferMatrixElements;

  // A base class encapsulating a memory buffer for the helicity mask
  typedef BufferBase<bool> BufferHelicityMask;

  // A class encapsulating a C++ host buffer for the helicity mask
  typedef HostBufferBase<bool> HostBufferHelicityMask;

  // A base class encapsulating a memory buffer for wavefunctions
  typedef BufferBase<fptype> BufferWavefunctions;

  // The size (number of elements) per event in a memory buffer for wavefunctions
  constexpr size_t sizePerEventWavefunctions = MemoryBuffers::nw6 * MemoryBuffers::nx2;

  // A class encapsulating a C++ host buffer for wavefunctions
  typedef HostBuffer<fptype, sizePerEventWavefunctions> HostBufferWavefunctions;

  // A base class encapsulating a memory buffer for helicity random numbers
  typedef BufferBase<fptype> BufferRndNumHelicity;

  // The size (number of elements) per event in a memory buffer for helicity random numbers
  constexpr size_t sizePerEventRndNumHelicity = 1;

  // A class encapsulating a C++ host buffer for helicity random numbers
  typedef HostBuffer<fptype, sizePerEventRndNumHelicity> HostBufferRndNumHelicity;

  // A base class encapsulating a memory buffer for color random numbers
  typedef BufferBase<fptype> BufferRndNumColor;

  // The size (number of elements) per event in a memory buffer for color random numbers
  constexpr size_t sizePerEventRndNumColor = 1;

  // A class encapsulating a C++ host buffer for color random numbers
  typedef HostBuffer<fptype, sizePerEventRndNumColor> HostBufferRndNumColor;

  // A base class encapsulating a memory buffer for helicity selection
  typedef BufferBase<int> BufferSelectedHelicity;

  // The size (number of elements) per event in a memory buffer for helicity selection
  constexpr size_t sizePerEventSelectedHelicity = 1;

  // A class encapsulating a C++ host buffer for helicity selection
  typedef HostBuffer<int, sizePerEventSelectedHelicity> HostBufferSelectedHelicity;

  // A base class encapsulating a memory buffer for color selection
  typedef BufferBase<int> BufferSelectedColor;

  // The size (number of elements) per event in a memory buffer for color selection
  constexpr size_t sizePerEventSelectedColor = 1;

  // A class encapsulating a C++ host buffer for color selection
  typedef HostBuffer<int, sizePerEventSelectedColor> HostBufferSelectedColor;
}
