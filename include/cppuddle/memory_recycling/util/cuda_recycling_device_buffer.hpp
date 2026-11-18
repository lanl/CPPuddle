// Copyright (c) 2020-2024 Gregor Daiß
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_RECYCLING_BUFFER_HPP
#define CUDA_RECYCLING_BUFFER_HPP

// import recycle_allocator_cuda_device
#include "cppuddle/memory_recycling/cuda_recycling_allocators.hpp""

/// \file
/// Contains a RAII wrappers for CUDA device buffers. Intended to be used with
/// the recycling allocators but technically any allocator should work

namespace cppuddle {
namespace memory_recycling {

/// Convienience struct to delete the device buffer once the reference counting
/// reaches 0
template<typename T, typename alloc_type>
struct device_buffer_deleter_deleter {
  alloc_type allocator;
  size_t number_of_elements;
  device_buffer_deleter_deleter(alloc_type &alloc, size_t number_of_elements) : allocator(alloc),
    number_of_elements(number_of_elements) {}
  void operator()(T* p) {
    allocator.deallocate(p, number_of_elements);
  }
};

/// RAII wrapper for CUDA device memory
/// (ideally used with a recycling allocator)
template <typename T, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct cuda_device_buffer {
  recycle_allocator_cuda_device<T> allocator;
  T *device_side_buffer;
  std::shared_ptr<T> data_ref_counter;
  size_t number_of_elements;

  cuda_device_buffer(const size_t number_of_elements, const size_t device_id = 0) 
      : allocator{device_id}, number_of_elements(number_of_elements), device_side_buffer(allocator.allocate(number_of_elements)),
        data_ref_counter(device_side_buffer, device_buffer_deleter_deleter<T, recycle_allocator_cuda_device<T>>(allocator, number_of_elements)) {
    assert(device_id < max_number_gpus);
  }
  ~cuda_device_buffer() = default;
  explicit cuda_device_buffer(cuda_device_buffer const &other) = default;
  cuda_device_buffer& operator=(cuda_device_buffer const &other) = default;
  explicit cuda_device_buffer(cuda_device_buffer &&other) = default;
  cuda_device_buffer& operator=(cuda_device_buffer &&other) = default;

};

/// RAII wrapper for CUDA device memory using a passed aggregated allocator
/// (which ideally should be an allocator_slice from the work aggregation)
template <typename T, typename Host_Allocator, std::enable_if_t<std::is_trivial<T>::value, int> = 0>
struct cuda_aggregated_device_buffer {
  T *device_side_buffer;
  std::shared_ptr<T> data_ref_counter;
  size_t number_of_elements;
  cuda_aggregated_device_buffer(size_t number_of_elements, Host_Allocator &alloc)
      : number_of_elements(number_of_elements), alloc(alloc), device_side_buffer(alloc.allocate(number_of_elements)),  
        data_ref_counter(device_side_buffer, device_buffer_deleter_deleter<T, Host_Allocator>(alloc, number_of_elements)) {
  }
  ~cuda_aggregated_device_buffer() = default;
  // not yet implemented
  explicit cuda_aggregated_device_buffer(cuda_aggregated_device_buffer const &other) = default;
  cuda_aggregated_device_buffer& operator=(cuda_aggregated_device_buffer const &other) = default;
  explicit cuda_aggregated_device_buffer(cuda_aggregated_device_buffer &&other) = default;
  cuda_aggregated_device_buffer& operator=(cuda_aggregated_device_buffer &&other) = default;

private:
  Host_Allocator &alloc; // will stay valid for the entire aggregation region and hence
                         // for the entire lifetime of this buffer
};

} // namespace memory_recycling
} // end namespace cppuddle
#endif
