/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Management of temporary device storage needed for maintaining partial
 * reductions between subsequent grids
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace util {

/**
 * Manages device storage needed for communicating partial reductions
 * between CTAs in subsequent grids
 */
struct Spine
{
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Device spine storage
	void *d_spine;

	// Host-mapped spine storage (if so constructed)
	void *h_spine;

	// Number of bytes backed by d_spine
	size_t spine_bytes;

	// GPU d_spine was allocated on
	int gpu;

	// Whether or not the spine is (a) allocated on the host and mapped
	// into gpu memory, or (b) allocated on the gpu
	bool host_mapped;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor (device-allocated spine)
	 */
	Spine() :
		d_spine(NULL),
		h_spine(NULL),
		spine_bytes(0),
		gpu(B40C_INVALID_DEVICE),
		host_mapped(false) {}


	/**
	 * Constructor
	 *
	 * @param host_mapped
	 * 		Whether or not the spine is (a) allocated on the host and mapped
	 * 		into gpu memory, or (b) allocated on the gpu
	 */
	Spine(bool host_mapped) :
		d_spine(NULL),
		h_spine(NULL),
		spine_bytes(0),
		gpu(B40C_INVALID_DEVICE),
		host_mapped(host_mapped) {}


	/**
	 * Deallocates and resets the spine
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		do {

			if (gpu == B40C_INVALID_DEVICE) return retval;

			// Save current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Spine cudaGetDevice failed: ", __FILE__, __LINE__)) break;
#if CUDA_VERSION >= 4000
			if (retval = util::B40CPerror(cudaSetDevice(gpu),
				"Spine cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif
			if (host_mapped) {
				if (h_spine) {
					// Deallocate
					if (retval = util::B40CPerror(cudaFreeHost((void *) h_spine),
						"Spine cudaFreeHost h_spine failed", __FILE__, __LINE__)) break;

					h_spine = NULL;
				}
			} else if (d_spine) {
				// Deallocate
				if (retval = util::B40CPerror(cudaFree(d_spine),
					"Spine cudaFree d_spine failed: ", __FILE__, __LINE__)) break;
			}
#if CUDA_VERSION >= 4000
			// Restore current gpu
			if (retval = util::B40CPerror(cudaSetDevice(current_gpu),
				"Spine cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif
			d_spine 		= NULL;
			gpu 			= B40C_INVALID_DEVICE;
			spine_bytes	 	= 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~Spine()
	{
		HostReset();
	}


	/**
	 * Device spine storage accessor
	 */
	void* operator()()
	{
		return d_spine;
	}


	/**
	 * Sets up the spine to accommodate partials of the specified type
	 * produced/consumed by grids of the specified sweep grid size (lazily
	 * allocating it if necessary)
	 *
	 * Grows as necessary.
	 */
	template <typename T>
	cudaError_t Setup(int spine_elements)
	{
		cudaError_t retval = cudaSuccess;
		do {
			size_t problem_spine_bytes = spine_elements * sizeof(T);

			// Get current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Spine cudaGetDevice failed: ", __FILE__, __LINE__)) break;

			// Check if big enough and if lives on proper GPU
			if ((problem_spine_bytes > spine_bytes) || (gpu != current_gpu)) {

				// Deallocate if exists
				if (retval = HostReset()) break;

				// Remember device
				gpu = current_gpu;

				// Reallocate
				spine_bytes = problem_spine_bytes;
				if (host_mapped) {

					// Allocate pinned memory for h_spine
					int flags = cudaHostAllocMapped;
					if (retval = util::B40CPerror(cudaHostAlloc((void **)&h_spine, problem_spine_bytes, flags),
						"Spine cudaHostAlloc h_spine failed", __FILE__, __LINE__)) break;

					// Map done into GPU space
					if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_spine, (void *) h_spine, 0),
						"Spine cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

				} else {

					// Allocate on device
					if (retval = util::B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
						"Spine cudaMalloc d_spine failed", __FILE__, __LINE__)) break;
				}
			}
		} while (0);

		return retval;
	}
};

} // namespace util
} // namespace b40c

