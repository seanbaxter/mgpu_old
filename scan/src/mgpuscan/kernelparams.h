// kernelparams.h is included by both the kernel and host code. Localizing the
// kernel size parameters here lets us modify, rebuild, and benchmark in quick
// succession to find optimal performance configurations.

#pragma once

#define REDUCTION_NUM_THREADS 256

#define SCAN_NUM_THREADS 1024
#define SCAN_VALUES_PER_THREAD 4
#define SCAN_BLOCKS_PER_SM 1

#define PACKED_NUM_THREADS 256
#define PACKED_VALUES_PER_THREAD 16
#define PACKED_BLOCKS_PER_SM 2

#define FLAGS_NUM_THREADS 1024
#define FLAGS_VALUES_PER_THREAD 4
#define FLAGS_BLOCKS_PER_SM 1

#define KEYS_NUM_THREADS 256
#define KEYS_VALUES_PER_THREAD 16
#define KEYS_BLOCKS_PER_SM 2

