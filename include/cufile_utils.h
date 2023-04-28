/*
 * Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __CUFILE_SAMPLE_UTILS_H_
#define __CUFILE_SAMPLE_UTILS_H_

#include <cassert>
#include <cstring>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <sys/stat.h>

#include <openssl/sha.h>
#include <openssl/evp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <dlfcn.h>
#define MAX_CHUNK_READ (64 * 1024UL)

#define check_cudadrivercall(fn) \
	do { \
		CUresult res = fn; \
		if (res != CUDA_SUCCESS) { \
			const char *str = nullptr; \
			cuGetErrorName(res, &str); \
			std::cerr << "cuda driver api call failed " << #fn \
				<< " res : "<< res << ", " <<  __LINE__ << ":" << str << std::endl; \
			std::cerr << "EXITING program!!!" << std::endl; \
			exit(1); \
		} \
	} while(0)

#define check_cudaruntimecall(fn) \
	do { \
		cudaError_t res = fn; \
		if (res != cudaSuccess) { \
			const char *str = cudaGetErrorName(res); \
			std::cerr << "cuda runtime api call failed " << #fn \
				<<  __LINE__ << ":" << str << std::endl; \
			std::cerr << "EXITING program!!!" << std::endl; \
			exit(1); \
		} \
	} while(0)

struct Prng {
	long rmax_;
	std::mt19937 rand_;
	std::uniform_int_distribution<long> dist_;
	Prng(long rmax) :
		rmax_(rmax),
		rand_(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
		dist_(std::uniform_int_distribution<long>(0, rmax_))
		{}
	long next_random_offset(void) {
		return dist_(rand_);
	}
};

//
// cuda driver error description
//
static inline const char *GetCuErrorString(CUresult curesult) {
	const char *descp;
	if (cuGetErrorName(curesult, &descp) != CUDA_SUCCESS)
		descp = "unknown cuda error";
	return descp;
}

//
// cuda runtime error description
//
static inline const char *GetCudaErrorString(cudaError_t cudaerr) {
	return cudaGetErrorName(cudaerr);
}


//
// cuFile APIs return both cuFile specific error codes as well as POSIX error codes
// for ease, the below template can be used for getting the error description depending
// on its type.

// POSIX
template<class T,
	typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
	status = std::abs(status);
	return IS_CUFILE_ERR(status) ?
		std::string(CUFILE_ERRSTR(status)) : std::string(std::strerror(status));
}

// CUfileError_t
template<class T,
	typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
	std::string errStr = cuFileGetErrorString(static_cast<int>(status.err));
	if (IS_CUDA_ERR(status))
		errStr.append(".").append(GetCuErrorString(status.cu_err));
	return errStr;
}

static inline size_t GetFileSize(int fd) {
	int ret;
	struct stat st;

	ret = fstat(fd, &st);
	return (ret == 0) ? st.st_size : -1;
}
#endif
