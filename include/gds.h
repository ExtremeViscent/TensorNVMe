#pragma once

#include <libaio.h>
#include "asyncio.h"
#include "unistd.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include "cufile_utils.h"
#include <cufile.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define MAX_DIO_SIZE 16384
#define MAX_BATCH_IOS 128
#define MAX_BUFFER_SIZE (size_t) 1024 * 1024 // Shadow buffer size
#define BAR_SIZE 256 * 1024 * 1024 // 256MB for V100, VRAM size for A100

class GDSAsyncIO : public AsyncIO
{
private:
    struct Batch
    {
        CUfileBatchHandle_t *batch_id;
        CUfileIOParams_t *iocbp;
        void *devPtr;
        void **buffers;
        int *batch_sizes;
        int num_chunks;
        int num_mini_batches;
        size_t n_bytes;
        callback_t callback;
    };
    io_context_t io_ctx = nullptr;
    int n_write_events = 0; /* event个数 */
    int n_read_events = 0;
    int max_nr;
    int min_nr = 1;
    int pending_nr = 0;
    struct timespec timeout;
    CUfileError_t status;
    CUfileHandle_t cf_handle = nullptr;
    std::queue<Batch*> batches_w;
    std::queue<Batch*> batches_r;
    std::stack<void*> avail_buffers;
    std::stack<void*> avail_batch_ids;

    void get_event(WaitType wt);
    void operate(int fd, void *devPtr, size_t n_bytes, unsigned long long offset, callback_t callback, bool is_write);
    void pad4k(CUfileIOParams_t *iocbp);
    

public:
    GDSAsyncIO(unsigned int n_entries);
    ~GDSAsyncIO();

    // void cu_prep_pwrite(int fd, void *devPtr, size_t n_bytes, unsigned long long offset);
    // void cu_prep_pread(int fd, void *devPtr, size_t n_bytes, unsigned long long offset);
    void cu_batchio_setup(CUfileBatchHandle_t *batch_id, int nr);
    void cu_batchio_submit(CUfileBatchHandle_t batch_id, CUfileIOParams_t *iocbp, int nr);

    void write(int fd, void *devPtr, size_t n_bytes, unsigned long long offset, callback_t callback);
    void read(int fd, void *devPtr, size_t n_bytes, unsigned long long offset, callback_t callback);
    void writev(int fd, const iovec *iov, unsigned int iovcnt, unsigned long long offset, callback_t callback);
    void readv(int fd, const iovec *iov, unsigned int iovcnt, unsigned long long offset, callback_t callback);

    void sync_write_events();
    void sync_read_events();
    void synchronize(bool is_write);
    void synchronize();

    void register_file(int fd);
    void init_driver();
};