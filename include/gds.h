#pragma once

#include <libaio.h>
#include "asyncio.h"
#include "unistd.h"
#include <queue>
#include "cufile_utils.h"
#include "cufile.h"
#include "cuda_runtime.h"
#include "cuda.h"

#define MAX_DIO_SIZE 16384
#define MAX_BATCH_IOS 128
#define MAX_BUFFER_SIZE 16 * 1024 * 1024 // 16MB for A100, chang to 1MB for V100 (Shadow buffer size)

class GDSAsyncIO : public AsyncIO
{
private:
    io_context_t io_ctx = nullptr;
    int n_write_events = 0; /* event个数 */
    int n_read_events = 0;
    int max_nr;
    int min_nr = 1;
    int pending_nr = 0;
    struct timespec timeout;
    CUfileError_t status;
    std::queue<CUfileHandle_t> cf_handles;
    std::queue<int> fds;
    std::queue<CUfileIOParams_t> iocbps;
    CUfileBatchHandle_t batch_id;

    void get_event(WaitType wt);

public:
    GDSAsyncIO(unsigned int n_entries);
    ~GDSAsyncIO();

    void cu_prep_pwrite(int fd, void *devPtr, size_t n_bytes, unsigned long long offset);
    void cu_prep_pread(int fd, void *devPtr, size_t n_bytes, unsigned long long offset);
    void cu_batchio_submit();

    void write(int fd, void *buffer, size_t n_bytes, unsigned long long offset, callback_t callback);
    void read(int fd, void *buffer, size_t n_bytes, unsigned long long offset, callback_t callback);
    void writev(int fd, const iovec *iov, unsigned int iovcnt, unsigned long long offset, callback_t callback);
    void readv(int fd, const iovec *iov, unsigned int iovcnt, unsigned long long offset, callback_t callback);

    void sync_write_events();
    void sync_read_events();
    void synchronize();

    void register_file(int fd, CUfileHandle_t *cf_handle);
    void register_file(int fd);
    void init_driver();
    void init_batch_io();
};