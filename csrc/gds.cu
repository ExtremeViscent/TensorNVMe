#include <stdexcept>
#include <memory>
#include "gds.h"
#include "cufile_utils.h"
#include "cufile.h"

GDSAsyncIO::GDSAsyncIO(unsigned int n_entries)
{
    // printf("Initializing the io Context\n");
    this->max_nr = n_entries;
    // TODO: initialize GDS and batch_io
    init_driver();
    init_batch_io();
    this->timeout.tv_sec = 0;
    this->timeout.tv_nsec = 100000000;
}

void GDSAsyncIO::register_file(int fd) {}

void GDSAsyncIO::register_file(int fd, CUfileHandle_t *cf_handle) {
    CUfileDescr_t cf_descr;
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "file register error:"
            << cuFileGetErrorString(status) << std::endl;
        close(fd);
        return;
    }
}

void GDSAsyncIO::init_driver(){
    int device_id;
	check_cudaruntimecall(cudaGetDevice(&device_id));

        status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
                throw std::runtime_error("cufile driver open error");
        }

	status = cuFileDriverSetMaxDirectIOSize(MAX_DIO_SIZE);
	if (status.err != CU_FILE_SUCCESS) {
		throw std::runtime_error("cufile driver set max direct io size error");
	}
}

void GDSAsyncIO::init_batch_io(){
    status = cuFileBatchIOSetUp(&batch_id, MAX_BATCH_IOS);
	if(status.err != 0) {
		throw std::runtime_error("cufile batch io setup error");
	}
}

GDSAsyncIO::~GDSAsyncIO()
{
    synchronize();
    cuFileBatchIODestroy(batch_id);
}

void GDSAsyncIO::get_event(WaitType wt)
{
    CUfileIOEvents_t io_batch_events[MAX_BATCH_IOS];
    uint nr;
    if (wt == WAIT)
        status = cuFileBatchIOGetStatus(batch_id, this->min_nr, &nr, io_batch_events, &(this->timeout));
    else
        status = cuFileBatchIOGetStatus(batch_id, 0, &nr, io_batch_events, &(this->timeout));
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cufile batch io get status error");
    }
    pending_nr -= nr;


    /* --------------------- ORIGINAL --------------------- */
    // std::unique_ptr<io_event> events(new io_event[this->max_nr]);
    // int num_events;

    // if(wt == WAIT)
    //     num_events = io_getevents(io_ctx, this->min_nr, this->max_nr, events.get(), &(this->timeout)); /* 获得异步I/O event个数 */
    // else
    //     num_events = io_getevents(io_ctx, 0, this->max_nr, events.get(), &(this->timeout)); /* 获得异步I/O event个数 */

    // for (int i = 0; i < num_events; i++) /* 开始获取每一个event并且做相应处理 */
    // {
    //     struct io_event event = events.get()[i];
    //     std::unique_ptr<IOData> data(static_cast<IOData *>(event.data));
    //     if (data->type == WRITE)
    //         this->n_write_events--;
    //     else if (data->type == READ)
    //         this->n_read_events--;
    //     else
    //         throw std::runtime_error("Unknown IO event type");
    //     if (data->callback != nullptr)
    //         data->callback();
    //     // printf("%d tasks to be done\n", this->n_write_events);
    // }
}

void GDSAsyncIO::cu_prep_pwrite(int fd, void *devPtr, size_t n_bytes, unsigned long long offset){
    // Split tensor into chunks
    int num_chunks = n_bytes / MAX_BUFFER_SIZE;
    int last_chunk_size = n_bytes % MAX_BUFFER_SIZE;
    int chunk_size = MAX_BUFFER_SIZE;
    if (last_chunk_size != 0) {
        num_chunks += 1;
    } else {
        last_chunk_size = MAX_BUFFER_SIZE;
    }
    off_t file_offset = offset;

    CUfileHandle_t cf_handle;
    register_file(fd, &cf_handle);
    fds.push(fd);
    cf_handles.push(cf_handle);

    CUfileIOParams_t iocbp;
    for (int i = 0; i < num_chunks; i++) {
        size_t size = i == num_chunks - 1 ? last_chunk_size : chunk_size;
        iocbp.mode = CUFILE_BATCH;
        iocbp.fh = cf_handle;
        iocbp.u.batch.devPtr_base = devPtr;
        iocbp.u.batch.file_offset = file_offset + i * chunk_size;
        iocbp.u.batch.devPtr_offset = i * chunk_size;
        iocbp.u.batch.size = size;
        iocbp.opcode = CUFILE_WRITE;
        iocbps.push(iocbp);
    }
}

void GDSAsyncIO::cu_prep_pread(int fd, void *devPtr, size_t n_bytes, unsigned long long offset){
    // Split tensor into chunks
    int num_chunks = n_bytes / MAX_BUFFER_SIZE;
    int last_chunk_size = n_bytes % MAX_BUFFER_SIZE;
    int chunk_size = MAX_BUFFER_SIZE;
    if (last_chunk_size != 0) {
        num_chunks += 1;
    } else {
        last_chunk_size = MAX_BUFFER_SIZE;
    }
    off_t file_offset = offset;

    CUfileHandle_t cf_handle;
    register_file(fd, &cf_handle);
    fds.push(fd);
    cf_handles.push(cf_handle);

    CUfileIOParams_t iocbp;
    for (int i = 0; i < num_chunks; i++) {
        size_t size = i == num_chunks - 1 ? last_chunk_size : chunk_size;
        iocbp.mode = CUFILE_BATCH;
        iocbp.fh = cf_handle;
        iocbp.u.batch.devPtr_base = devPtr;
        iocbp.u.batch.file_offset = file_offset + i * chunk_size;
        iocbp.u.batch.devPtr_offset = i * chunk_size;
        iocbp.u.batch.size = size;
        iocbp.opcode = CUFILE_READ;
        iocbps.push(iocbp);
    }
}

void GDSAsyncIO::cu_batchio_submit(){
    if (this->n_write_events > 0 || this->n_read_events > 0) 
        return;
    CUfileError_t status;
    // Split iocbps into batches
    int batch_size = std::min((int) iocbps.size(), MAX_BATCH_IOS);
    CUfileIOParams_t *iocbp_batch = new CUfileIOParams_t[batch_size];
    for (int i = 0; i < batch_size; i++) {
        iocbp_batch[i] = iocbps.front();
        iocbps.pop();
    }
    status = cuFileBatchIOSubmit(batch_id, batch_size, &(iocbps.front()), 0);
    if (status.err != 0) {
        throw std::runtime_error("cufile batch io submit error");
    }
    pending_nr += batch_size;
}

void GDSAsyncIO::write(int fd, void *buffer, size_t n_bytes, unsigned long long offset, callback_t callback)
{
    cu_prep_pwrite(fd, buffer, n_bytes, offset);

    cu_batchio_submit();

    this->n_write_events++;
}

void GDSAsyncIO::read(int fd, void *buffer, size_t n_bytes, unsigned long long offset, callback_t callback)
{
    cu_prep_pread(fd, buffer, n_bytes, offset);

    cu_batchio_submit();

    this->n_read_events++;
}

void GDSAsyncIO::sync_write_events()
{   // Cannot distinguish rw, so just wait for all events
    // while (this->n_write_events > 0)
    //     get_event(WAIT);
    synchronize();
}

void GDSAsyncIO::sync_read_events()
{
    // while (this->n_read_events > 0)
    //     get_event(WAIT);
    synchronize();
}

void GDSAsyncIO::synchronize()
{
    while (pending_nr > 0)
        get_event(WAIT);
}

void GDSAsyncIO::writev(int fd, const iovec *iov, unsigned int iovcnt, unsigned long long offset, callback_t callback)
{
    // struct iocb iocb
    // {
    // }; //建立一个异步I/O需求
    // struct iocb *iocbs = &iocb;
    // auto *data = new IOData(WRITE, callback, iov);

    // io_prep_pwritev(&iocb, fd, iov, iovcnt, (long long)offset); // 初始化这个异步I/O需求 counter为偏移量

    // iocb.data = data;
    // io_submit(this->io_ctx, 1, &iocbs); // 提交这个I/O不会堵塞

    // this->n_write_events++;
}

void GDSAsyncIO::readv(int fd, const iovec *iov, unsigned int iovcnt, unsigned long long offset, callback_t callback)
{
    // struct iocb iocb
    // {
    // }; //建立一个异步I/O需求
    // struct iocb *iocbs = &iocb;
    // auto *data = new IOData(READ, callback, iov);

    // io_prep_preadv(&iocb, fd, iov, iovcnt, (long long)offset);

    // iocb.data = data;
    // io_submit(this->io_ctx, 1, &iocbs); /* 提交这个I/O不会堵塞 */

    // this->n_read_events++;
}