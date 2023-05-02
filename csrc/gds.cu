#include <stdexcept>
#include <memory>
#include "gds.h"
#include "cufile_utils.h"
#include "cufile.h"
#include "nvToolsExt.h"

GDSAsyncIO::GDSAsyncIO(unsigned int n_entries)
{
    // printf("Initializing the io Context\n");
    this->max_nr = n_entries;
    // TODO: initialize GDS and batch_io
    init_driver();
    this->timeout.tv_sec = 0;
    this->timeout.tv_nsec = 100000000;
}

void GDSAsyncIO::register_file(int fd) {
    CUfileDescr_t cf_descr;
    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    memset((void *)&cf_handle, 0, sizeof(CUfileHandle_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    // std::cout << "file handle: " << cf_handle << std::endl;
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "file register error:"
            << cuFileGetErrorString(status) << std::endl;
    }
}

void GDSAsyncIO::init_driver(){
    int device_id;
	check_cudaruntimecall(cudaGetDevice(&device_id));

    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("cufile driver open error");
    }
    else{
        std::cout << "cufile driver opened on device " << device_id << std::endl;
    }

	status = cuFileDriverSetMaxDirectIOSize(MAX_DIO_SIZE);
	if (status.err != CU_FILE_SUCCESS) {
		throw std::runtime_error("cufile driver set max direct io size error");
	}

    // Register buffer
    // status = cuFileBufRegister(
    //     devPtr, 
    //     n_bytes, 0);
    // if (status.err != CU_FILE_SUCCESS) {
    //     std::cerr << "cuFileBufRegister failed" << std::endl;
    // }
    int num_buffers = 128;
    for (int i = 0; i < num_buffers; i++){
        void *buffer;
        size_t n_bytes = MAX_BUFFER_SIZE;
        check_cudaruntimecall(cudaMalloc(&buffer, n_bytes));
        check_cudaruntimecall(cudaMemset(buffer, 0, n_bytes));
        status = cuFileBufRegister(
            buffer, 
            n_bytes, 0);
        if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cuFileBufRegister failed" << std::endl;
        }
        avail_buffers.push(buffer);
    }
}


GDSAsyncIO::~GDSAsyncIO()
{
    synchronize();
    // Destroy batch io
    while (!avail_batch_ids.empty())
    {
        CUfileBatchHandle_t batch_id = avail_batch_ids.top();
        cuFileBatchIODestroy(batch_id);
        avail_batch_ids.pop();
    }
    
    // Deregsiter buffer
    while (!avail_buffers.empty())
    {
        cuFileBufDeregister(avail_buffers.top());
        avail_buffers.pop();
    }
    
    // Close driver
    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cufile driver close error");
    }
    
}

void GDSAsyncIO::get_event(WaitType wt)
{
    // CUfileIOEvents_t io_batch_events_w[MAX_BATCH_IOS];
    // CUfileIOEvents_t io_batch_events_r[MAX_BATCH_IOS];
    // uint nr;
    // if (!batch_ids_w.empty()){
    //     CUfileBatchHandle_t batch_id = batch_ids_w.front();
    //     if (wt == WAIT)
    //         status = cuFileBatchIOGetStatus(batch_id, this->min_nr, &nr, io_batch_events_w, &(this->timeout));
    //     else
    //         status = cuFileBatchIOGetStatus(batch_id, 0, &nr, io_batch_events_w, &(this->timeout));
    //     if (status.err != CU_FILE_SUCCESS) {
    //         throw std::runtime_error("cufile batch io get status error");
    //     }
    //     this->n_write_events -= nr;
    // }
    
    // if (!batch_ids_r.empty()){
    //     CUfileBatchHandle_t batch_id = batch_ids_r.front();
    //     if (wt == WAIT)
    //         status = cuFileBatchIOGetStatus(batch_id, this->min_nr, &nr, io_batch_events_r, &(this->timeout));
    //     else
    //         status = cuFileBatchIOGetStatus(batch_id, 0, &nr, io_batch_events, &(this->timeout));
    //     if (status.err != CU_FILE_SUCCESS) {
    //         throw std::runtime_error("cufile batch io get status error");
    //     }
    // }
    


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


void GDSAsyncIO::cu_batchio_setup(CUfileBatchHandle_t *batch_id, int nr){
    assert(nr <= MAX_BATCH_IOS);
    status = cuFileBatchIOSetUp(batch_id, nr);
	if(status.err != 0) {
		throw std::runtime_error("cuFileBatchIOSetUp failed");
	}
}

void print_metedata(CUfileIOParams_t iocbp_0){
    CUfileBatchMode_t mode = iocbp_0.mode;
    CUfileHandle_t fh = iocbp_0.fh;
    void *devPtr = iocbp_0.u.batch.devPtr_base;
    off_t file_offset = iocbp_0.u.batch.file_offset;
    off_t devPtr_offset = iocbp_0.u.batch.devPtr_offset;
    size_t size = iocbp_0.u.batch.size;
    int opcode = iocbp_0.opcode;

    std::cout << "mode: " << mode << std::endl;
    std::cout << "fh: " << fh << std::endl;
    std::cout << "devPtr: " << devPtr << std::endl;
    std::cout << "file_offset: " << file_offset << std::endl;
    std::cout << "devPtr_offset: " << devPtr_offset << std::endl;
    std::cout << "size: " << size << std::endl;
    std::cout << "opcode: " << opcode << std::endl;
}

void GDSAsyncIO::cu_batchio_submit(CUfileBatchHandle_t batch_id, CUfileIOParams_t *iocbp, int nr){
    // Register buffer
    // for (int i = 0; i < nr; i++){
    //     status = cuFileBufRegister(
    //         iocbp[i].u.batch.devPtr_base + iocbp[i].u.batch.devPtr_offset, 
    //         iocbp[i].u.batch.size, 0);
    //     if (status.err != CU_FILE_SUCCESS) {
    //         std::cerr << "cuFileBufRegister failed" << std::endl;
    //     }
    // }
    nvtxRangePushA("cuFileBatchIOSubmit");
    status = cuFileBatchIOSubmit(batch_id, nr, iocbp, 0);
    // std::cout << "submitting " << nr << " iocbp" << std::endl;

    // std::cout << "metadata of first iocbp: "<< std::endl;
    // print_metedata(iocbp[0]);
    // std::cout << "metadata of second iocbp: "<< std::endl;
    // print_metedata(iocbp[1]);

    if(status.err != 0) {
        throw std::runtime_error("cuFileBatchIOSubmit failed");
    }
    // std::cout << "submitted" << std::endl;
    void *cookie = iocbp[0].cookie;
    // std::cout << "cookie: " << cookie << std::endl;
    nvtxRangePop();
}

void GDSAsyncIO::pad4k(CUfileIOParams_t *iocbp){
    size_t size = iocbp->u.batch.size;
    if (size % 4096 == 0) return;
    // std::cout << "padding" << std::endl;
    size_t new_size = size + (4096 - size % 4096);
    void *devPtr = iocbp->u.batch.devPtr_base;
    void *new_devPtr = (void *)((char *)devPtr + size);
    size_t padding_size = new_size - size;
    cudaMalloc(&new_devPtr, new_size);
    // cudaMemset((void *)((char *)new_devPtr + size), 0, padding_size);
    if (iocbp->opcode == CUFILE_WRITE)
        cudaMemcpy(new_devPtr, devPtr, size, cudaMemcpyDeviceToHost);
    iocbp->u.batch.devPtr_base = new_devPtr;
    iocbp->u.batch.devPtr_offset = 0;
    // iocbp->u.batch.size = new_size;
    // Update buffer
    // status = cuFileBufRegister(new_devPtr, new_size, 0);
    // if (status.err != CU_FILE_SUCCESS) {
    //     // std::cerr << "cuFileBufRegister failed" << std::endl;
    // }
}

void GDSAsyncIO::operate(int fd, void *devPtr, size_t n_bytes, unsigned long long offset, callback_t callback, bool is_write)
{
    nvtxRangePushA("operate");
    Batch *batch = new Batch();
    batch->callback = callback;
    // Split tensor into chunks
    size_t chunk_size = MAX_BUFFER_SIZE;
    int num_chunks = n_bytes / chunk_size;    // int num_buffers = 128;
    // for (int i = 0; i < num_buffers; i++){
    //     void *buffer;
    //     size_t n_bytes = MAX_BUFFER_SIZE;
    //     check_cudaruntimecall(cudaMalloc(&buffer, n_bytes));
    //     check_cudaruntimecall(cudaMemset(buffer, 0, n_bytes));
    //     status = cuFileBufRegister(
    //         buffer, 
    //         n_bytes, 0);
    //     if (status.err != CU_FILE_SUCCESS) {
    //         std::cerr << "cuFileBufRegister failed" << std::endl;
    //     }
    //     avail_buffers.push(buffer);
    // }
    int last_chunk_size = n_bytes % chunk_size;
    batch->n_bytes = n_bytes;

    // std::cout << n_bytes / MAX_BUFFER_SIZE << std::endl;

    if (last_chunk_size != 0) {
        num_chunks += 1;
    } else {
        last_chunk_size = MAX_BUFFER_SIZE;
    }
    off_t file_offset = offset;

    // Pad file offset to 4k
    if (offset % 4096 != 0) {
        file_offset = offset + (4096 - offset % 4096);
    }

    // std::cout << "n_bytes: " << n_bytes << std::endl; 
    // std::cout << "num_chunks: " << num_chunks << std::endl;
    // std::cout << "last_chunk_size: " << last_chunk_size << std::endl;
    // std::cout << "chunk_size: " << chunk_size << std::endl;
    // std::cout << "file_offset: " << file_offset << std::endl;
    // std::cout << "devPtr: " << devPtr << std::endl;

    // // Register buffer
    // status = cuFileBufRegister(
    //     devPtr, 
    //     n_bytes, 0);
    // if (status.err != CU_FILE_SUCCESS) {
    //     std::cerr << "cuFileBufRegister failed" << std::endl;
    // }

    // Register file handle if not registered
    if (cf_handle == nullptr) {
        register_file(fd);
    }
    
    // Prepare batch IO control block
    CUfileIOParams_t iocbp[num_chunks];
    batch->iocbp = iocbp;
    batch->buffers = new void*[num_chunks];
    batch->num_chunks = num_chunks;
    batch->devPtr = devPtr;
    for (int i = 0; i < num_chunks; i++) {
        size_t size = i == num_chunks - 1 ? last_chunk_size : chunk_size;
        // Copy to buffer
        assert(!avail_buffers.empty());
        void *buffer = avail_buffers.top();
        avail_buffers.pop();
        if (is_write) {
            cudaMemcpy(buffer, devPtr + i * chunk_size, size, cudaMemcpyDefault);
        }
        // Prepare IO control block
        iocbp[i].mode = CUFILE_BATCH;
        iocbp[i].fh = cf_handle;
        iocbp[i].u.batch.devPtr_base = buffer;
        iocbp[i].u.batch.file_offset = file_offset + i * chunk_size;
        iocbp[i].u.batch.devPtr_offset = 0;
        iocbp[i].u.batch.size = chunk_size;
        iocbp[i].opcode = is_write ? CUFILE_WRITE : CUFILE_READ;
        iocbp[i].cookie = nullptr;
        batch->buffers[i] = buffer;
        // Pad to 4K
        // pad4k(&iocbp[i]);
    }

    // Submit batch IO
    int queue_depth = MAX_BATCH_IOS;
    batch->num_mini_batches = num_chunks / queue_depth + (num_chunks % queue_depth == 0 ? 0 : 1);
    // std::cout << "num_mini_batches: " << batch->num_mini_batches << std::endl;
    batch->batch_sizes = new int[batch->num_mini_batches];
    batch->batch_id = new CUfileBatchHandle_t[batch->num_mini_batches];
    // print_metedata(iocbp[0]);
    for (int i = 0; i < batch->num_mini_batches; i ++) {
        int nr = std::min(queue_depth, num_chunks - i);
        batch->batch_sizes[i] = nr;
        // std::cout << "batch_size: " << batch->batch_sizes[i] << std::endl;
        CUfileIOParams_t *mini_batch;
        mini_batch = &iocbp[i];
        CUfileBatchHandle_t batch_id;
        if (!avail_batch_ids.empty()) {
            batch_id = avail_batch_ids.top();
            avail_batch_ids.pop();
        } else {
            cu_batchio_setup(&batch_id, nr);
        }
        cu_batchio_submit(batch_id, mini_batch, nr);
        batch->batch_id[i] = batch_id;
    }
    if (is_write) {
        batches_w.push(batch);
    }
    else {
        batches_r.push(batch);
    }
    nvtxRangePop();
}
void GDSAsyncIO::write(int fd, void *devPtr, size_t n_bytes, unsigned long long offset, callback_t callback)
{
    operate(fd, devPtr, n_bytes, offset, callback, true);
}

void GDSAsyncIO::read(int fd, void *devPtr, size_t n_bytes, unsigned long long offset, callback_t callback)
{
    operate(fd, devPtr, n_bytes, offset, callback, false);
}

void GDSAsyncIO::sync_write_events()
{   
    synchronize(true);
}

void GDSAsyncIO::sync_read_events()
{
    synchronize(false);
}

void GDSAsyncIO::synchronize(bool is_write)
{   
    nvtxRangePushA("synchronize");
    std::queue<Batch *> *batches = is_write ? &batches_w : &batches_r;
    while (!batches->empty()){
        Batch *batch = batches->front();
        int *batch_sizes = batch->batch_sizes;
        int num_mini_batches = batch->num_mini_batches;
        int queue_depth = MAX_BATCH_IOS;
        // std::cout << "num_mini_batches: " << num_mini_batches << std::endl;
        for (int i = 0; i < num_mini_batches; i++) {
            // std::cout << "polling mini batch " << i << std::endl;
            // std::cout << "batch_sizes[i]: " << batch_sizes[i] << std::endl;
            if (batch_sizes[i] == 0) continue;
            int pending = batch_sizes[i];
            CUfileIOEvents_t io_batch_events[batch_sizes[i]];
            CUfileBatchHandle_t batch_id = batch->batch_id[i];
            int timeout = 1000;
            while (pending > 0 && timeout > 0)
            {
                nvtxRangePushA("polling");
                uint nr=batch_sizes[i];
                status = cuFileBatchIOGetStatus(batch_id, batch_sizes[i], &nr, io_batch_events, &(this->timeout));
                if (status.err != CU_FILE_SUCCESS) {
                    throw std::runtime_error("cufile batch io get status error");
                }
                pending -= nr;
                timeout--;
                nvtxRangePop();
            }
            // return batch_id to pool
            avail_batch_ids.push(batch_id);
        }
        // Transfer buffers back
        {
            size_t chunk_size = MAX_BUFFER_SIZE;
            int num_chunks = batch->num_chunks;
            size_t last_chunk_size = batch->n_bytes % chunk_size;
            CUfileIOParams_t *iocbp = batch->iocbp;
            void **buffers = batch->buffers;
            for (int i = 0; i < num_chunks; i++)
            {
                size_t size = i == num_chunks - 1 ? last_chunk_size : chunk_size;
                void *src = buffers[i];
                if (!is_write)
                {
                    void *dst = batch->devPtr + i * chunk_size;
                    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
                }
                avail_buffers.push(src);
            }
        }

        batches->pop();
        if (batch->callback != nullptr) {
            // std::cout << "callback" << std::endl;
            batch->callback();
        }
        delete batch;
    }
    nvtxRangePop();
}

void GDSAsyncIO::synchronize()
{
    synchronize(true);
    synchronize(false);
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

void rw_verified(){
    GDSAsyncIO *gds = new GDSAsyncIO(128);
    char *fp = tmpnam(NULL);
    fp = "./gds_test.bin";
    char *fp2 = "./gds_ret.bin";
    char *fp3 = "./gds_in.bin";
    int fd = open(fp, O_CREAT | O_RDWR | O_DIRECT, 0664);
    int fd2 = open(fp2, O_CREAT | O_RDWR, 0664);
    int fd3 = open(fp3, O_CREAT | O_RDWR, 0664);
    char *buf;
    size_t n_bytes = 4 * 1000 * 1000;
    CUfileError_t status;
    void *input = malloc(2 * n_bytes);
    memset(input, 114, n_bytes);
    memset(input + n_bytes, 514, n_bytes);
    write(fd3, input, 2 * n_bytes);
    cudaMalloc(&buf, 2 * n_bytes);
    cudaMemcpy(buf, input, 2 * n_bytes, cudaMemcpyHostToDevice);

    gds->write(fd, buf, n_bytes, 0, NULL);
    gds->write(fd, buf + n_bytes, n_bytes, n_bytes, NULL);
    gds->synchronize();
    cudaFree(buf);
    cudaMalloc(&buf, 2 * n_bytes);
    cudaMemset(buf, 1, 2 * n_bytes);
    gds->read(fd, buf, n_bytes, 0, NULL);
    gds->read(fd, buf + n_bytes, n_bytes, n_bytes, NULL);
    gds->synchronize();
    char *ret = (char *)malloc(2 * n_bytes);
    cudaMemcpy(ret, buf, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ret + n_bytes, buf + n_bytes, n_bytes, cudaMemcpyDeviceToHost);
    write(fd2, ret, 2 * n_bytes);
    close(fd);
    close(fd2);
    close(fd3);
}

void w_only(){
    GDSAsyncIO *gds = new GDSAsyncIO(128);
    char *fp = tmpnam(NULL);
    fp = "/data/gds_test.bin";
    int fd = open(fp, O_CREAT | O_RDWR | O_DIRECT, 0664);
    size_t n_bytes = 4 * 1000 * 1000;
    void *devPtr;
    cudaMalloc(&devPtr, n_bytes);
    cudaMemset(devPtr, 1, n_bytes);
    gds->write(fd, devPtr, n_bytes, 0, NULL);
    gds->synchronize();
    cudaFree(devPtr);
}

int main()
{
    cudaSetDevice(1);
    rw_verified();
    

    return 0;
}