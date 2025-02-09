#include <stdio.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <error.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <sys/uio.h>
#include "offload.h"
#include "space_mgr.h"

iovec *tensors_to_iovec(const std::vector<at::Tensor> &tensors)
{
    iovec *iovs = static_cast<iovec *>(calloc(tensors.size(), sizeof(iovec)));
    for (size_t i = 0; i < tensors.size(); i++)
    {
        iovs[i].iov_base = tensors[i].data_ptr();
        iovs[i].iov_len = tensors[i].storage().nbytes();
    }
    return iovs;
}

std::unordered_set<std::string> get_backends()
{
    std::unordered_set<std::string> backends;
#ifndef DISABLE_URING
    backends.insert("uring");
#endif
#ifndef DISABLE_AIO
    backends.insert("aio");
#endif
#ifndef DISABLE_GDS
    backends.insert("gds");
#endif
    return backends;
}

void probe_asyncio(const std::string &backend)
{
    char* fp = tmpnam(nullptr);
    if (!fp)
    {
        printf("Create tmpfile error: %s\n", strerror(errno));
        throw std::runtime_error("uring probe failed\n");
    }
    printf("tmpfile: %s\n", fp);
    try
    {
        std::unique_ptr<AsyncIO> aio;
        if (backend == "uring")
#ifndef DISABLE_URING
            aio.reset(new UringAsyncIO(2));
#else
            throw std::runtime_error("backend is not installed\n");
#endif
        else if (backend == "aio")
#ifndef DISABLE_AIO
            aio.reset(new AIOAsyncIO(2));
#else
            throw std::runtime_error("backend is not installed\n");
#endif
        else if (backend == "gds")
#ifndef DISABLE_GDS
    aio.reset(new GDSAsyncIO(128));
#else
    throw std::runtime_error("GDS is not installed\n");
#endif
        else
            throw std::runtime_error("Invalid backend\n");
        std::cout<< "opening file: " << fp << std::endl;
        int fd = open(fp, O_CREAT | O_RDWR | O_DIRECT, 0664);
        if (fd < 0)
        {
            printf("Open tmpfile error: %s\n", strerror(errno));
            throw std::runtime_error("uring probe failed\n");
        }
        const int n_loop = 5, n_len = 4096;

        char text[n_loop][n_len];
        void *devPtr[n_loop];
        
        {
            for (int i = 0; i < n_loop; i++){
                cudaMalloc(&devPtr[i], n_len * sizeof(char));
                cudaMemset(devPtr[i], 'V', n_len * sizeof(char));
            }
        }

        int offset = 0;
        size_t len;
        for (int i = 0; i < n_loop; i++)
        {
            len = n_len;
            aio->write(fd, backend == "gds" ? devPtr[i] : text[i], len, offset, nullptr);
            offset += len;
        }
        aio->sync_write_events();

        std::cout << "sync_write_events" << std::endl;
        char new_text[n_loop][n_len];
        void *new_devPtr[n_loop];
        if (backend == "gds")
        {
            for (int i = 0; i < n_loop; i++){
                cudaMalloc(&new_devPtr[i], n_len * sizeof(char));
            }
        }
        offset = 0;
        for (int i = 0; i < n_loop; i++)
        {
            len = n_len;
            aio->read(fd, backend == "gds" ? new_devPtr[i] : new_text[i], len, offset, nullptr);
            offset += len;
        }
        if (backend == "gds")
        {
            for (int i = 0; i < n_loop; i++){
                cudaMemcpy(new_text[i], new_devPtr[i], n_len * sizeof(char), cudaMemcpyDeviceToHost);
            }
        }
        aio->sync_read_events();
        for (int i = 0; i < n_loop; i++)
        {
            for (int j = 0; j < n_len; j++)
            {
                assert(text[i][j] == new_text[i][j]);
            }
        }
        close(fd);
    }
    catch (...)
    {
        // close(fd);
        throw std::runtime_error("probe failed\n");
    }
}

bool probe_backend(const std::string &backend)
{
    std::unordered_set<std::string> backends = get_backends();
    if (backends.find(backend) == backends.end())
        return false;
    try
    {
        probe_asyncio(backend);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

AsyncIO *create_asyncio(unsigned int n_entries, const std::string &backend)
{
    std::unordered_set<std::string> backends = get_backends();
    if (backends.empty())
        throw std::runtime_error("No asyncio backend is installed");
    if (backends.find(backend) == backends.end())
        throw std::runtime_error("Unsupported backend: " + backend);
    if (!probe_backend(backend))
        throw std::runtime_error("Backend \"" + backend + "\" is not install correctly");
#ifndef DISABLE_URING
    if (backend == "uring")
        return new UringAsyncIO(n_entries);
#endif
#ifndef DISABLE_AIO
    if (backend == "aio")
        return new AIOAsyncIO(n_entries);
#endif
#ifndef DISABLE_GDS
    if (backend == "gds")
        return new GDSAsyncIO(n_entries);
#endif
    throw std::runtime_error("Unsupported backend: " + backend);
}

Offloader::Offloader(const std::string &filename, unsigned int n_entries, const std::string &backend) : filename(filename), space_mgr(SpaceManager(0))
{
    this->aio = create_asyncio(n_entries, backend);
    if (backend == "gds") gds_enabled = true;
    this->fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
    this->aio->register_file(fd);
}

SpaceInfo Offloader::prepare_write(const at::Tensor &tensor, const std::string &key)
{
    if (!tensor.is_contiguous() || gds_enabled ? !tensor.is_cuda(): !tensor.is_cpu())
        throw std::runtime_error("Tensor must be contiguous and on cpu");
    ull bytes = tensor.storage().nbytes();
    ull offset = this->space_mgr.alloc(bytes);
    SpaceInfo space_info(offset, bytes);
    this->tensors_info[key] = space_info;
    return space_info;
}

SpaceInfo Offloader::prepare_read(const at::Tensor &tensor, const std::string &key)
{
    if (!tensor.is_contiguous() || gds_enabled ? !tensor.is_cuda(): !tensor.is_cpu())
        throw std::runtime_error("Tensor must be contiguous and on cpu");
    if (this->tensors_info.find(key) == this->tensors_info.end())
        throw std::runtime_error("Read error, tensor not found");
    ull bytes = tensor.storage().nbytes();
    SpaceInfo space_info = this->tensors_info[key];
    if (bytes != space_info.second)
        throw std::runtime_error("Read error, tensor shape mismatch");
    this->tensors_info.erase(key);
    return space_info;
}

void Offloader::async_write(const at::Tensor &tensor, const std::string &key, callback_t callback)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_write(tensor, key);
    this->aio->write(this->fd, tensor.data_ptr(), bytes, offset, callback);

    this->aio->get_event(NOWAIT);
}

void Offloader::async_read(const at::Tensor &tensor, const std::string &key, callback_t callback)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_read(tensor, key);
    auto fn = std::bind(&Offloader::release, this, offset, bytes, callback);
    this->aio->read(this->fd, tensor.data_ptr(), bytes, offset, fn);

    this->aio->get_event(NOWAIT);
}

void Offloader::sync_write(const at::Tensor &tensor, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_write(tensor, key);
    lseek(this->fd, offset, SEEK_SET);
    write(this->fd, tensor.data_ptr(), bytes);
}

void Offloader::sync_read(const at::Tensor &tensor, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_read(tensor, key);
    lseek(this->fd, offset, SEEK_SET);
    read(this->fd, tensor.data_ptr(), bytes);
    release(offset, bytes);
}

void Offloader::sync_write_events()
{
    this->aio->sync_write_events();
}

void Offloader::sync_read_events()
{
    this->aio->sync_read_events();
}

void Offloader::synchronize()
{
    this->aio->synchronize();
}

Offloader::~Offloader()
{
    errno = 0;
    delete this->aio;
    close(this->fd);
    if (remove(this->filename.c_str()) != 0)
        printf("Remove \"%s\" error(%d): %s\n", this->filename.c_str(), errno, strerror(errno));
}

SpaceInfo Offloader::prepare_writev(const std::vector<at::Tensor> &tensors, const std::string &key)
{
    ull total_bytes = 0;
    for (const at::Tensor &tensor : tensors)
    {
        if (!tensor.is_contiguous() || !tensor.is_cpu())
            throw std::runtime_error("Tensor must be contiguous and on cpu");
        total_bytes += tensor.storage().nbytes();
    }
    ull offset = this->space_mgr.alloc(total_bytes);
    SpaceInfo space_info(offset, total_bytes);
    this->tensors_info[key] = space_info;
    return space_info;
}

SpaceInfo Offloader::prepare_readv(const std::vector<at::Tensor> &tensors, const std::string &key)
{
    ull total_bytes = 0;
    for (const at::Tensor &tensor : tensors)
    {
        if (!tensor.is_contiguous() || !tensor.is_cpu())
            throw std::runtime_error("Tensor must be contiguous and on cpu");
        total_bytes += tensor.storage().nbytes();
    }
    if (this->tensors_info.find(key) == this->tensors_info.end())
        throw std::runtime_error("Read error, tensor not found");
    SpaceInfo space_info = this->tensors_info[key];
    if (total_bytes != space_info.second)
        throw std::runtime_error("Read error, tensor shape mismatch");
    this->tensors_info.erase(key);
    return space_info;
}

void Offloader::async_writev(const std::vector<at::Tensor> &tensors, const std::string &key, callback_t callback)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_writev(tensors, key);
    iovec *iov = tensors_to_iovec(tensors);
    this->aio->writev(this->fd, iov, tensors.size(), offset, callback);

    this->aio->get_event(NOWAIT);
}

void Offloader::async_readv(const std::vector<at::Tensor> &tensors, const std::string &key, callback_t callback)
{

    ull offset, bytes;
    std::tie(offset, bytes) = prepare_readv(tensors, key);
    iovec *iov = tensors_to_iovec(tensors);
    auto fn = std::bind(&Offloader::release, this, offset, bytes, callback);
    this->aio->readv(this->fd, iov, tensors.size(), offset, fn);

    this->aio->get_event(NOWAIT);
}

void Offloader::sync_writev(const std::vector<at::Tensor> &tensors, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_writev(tensors, key);
    iovec *iov = tensors_to_iovec(tensors);
    lseek(this->fd, offset, SEEK_SET);
    writev(this->fd, iov, tensors.size());
    delete iov;
}

void Offloader::sync_readv(const std::vector<at::Tensor> &tensors, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_readv(tensors, key);
    iovec *iov = tensors_to_iovec(tensors);
    lseek(this->fd, offset, SEEK_SET);
    readv(this->fd, iov, tensors.size());
    delete iov;
}

void Offloader::release(ull offset, ull bytes, callback_t callback)
{
    this->space_mgr.free(offset, bytes);
    if (callback != nullptr)
        callback();
}

GDSOffloader::GDSOffloader(const std::string &filename, unsigned int n_entries, const std::string &backend) : filename(filename), space_mgr(SpaceManager(0))
{
    this->aio = create_asyncio(n_entries, backend);
    this->fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
    this->aio->register_file(fd);
}

SpaceInfo GDSOffloader::prepare_write(const at::Tensor &tensor, const std::string &key)
{
    if (!tensor.is_contiguous() || !tensor.is_cuda())
        throw std::runtime_error("Tensor must be contiguous and on gpu");
    ull bytes = tensor.storage().nbytes();
    ull offset = this->space_mgr.alloc(bytes);
    SpaceInfo space_info(offset, bytes);
    this->tensors_info[key] = space_info;
    return space_info;
}

SpaceInfo GDSOffloader::prepare_read(const at::Tensor &tensor, const std::string &key)
{
    if (!tensor.is_contiguous() || !tensor.is_cuda())
        throw std::runtime_error("Tensor must be contiguous and on gpu");
    if (this->tensors_info.find(key) == this->tensors_info.end())
        throw std::runtime_error("Read error, tensor not found");
    ull bytes = tensor.storage().nbytes();
    SpaceInfo space_info = this->tensors_info[key];
    if (bytes != space_info.second)
        throw std::runtime_error("Read error, tensor shape mismatch");
    this->tensors_info.erase(key);
    return space_info;
}

void GDSOffloader::async_write(const at::Tensor &tensor, const std::string &key, callback_t callback)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_write(tensor, key);
    this->aio->write(this->fd, tensor.data_ptr(), bytes, offset, callback);

    this->aio->get_event(NOWAIT);
}

void GDSOffloader::async_read(const at::Tensor &tensor, const std::string &key, callback_t callback)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_read(tensor, key);
    auto fn = std::bind(&GDSOffloader::release, this, offset, bytes, callback);
    this->aio->read(this->fd, tensor.data_ptr(), bytes, offset, fn);

    this->aio->get_event(NOWAIT);
}


void GDSOffloader::sync_write_events()
{
    this->aio->sync_write_events();
}

void GDSOffloader::sync_read_events()
{
    this->aio->sync_read_events();
}

void GDSOffloader::synchronize()
{
    this->aio->synchronize();
}

GDSOffloader::~GDSOffloader()
{
    errno = 0;
    delete this->aio;
    close(this->fd);
    if (remove(this->filename.c_str()) != 0)
        printf("Remove \"%s\" error(%d): %s\n", this->filename.c_str(), errno, strerror(errno));
}

void GDSOffloader::release(ull offset, ull bytes, callback_t callback)
{
    this->space_mgr.free(offset, bytes);
    if (callback != nullptr)
        callback();
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<Offloader>(m, "Offloader")
        .def(py::init<const std::string &, unsigned int, const std::string &>(), py::arg("filename"), py::arg("n_entries"), py::arg("backend") = "uring")
        .def("async_write", &Offloader::async_write, py::arg("tensor"), py::arg("key"), py::arg("callback") = py::none())
        .def("async_read", &Offloader::async_read, py::arg("tensor"), py::arg("key"), py::arg("callback") = py::none())
        .def("sync_write", &Offloader::sync_write, py::arg("tensor"), py::arg("key"))
        .def("sync_read", &Offloader::sync_read, py::arg("tensor"), py::arg("key"))
        .def("sync_write_events", &Offloader::sync_write_events)
        .def("sync_read_events", &Offloader::sync_write_events)
        .def("synchronize", &Offloader::synchronize)
        .def("async_writev", &Offloader::async_writev, py::arg("tensors"), py::arg("key"), py::arg("callback") = py::none())
        .def("async_readv", &Offloader::async_readv, py::arg("tensors"), py::arg("key"), py::arg("callback") = py::none())
        .def("sync_writev", &Offloader::sync_writev, py::arg("tensors"), py::arg("key"))
        .def("sync_readv", &Offloader::sync_readv, py::arg("tensors"), py::arg("key"));
    py::class_<GDSOffloader>(m, "GDSOffloader")
        .def(py::init<const std::string &, unsigned int, const std::string &>(), py::arg("filename"), py::arg("n_entries"), py::arg("backend") = "gds")
        .def("async_write", &GDSOffloader::async_write, py::arg("tensor"), py::arg("key"), py::arg("callback") = py::none())
        .def("async_read", &GDSOffloader::async_read, py::arg("tensor"), py::arg("key"), py::arg("callback") = py::none())
        // .def("sync_write", &GDSOffloader::sync_write, py::arg("tensor"), py::arg("key"))
        // .def("sync_read", &GDSOffloader::sync_read, py::arg("tensor"), py::arg("key"))
        .def("sync_write_events", &GDSOffloader::sync_write_events)
        .def("sync_read_events", &GDSOffloader::sync_write_events)
        .def("synchronize", &GDSOffloader::synchronize);
    m.def("get_backends", get_backends);
    m.def("probe_backend", probe_backend, py::arg("backend"));
}