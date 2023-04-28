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
#include "offload_gds.h"
#include "space_mgr.h"

// iovec *tensors_to_iovec(const std::vector<at::Tensor> &tensors)
// {
//     iovec *iovs = static_cast<iovec *>(calloc(tensors.size(), sizeof(iovec)));
//     for (size_t i = 0; i < tensors.size(); i++)
//     {
//         iovs[i].iov_base = tensors[i].data_ptr();
//         iovs[i].iov_len = tensors[i].storage().nbytes();
//     }
//     return iovs;
// }
std::unordered_set<std::string> GDSOffloader::get_backends()
{
    std::unordered_set<std::string> backends;
    backends.insert("gds");
    return backends;
}

void GDSOffloader::probe_asyncio(const std::string &backend)
{
    FILE *fp = tmpfile();
    if (!fp)
    {
        printf("Create tmpfile error: %s\n", strerror(errno));
        throw std::runtime_error("GDS probe failed\n");
    }
    try
    {
        std::unique_ptr<GDSAsyncIO> aio;
        #ifndef DISABLE_GDS
            aio.reset(new GDSAsyncIO(2));
        #else
            throw std::runtime_error("GDS is not installed\n");
        #endif

        int fd = fileno(fp);        
        const int n_loop = 5, n_len = 18;

        char text[n_loop][n_len];

        int offset = 0;
        size_t len;
        for (int i = 0; i < n_loop; i++)
        {
            len = n_len;
            aio->write(fd, text[i], len, offset, nullptr);
            offset += len;
        }
        aio->sync_write_events();

        char new_text[n_loop][n_len];
        offset = 0;
        for (int i = 0; i < n_loop; i++)
        {
            len = n_len;
            aio->read(fd, new_text[i], len, offset, nullptr);
            offset += len;
        }
        aio->sync_read_events();
        for (int i = 0; i < n_loop; i++)
        {
            for (int j = 0; j < n_len; j++)
            {
                assert(text[i][j] == new_text[i][j]);
            }
        }
        fclose(fp);
    }
    catch (...)
    {
        fclose(fp);
        throw std::runtime_error("GDS probe failed\n");
    }
}

bool GDSOffloader::probe_backend(const std::string &backend)
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

AsyncIO *GDSOffloader::create_asyncio(unsigned int n_entries, const std::string &backend)
{
    if (!probe_backend(nullptr))
        throw std::runtime_error("Backend \"" + backend + "\" is not install correctly");
    return new GDSAsyncIO(n_entries);
    throw std::runtime_error("Unsupported backend: " + backend);
}

GDSOffloader::GDSOffloader(const std::string &filename, unsigned int n_entries, const std::string &backend) : filename(filename), space_mgr(SpaceManager(0))
{
    this->aio = create_asyncio(n_entries, backend);
    this->fd = open(filename.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
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

// void GDSOffloader::sync_write(const at::Tensor &tensor, const std::string &key)
// {
//     ull offset, bytes;
//     std::tie(offset, bytes) = prepare_write(tensor, key);
//     lseek(this->fd, offset, SEEK_SET);
//     write(this->fd, tensor.data_ptr(), bytes);
// }

// void GDSOffloader::sync_read(const at::Tensor &tensor, const std::string &key)
// {
//     ull offset, bytes;
//     std::tie(offset, bytes) = prepare_read(tensor, key);
//     lseek(this->fd, offset, SEEK_SET);
//     read(this->fd, tensor.data_ptr(), bytes);
//     release(offset, bytes);
// }

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

// SpaceInfo GDSOffloader::prepare_writev(const std::vector<at::Tensor> &tensors, const std::string &key)
// {
//     ull total_bytes = 0;
//     for (const at::Tensor &tensor : tensors)
//     {
//         if (!tensor.is_contiguous() || !tensor.is_cuda())
//             throw std::runtime_error("Tensor must be contiguous and on gpu");
//         total_bytes += tensor.storage().nbytes();
//     }
//     ull offset = this->space_mgr.alloc(total_bytes);
//     SpaceInfo space_info(offset, total_bytes);
//     this->tensors_info[key] = space_info;
//     return space_info;
// }

// SpaceInfo GDSOffloader::prepare_readv(const std::vector<at::Tensor> &tensors, const std::string &key)
// {
//     ull total_bytes = 0;
//     for (const at::Tensor &tensor : tensors)
//     {
//         if (!tensor.is_contiguous() || !tensor.is_cuda())
//             throw std::runtime_error("Tensor must be contiguous and on gpu");
//         total_bytes += tensor.storage().nbytes();
//     }
//     if (this->tensors_info.find(key) == this->tensors_info.end())
//         throw std::runtime_error("Read error, tensor not found");
//     SpaceInfo space_info = this->tensors_info[key];
//     if (total_bytes != space_info.second)
//         throw std::runtime_error("Read error, tensor shape mismatch");
//     this->tensors_info.erase(key);
//     return space_info;
// }

// void GDSOffloader::async_writev(const std::vector<at::Tensor> &tensors, const std::string &key, callback_t callback)
// {
//     ull offset, bytes;
//     std::tie(offset, bytes) = prepare_writev(tensors, key);
//     iovec *iov = tensors_to_iovec(tensors);
//     this->aio->writev(this->fd, iov, tensors.size(), offset, callback);

//     this->aio->get_event(NOWAIT);
// }

// void GDSOffloader::async_readv(const std::vector<at::Tensor> &tensors, const std::string &key, callback_t callback)
// {

//     ull offset, bytes;
//     std::tie(offset, bytes) = prepare_readv(tensors, key);
//     iovec *iov = tensors_to_iovec(tensors);
//     auto fn = std::bind(&GDSOffloader::release, this, offset, bytes, callback);
//     this->aio->readv(this->fd, iov, tensors.size(), offset, fn);

//     this->aio->get_event(NOWAIT);
// }

// void GDSOffloader::sync_writev(const std::vector<at::Tensor> &tensors, const std::string &key)
// {
//     ull offset, bytes;
//     std::tie(offset, bytes) = prepare_writev(tensors, key);
//     iovec *iov = tensors_to_iovec(tensors);
//     lseek(this->fd, offset, SEEK_SET);
//     writev(this->fd, iov, tensors.size());
//     delete iov;
// }

// void GDSOffloader::sync_readv(const std::vector<at::Tensor> &tensors, const std::string &key)
// {
//     ull offset, bytes;
//     std::tie(offset, bytes) = prepare_readv(tensors, key);
//     iovec *iov = tensors_to_iovec(tensors);
//     lseek(this->fd, offset, SEEK_SET);
//     readv(this->fd, iov, tensors.size());
//     delete iov;
// }

void GDSOffloader::release(ull offset, ull bytes, callback_t callback)
{
    this->space_mgr.free(offset, bytes);
    if (callback != nullptr)
        callback();
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<GDSOffloader>(m, "GDSOffloader")
        .def(py::init<const std::string &, unsigned int, const std::string &>(), py::arg("filename"), py::arg("n_entries"), py::arg("backend") = "gds")
        .def("async_write", &GDSOffloader::async_write, py::arg("tensor"), py::arg("key"), py::arg("callback") = py::none())
        .def("async_read", &GDSOffloader::async_read, py::arg("tensor"), py::arg("key"), py::arg("callback") = py::none())
        // .def("sync_write", &GDSOffloader::sync_write, py::arg("tensor"), py::arg("key"))
        // .def("sync_read", &GDSOffloader::sync_read, py::arg("tensor"), py::arg("key"))
        .def("sync_write_events", &GDSOffloader::sync_write_events)
        .def("sync_read_events", &GDSOffloader::sync_write_events)
        .def("synchronize", &GDSOffloader::synchronize);
        // .def("async_writev", &GDSOffloader::async_writev, py::arg("tensors"), py::arg("key"), py::arg("callback") = py::none())
        // .def("async_readv", &GDSOffloader::async_readv, py::arg("tensors"), py::arg("key"), py::arg("callback") = py::none())
        // .def("sync_writev", &GDSOffloader::sync_writev, py::arg("tensors"), py::arg("key"))
        // .def("sync_readv", &GDSOffloader::sync_readv, py::arg("tensors"), py::arg("key"));
    // m.def("get_backends", get_backends);
    // m.def("probe_backend", probe_backend, py::arg("backend"));
}