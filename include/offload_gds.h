#pragma once

#include <ATen/ATen.h>
#include "space_mgr.h"
#include "gds.h"

class GDSOffloader
{
public:
    GDSOffloader(const std::string &filename, unsigned int n_entries, const std::string &backend);
    SpaceInfo prepare_write(const at::Tensor &tensor, const std::string &key);
    SpaceInfo prepare_read(const at::Tensor &tensor, const std::string &key);
    void async_write(const at::Tensor &tensor, const std::string &key, callback_t callback = nullptr);
    void async_read(const at::Tensor &tensor, const std::string &key, callback_t callback = nullptr);
    // void sync_write(const at::Tensor &tensor, const std::string &key);
    // void sync_read(const at::Tensor &tensor, const std::string &key);
    void sync_write_events();
    void sync_read_events();
    void synchronize();
    std::unordered_set<std::string> get_backends(); 
    void probe_asyncio(const std::string &backend);
    bool probe_backend(const std::string &backend);
    AsyncIO *create_asyncio(unsigned int n_entries, const std::string &backend);
    ~GDSOffloader();
    // SpaceInfo prepare_writev(const std::vector<at::Tensor> &tensors, const std::string &key);
    // SpaceInfo prepare_readv(const std::vector<at::Tensor> &tensors, const std::string &key);
    // void async_writev(const std::vector<at::Tensor> &tensors, const std::string &key, callback_t callback = nullptr);
    // void async_readv(const std::vector<at::Tensor> &tensors, const std::string &key, callback_t callback = nullptr);
    // void sync_writev(const std::vector<at::Tensor> &tensors, const std::string &key);
    // void sync_readv(const std::vector<at::Tensor> &tensors, const std::string &key);
private:
    const std::string filename;
    int fd;
    AsyncIO *aio;
    SpaceManager space_mgr;
    std::unordered_map<std::string, SpaceInfo> tensors_info;

    void release(ull offset, ull bytes, callback_t callback = nullptr);
};