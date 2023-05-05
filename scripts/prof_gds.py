import torch
import tensornvme
from tqdm import tqdm, trange
torch.cuda.set_device(1)


# import nvidia_dlprof_pytorch_nvtx
# nvidia_dlprof_pytorch_nvtx.init()

offload_gds = tensornvme.DiskOffloader('/data/gds',backend='gds')

def test_loop(num_iter):
    for i in trange(num_iter):
        # with torch.autograd.profiler.emit_nvtx():
        gpu_tensor = torch.cuda.FloatTensor(1000 * 1000)
        gpu_tensor.fill_(1.0)
        offload_gds.async_write(gpu_tensor)
        offload_gds.synchronize()
        offload_gds.async_read(gpu_tensor)
        offload_gds.synchronize()

test_loop(1000)