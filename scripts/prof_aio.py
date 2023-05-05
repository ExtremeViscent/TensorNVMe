import torch
import tensornvme
from tqdm import tqdm, trange
torch.cuda.set_device(1)


# import nvidia_dlprof_pytorch_nvtx
# nvidia_dlprof_pytorch_nvtx.init()

offload_aio = tensornvme.DiskOffloader('./',backend='aio')

def test_loop(num_iter):
    for i in trange(num_iter):
        gpu_tensor = torch.cuda.FloatTensor(1000 * 1000)
        gpu_tensor.fill_(1.0)
        cpu_tensor = gpu_tensor.cpu()
        offload_aio.async_write(cpu_tensor)
        offload_aio.synchronize()
        offload_aio.async_read(cpu_tensor)
        offload_aio.synchronize()
        gpu_tensor = cpu_tensor.cuda()



test_loop(1000)