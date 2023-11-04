import torch

device = torch.device('cuda:0')
dtype = torch.float32

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

def init_dtype(set_dtype=torch.float32):
    global dtype
    dtype = set_dtype
    print(f"Use dtype: {dtype}")

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).type(dtype).to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def tensor(list):
    return torch.tensor(list).type(dtype).to(device)

def create_zeros(shape):
    return torch.zeros(shape).type(dtype).to(device)

init_gpu(use_gpu=True)