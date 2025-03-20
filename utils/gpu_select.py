import pynvml
def get_least_loaded_gpu_id():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            print("No GPU available.")
            return None

        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                'id': i,
                'load': util.gpu,
                'memoryUsed': mem_info.used / 1024 / 1024  # Convert bytes to MB
            })

        # 按照负载升序排序
        sorted_gpus = sorted(gpus, key=lambda gpu: (gpu['load'], gpu['memoryUsed']))

        # 选择负载最小的GPU
        least_loaded_gpu = sorted_gpus[0]

        print(
            f"Selecting GPU {least_loaded_gpu['id']} with load {least_loaded_gpu['load']}% and memory used {least_loaded_gpu['memoryUsed']:.2f}MB")

        return str(least_loaded_gpu['id'])

    except pynvml.NVMLError as error:
        print(f"NVIDIA Management Library Error: {error}")
        return None
    finally:
        pynvml.nvmlShutdown()