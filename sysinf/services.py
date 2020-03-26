import warnings
import psutil
import gpustat
from pynvml import NVMLError

MB = 1024 * 1024


def cpu(interval=0.5):
    return psutil.cpu_percent(interval=interval, percpu=True)


def gpu():
    try:
        query = gpustat.new_query()
    except NVMLError:
        return [{
            'name': 'GPU not available',
            'temperature': 0,
            'fan_speed': 0,
            'utilization': 0,
            'memory_used': 0,
            'memory_total': 0,
            'power_draw': 0,
        }]
    return [{
        'name': gpu.name,
        'temperature': gpu.temperature,
        'fan_speed': gpu.fan_speed,
        'utilization': gpu.utilization,
        'memory_used': gpu.memory_used * MB,
        'memory_total': gpu.memory_total * MB,
        'power_draw': gpu.power_draw,
    } for gpu in query.gpus]


def memory():
    usage = psutil.virtual_memory()
    return {
        'total': usage.total,
        'used': usage.used,
        'free': usage.available
    }


def disk():
    partitions = psutil.disk_partitions(all=False)
    usage = psutil.disk_usage(partitions[0].mountpoint)
    io = psutil.disk_io_counters(perdisk=False, nowrap=True)
    return {
        'total': usage.total,
        'used': usage.used,
        'free': usage.free,
        'read_bytes': io.read_bytes,
        'write_bytes': io.write_bytes,
        'read_time': io.read_time,
        'write_time': io.write_time,
    }


def network():
    usage = psutil.net_io_counters(pernic=False, nowrap=True)
    return {
        'bytes_sent': usage.bytes_sent,
        'bytes_recv': usage.bytes_recv,
        'packets_sent': usage.packets_sent,
        'packets_recv': usage.packets_recv,
    }


def temperature():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reads = psutil.sensors_temperatures(fahrenheit=False)
        temps = []
        if isinstance(reads, dict):
            for read in reads.values():
                if isinstance(read, list):
                    temps.extend([{
                        'label': temp.label,
                        'value': temp.current
                        } for temp in read if temp.label
                    ])
        return temps
