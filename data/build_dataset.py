import time
import numpy as np
import pandas as pd
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

net_dtype = np.dtype([
    ("day", "i1"), ("hours", "i2"), ("minutes", "i2"), ("seconds", "i2"), ("milliseconds since start", "i4")
    , ("bytes", "i4"), ("Transfer time", "i4"), ("throughput", "f4")
])
refer_time = "1970-01-01 08:00:00"


class bandwidth_entry:
    # Number of milliseconds since epoch;
    day = -1
    hour = -1
    minutes = -1
    sec = -1
    # Number of milliseconds since start of experiment;
    time_since_start = -1
    # Number of bytes received since last datapoint;
    num_bytes = -1
    # Number of milliseconds since last datapoint.
    num_msec = -1
    #  the average throughput in the last interval
    throughput = -1.0

    def __str__(self):
        return (
                "day:%d hours:%d minutes:%d seconds:%d time_since_start:%d num_bytes:%d num_msec:%d throughput:%f"
                % (
                    self.day, self.hour, self.minutes, self.sec, self.time_since_start, self.num_bytes, self.num_msec,
                    self.throughput
                ))

    def numpy(self):
        return np.array(
            (self.day, self.hour, self.minutes, self.sec, self.time_since_start, self.num_bytes, self.num_msec,
             self.throughput
             ), dtype=net_dtype
        )


def read_file(filename):
    rows = open(filename, 'r').readlines()
    data = [[float(ti) for ti in line.strip().split(' ')] for line in rows]
    return data


def get_time(endtime, begin_time=0):
    timeArray = time.strptime(refer_time, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(timeArray)
    dt = endtime + timestamp
    dt = time.localtime(dt / 1000)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", dt)
    dt = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    return dt.tm_wday, dt.tm_hour, dt.tm_min, dt.tm_sec
    # tm_year=2021, tm_mon=5, tm_mday=26, tm_hour=23, tm_min=0, tm_sec=0, tm_wday=2, tm_yday=146, tm_isdst=-1


def build_dataset():
    bandwidth_info_list = []
    bandwidth_list = []

    with open('E:/Net_Traffic_Predict/data/logs_all/report_foot_0001.log', 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)
        for idx, net_value in enumerate(data.values):
            entry = bandwidth_entry()
            message = net_value[0].split(" ")
            entry.day, entry.hour, entry.minutes, entry.sec = get_time(int(message[0]))
            entry.time_since_start = int(message[1])
            entry.num_bytes = int(message[4])
            entry.num_msec = int(message[5])
            entry.throughput = (entry.num_bytes / entry.num_msec) / 1024  # b/s

            bandwidth_info_list.append(entry.numpy())

        print(len(bandwidth_info_list))
        netinfo_array = np.array(bandwidth_info_list, dtype=net_dtype)

        np.save("e:/Net_Traffic_Predict/data/bandwidth_trace.npy", netinfo_array, allow_pickle=True, fix_imports=True)
        print(netinfo_array.shape)



def sliding_window(dataset, sequence_length):
    x, y = [], []
    for i in range(len(dataset) - sequence_length):
        if i + sequence_length < len(dataset):
            x.append(dataset[i:i + sequence_length, :])
            y.append(dataset[i + sequence_length, 7])

    return np.array(x), np.array(y).reshape((-1, 1))


def build_bandwidth_dataset():
    bandwidth_data = np.load("e:/Net_Traffic_Predict/data/bandwidth_trace.npy", mmap_mode=None, allow_pickle=True,
                             fix_imports=True)
    train_set, test_set = train_test_split(bandwidth_data, train_size=0.8, random_state=0)
    train_set = normalization(np.array(train_set.tolist(), np.float32))
    test_set = normalization(np.array(test_set.tolist(), np.float32))
    sliding_window(train_set, sequence_length=10)
    train_x, train_y = sliding_window(train_set, sequence_length=10)
    test_x, test_y = sliding_window(test_set, sequence_length=10)

    return train_x, train_y, test_x, test_y


def normalization(data):
    # data[0]
    # data[:, 0] = data[:, 0] / 1000
    data[:, 1] = data[:, 1] / 60
    data[:, 2] = (data[:, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))
    data[:, 3] = (data[:, 3] - np.min(data[:, 3])) / (np.max(data[:, 3]) - np.min(data[:, 3]))
    data[:, 4] = (data[:, 4] - np.min(data[:, 4])) / (np.max(data[:, 4]) - np.min(data[:, 4]))
    data[:, 5] = (data[:, 5] - np.min(data[:, 5])) / (np.max(data[:, 5]) - np.min(data[:, 5]))
    data[:, 6] = (data[:, 6] - np.min(data[:, 6])) / (np.max(data[:, 6]) - np.min(data[:, 6]))
    data[:, 7] = (data[:, 7] - np.min(data[:, 7])) / (np.max(data[:, 7]) - np.min(data[:, 7]))
    return data
