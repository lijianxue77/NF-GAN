import torch
import numpy as np
import time
import math
import pandas as pd


def binary_encode(value, bits):
    """convert a numerical value into binary representation"""
    encoding = []
    while value != 0:
        encoding.append(value % 2)
        value = value // 2
    if len(encoding) > bits:
        return None
    for _ in range(bits - len(encoding)):
        encoding.append(0)
    encoding.reverse()
    return encoding


def binary_decode(encoding):
    """convert binary representation to a numerical value"""
    if len(encoding) == 0:
        return None
    res = 0
    for i in range(len(encoding)):
        res *= 2
        res += encoding[i]
    return res


def float2binary(value):
    """convert float to binary"""
    out = []
    for i in range(len(value)):
        if value[i] < 0.5:
            out.append(0)
        else:
            out.append(1)
    return out


def get_one_hot(target, num_classes):
    one_hot = [0.0 for _ in range(num_classes)]
    one_hot[target] = 1.0
    return one_hot


def from_one_hot(one_hot):
    for c in range(0, len(one_hot)):
        if c == 1.0 or c == 1:
            return c
    return None


def build_input_indices(dataset, traffic_data_dict):
    """return a LongTensor of indices to represent the network traffic"""
    input_indices = []
    cur = 0
    for feature in dataset.categorical_features:
        feature_vocab_len = dataset.feature_bits_dict[feature]
        feature_idx = np.where(dataset.categorical_values[feature] == traffic_data_dict[feature])[0][0]
        input_indices.append(cur + feature_idx)
        cur += feature_vocab_len

    return torch.LongTensor(input_indices)


def decode_feature_indices(dataset, feature_indices):
    """decode network traffic from a LongTensor of indices"""
    output_values = {}
    feature_indices = feature_indices.data.numpy()  # convert a tensor to a list
    sorted(feature_indices)
    cur = 0

    for i, feature in enumerate(dataset.categorical_features):
        feature_vocab_len = dataset.feature_bits_dict[feature]
        feature_idx = feature_indices[i] - cur
        # This random mechanism need to be changed and may only retain some valid network traffics
        if feature_idx < 0 or feature_idx >= feature_vocab_len:
            return None
        output_values[feature] = dataset.categorical_values[feature][feature_idx]
        cur += feature_vocab_len

    return output_values


def ipaddr2binary(ipaddr):
    """convert IP address to binary representation"""
    ip_bits = []
    ip_segments = ipaddr.split('.')
    for segment in ip_segments:
        for i in binary_encode(int(segment), 8):
            ip_bits.append(i)
    return ip_bits


def ipaddr2numerical(ipaddr):
    """convert ip address to a numerical value"""
    ip_bits = ipaddr2binary(ipaddr)
    return binary_decode(ip_bits)


def ip_transform(ips):
    """transform ip series to a numerical series"""
    return [ipaddr2numerical(val) for val in ips]


def time_since(start):
    now = time.time()
    dur = now - start
    m = math.floor(dur / 60)
    s = dur - m * 60
    return '%d m, %d s' % (m, s)


def judge_protocol_port(row, service_list, protocol_service_dict, service_port_dict):
    protocol = row['proto']
    service = row['service']
    port = row['dsport']

    # check protocol mapping to service
    if (protocol == 'tcp' and service in protocol_service_dict['udp']) or (
            protocol == 'udp' and service in protocol_service_dict['tcp']):
        return False

    # check service mapping to dsport
    if service in service_list and port not in service_port_dict[service]:
        return False

    return True


def build_traffic_dfs(path_list, use_target_list, r_columns, f_columns):
    df_list = []
    for path, use_target in zip(path_list, use_target_list):
        if use_target:
            columns = f_columns
        else:
            columns = r_columns
        df_r = pd.read_csv(path, names=columns, header=None, index_col=None)
        df_list.append(df_r)
    return df_list


def build_losses_dfs(path_d, path_g):
    df_r = pd.read_csv(path_d, index_col=False)
    df_f = pd.read_csv(path_g, index_col=False)
    return df_r, df_f


def get_feature_distribution_vectors(d_r, d_f, feature):
    v_r = d_r.groupby(feature).size() / len(d_r)
    v_f = d_f.groupby(feature).size() / len(d_f)
    d_uq = d_r.append(d_f)[feature].unique()
    r_dict = {}
    f_dict = {}
    for val in d_uq:
        r_dict[val] = 0.0
        f_dict[val] = 0.0
        if val in v_r.index:
            r_dict[val] = v_r.loc[val]
        if val in v_f.index:
            f_dict[val] = v_f.loc[val]
    v_r = list(r_dict.values())
    v_f = list(f_dict.values())
    return v_r, v_f


def record_losses(path, losses):
    with open(path, 'w') as file:
        for loss in losses:
            file.write(str(loss))
            file.write('\n')