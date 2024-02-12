def mean(data_list):
    return sum(data_list) / len(data_list)

def median(data_list):
    sorted_data = sorted(data_list)
    length = len(sorted_data)
    if length % 2 == 0:
        return (sorted_data[length // 2 - 1] + sorted_data[length // 2]) / 2
    else:
        return sorted_data[length // 2]

def mode(data_list):
    count_dict = {}
    for data in data_list:
        if data in count_dict:
            count_dict[data] += 1
        else:
            count_dict[data] = 1
    max_count = max(count_dict.values())
    mode_list = [key for key, value in count_dict.items() if value == max_count]
    return mode_list[0] if len(mode_list) == 1 else mode_list
