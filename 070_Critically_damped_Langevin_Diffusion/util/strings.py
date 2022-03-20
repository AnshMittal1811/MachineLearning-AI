# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

def string_to_list(string, delimiter=',', type=int):
    return list(map(type, string.split(delimiter)))


def string_to_tuple(string, delimiter=',', type=int):
    return tuple(string_to_list(string, delimiter, type))
