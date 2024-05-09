import numpy as np
import torch

class MSEWithShift:
    def __init__(self):
        pass


    def calc(self, arr1, arr2, shift):
        '''
        :param arr1: Array with bigger length
        :param arr2: Array with shorter length
        :param shift: Shift for the arr1 to meet the length of arr2
        :return: MSE
        '''
        arr1_trimmed = arr1[shift:len(arr2) + shift]
        arr2_trimmed = arr2
        mse = torch.mean(torch.square(arr1_trimmed - arr2_trimmed))
        return mse
