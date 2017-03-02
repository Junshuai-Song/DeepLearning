#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:19:45 2017

@author: songjs
"""


# 数值计算精度损失：一般令输入数据归一化、W从均值为0、sigma较小的高斯分布中初始化
def testPre():
    t = 1e9
    for i in range(1000000):
        t += 1e-6
        print(t-1e9)
        # 0.953...
