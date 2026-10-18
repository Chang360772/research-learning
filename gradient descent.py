#初始设置
import numpy as np
x = 1
Ir = 0.0001

print(f"Initial x:{x}")

for step in range(1000):
    #计算梯度(导数)：dy/dx = 2(x-3)
    gradient = 1.0 / (x*np.log(2))


    #更新参数：x_new = x_old - Ir*gradient
    x = x - Ir * gradient

    print(f"Step {step}:x = {x:.4f},gradient = {gradient:.4f}")