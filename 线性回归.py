import numpy as np
import matplotlib.pyplot as plt


#制造假数据
np.random.seed(42)
x_data = np.linspace(-5,5,100)#起始值是-5 结束值是5 中间随机均匀生成100个数 linspace确保均匀分布
noise = np.random.normal(0,1,100)#随机噪音   0表示分布的中心点 1表示波动范围 100表示样本数量
y_data = 2 * x_data + 1 + noise  #真实公式加噪音


#初始化参数
w = 0.0  #斜率
b = 0.0  #截距
Ir = 0.01  #学习率
epochs = 100  #训练次数
n_samples = len(x_data) #获取x_data数组的长度 样本数量


#记录loss,方便最后画图看效果
loss_history = [] #记录误差

print("开始训练")

#循环训练----核心

for epoch in range(epochs):  #对序列内的每一个数字都执行下面缩进的代码，range（epochs)表示生成数字序列
    #----A.向前传播（猜答案）---
    #用当前的w和b算出预测值 y_pred
    y_pred = w * x_data + b

    #----B。计算损失（算误差）---
    #均方误差（MSE):(预测值-真实值)的平方，再求平均
    loss = np.mean((y_pred - y_data)**2) #np.mean表示除以样本总数
    loss_history.append(loss) #将误差记录到列表末尾

    #----C。计算梯度（找方向）---
    #链式求导法则
    #dw：损失对w的导数 -》跟x有关
    dw = (2/n_samples)*np.sum((y_pred - y_data)*x_data) #2是因为对误差（误差公式里有平方）求导会下来一个2

    #db:损失对w的导数 -》与x无关，直接求和
    db = (2/n_samples)*np.sum(y_pred - y_data)

    #----D。更新参数（迈步子）---
    #沿着梯度的反方向走，让loss变小
    w = w - Ir * dw
    b = b - Ir * db

    #每打印10步看一看进度
    if epoch % 10 == 0:
        print(f"Epoch{epoch}:Loss={loss:.4f},w={w:.2f},b={b:.2f}")


    print(f"\n训练结束！")
    print(f"最终模型：y = {w:.2f}x + {b:.2f}")
    print(f"(真实值是y = 2.00x + 1.00)")

    #==============================================================================================
    #展示画图结果
    #==============================================================================================
    plt.figure(figsize=(12,5))


    #左图 拟合效果
    plt.subplot(1,2,1)  #在一个图形中创建多个子图  画布分为1行 2列 1表示左侧位置
    plt.scatter(x_data,y_data,color="blue",alpha=0.5,label="Data") #0.5表示透明度为0.5
    plt.plot(x_data,w*x_data + b,color="red",linewidth=2,label="Prediction") #线宽为2，数越大线越粗
    plt.title("Linear Regression Result")
    plt.legend()  #添加图例

    #右图  Loss下降曲线
    plt.subplot(1,2,2)
    plt.plot(loss_history,color='green')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig("result.png")
    print("图片已保存为 result.png")