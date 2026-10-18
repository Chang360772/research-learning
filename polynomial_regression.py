import numpy as np
import matplotlib
matplotlib.use('Agg') #非交互式画图模式，代码会把图生成并保存成文件，不会一直弹出
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures #制造多项式特征，把x变为x,x^2,x^3....
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #把数据切成训练集和测试集
from sklearn.metrics import mean_squared_error, r2_score #均方误差和R^2分数
import os #操作系统接口库

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #解决负号显示问题，否则负号可能显示为方块

output_dir = os.path.dirname(os.path.abspath(__file__)) #获取文件路径

# 1. 生成带噪声的非线性数据
np.random.seed(1)
X = np.sort(np.random.uniform(-3, 3, size=100)).reshape(-1, 1) #sort表示从小到大排序 reshape表示几行几列，这里的-1表示是自动计算几行，sklearn输入的矩阵必须是二维的
y = np.sin(X).ravel() + np.random.normal(0, 0.2, size=100) #均值为0，标准差为2的高斯噪声

# 2. 划分 Training Set 和 Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #%70用来训练，%30用来测试

# 3. 分别训练 Degree = 1, 2, 5, 15 的 Polynomial Regression
degrees = [1, 2, 5, 15] #x^degrees,越复杂越容易过拟合
fig, axes = plt.subplots(2, 2, figsize=(14, 10)) #创建一个有2x2个子图的图形窗口，figsize14英寸宽，10英寸高

for idx, degree in enumerate(degrees): #开始循环，idx即index提供索引
    # 多项式特征转换
    poly = PolynomialFeatures(degree=degree) #创建转换器对象
    X_train_poly = poly.fit_transform(X_train) #fit分析训练集的特征，transform将训练集数据转化为多项式特征矩阵
    X_test_poly = poly.transform(X_test) #

    # 训练线性回归模型（在多项式特征上）
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 预测
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # 计算误差
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    #========================================================================================================================
    
    # 绘制拟合曲线
    ax = axes[idx // 2][idx % 2] #自动排版
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    ax.scatter(X_train, y_train, color='blue', s=30, label='Training data', zorder=3) #s=30表示点的大小，数值越大点越大，zorder代表层级，数越大越高
    ax.scatter(X_test, y_test, color='orange', s=30, label='Test data', zorder=3)
    ax.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree={degree} fit')
    ax.set_title(f'Polynomial Regression (Degree={degree})\n'
                 f'Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}\n'
                 f'Train R2={train_r2:.4f}, Test R2={test_r2:.4f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(loc='upper right') #显示图例

# 4. 绘制 Training/Error 对比图
fig2, ax2 = plt.subplots(figsize=(10, 6))
train_errors = [] #创建空表格
test_errors = []
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
#绘制折线
ax2.plot(degrees, train_errors, 'o-', label='Training MSE', color='blue', linewidth=2)
ax2.plot(degrees, test_errors, 's-', label='Test MSE', color='red', linewidth=2)
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Training vs Test Error vs Model Complexity')
ax2.legend(loc='upper right')
ax2.set_xticks(degrees)

plt.tight_layout() #自动调节子图间距，防止标签被截断
fig_path = os.path.join(output_dir, 'polynomial_regression_results.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Polynomial Regression 结果图已保存: {fig_path}")

# 打印详细结果 
print("\n=== Polynomial Regression 结果汇总 ===")
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Degree={degree}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")

print("\n分析:")
print("- Degree=1 (线性模型): Underfitting, 无法捕捉 sin(x) 的非线性关系")
print("- Degree=2: 初步捕捉非线性, 但仍有偏差")
print("- Degree=5: 较好的拟合, Train/Test误差接近")
print("- Degree=15: Overfitting, Train误差极低但Test误差很高, 模型复杂度太高")