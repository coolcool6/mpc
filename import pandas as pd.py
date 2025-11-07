import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
import seaborn as sns
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def smooth_cylinder_pressure(data):
    """
    使用多种滤波方法平滑缸压数据
    
    参数:
    data: 包含曲轴转角、纯柴油缸压和双燃料缸压的DataFrame
    
    返回:
    包含原始数据和滤波后数据的DataFrame
    """
    
    # 创建结果DataFrame
    result = data.copy()
    
    # 方法1: Savitzky-Golay滤波器 (保留信号特征的同时平滑)
    result['纯柴油缸压_SG'] = savgol_filter(data['纯柴油缸压'], window_length=21, polyorder=3)
    result['双燃料缸压_SG'] = savgol_filter(data['双燃料缸压'], window_length=21, polyorder=3)
    
    # 方法2: 移动平均滤波
    window_size = 15
    result['纯柴油缸压_MA'] = data['纯柴油缸压'].rolling(window=window_size, center=True).mean()
    result['双燃料缸压_MA'] = data['双燃料缸压'].rolling(window=window_size, center=True).mean()
    
    # 方法3: 巴特沃斯低通滤波器
    def butter_lowpass_filter(data, cutoff_freq=0.1, fs=5, order=4):
        """巴特沃斯低通滤波器"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    # 应用巴特沃斯滤波器
    result['纯柴油缸压_BW'] = butter_lowpass_filter(data['纯柴油缸压'])
    result['双燃料缸压_BW'] = butter_lowpass_filter(data['双燃料缸压'])
    
    # 方法4: 指数加权移动平均
    result['纯柴油缸压_EWMA'] = data['纯柴油缸压'].ewm(span=10).mean()
    result['双燃料缸压_EWMA'] = data['双燃料缸压'].ewm(span=10).mean()
    
    # 填充边缘的NaN值
    result = result.fillna(method='bfill').fillna(method='ffill')
    
    return result

def calculate_smoothness_metrics(original, smoothed):
    """
    计算平滑效果的指标
    """
    # 计算标准差减少比例
    std_original = np.std(original)
    std_smoothed = np.std(smoothed)
    std_reduction = (std_original - std_smoothed) / std_original * 100
    
    # 计算均方根误差
    rmse = np.sqrt(np.mean((original - smoothed) ** 2))
    
    return {
        '原始标准差': std_original,
        '平滑后标准差': std_smoothed,
        '标准差减少(%)': std_reduction,
        'RMSE': rmse
    }

def plot_comparison(result):
    """
    绘制原始数据与平滑后数据的对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 纯柴油缸压对比
    axes[0, 0].plot(result['曲轴转角'], result['纯柴油缸压'], 'b-', alpha=0.7, label='原始数据', linewidth=1)
    axes[0, 0].plot(result['曲轴转角'], result['纯柴油缸压_SG'], 'r-', label='SG滤波', linewidth=2)
    axes[0, 0].plot(result['曲轴转角'], result['纯柴油缸压_MA'], 'g-', label='移动平均', linewidth=2)
    axes[0, 0].set_title('纯柴油缸压 - 滤波效果对比')
    axes[0, 0].set_xlabel('曲轴转角 (度)')
    axes[0, 0].set_ylabel('缸压')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 双燃料缸压对比
    axes[0, 1].plot(result['曲轴转角'], result['双燃料缸压'], 'b-', alpha=0.7, label='原始数据', linewidth=1)
    axes[0, 1].plot(result['曲轴转角'], result['双燃料缸压_SG'], 'r-', label='SG滤波', linewidth=2)
    axes[0, 1].plot(result['曲轴转角'], result['双燃料缸压_MA'], 'g-', label='移动平均', linewidth=2)
    axes[0, 1].set_title('双燃料缸压 - 滤波效果对比')
    axes[0, 1].set_xlabel('曲轴转角 (度)')
    axes[0, 1].set_ylabel('缸压')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 所有滤波方法对比 - 纯柴油
    axes[1, 0].plot(result['曲轴转角'], result['纯柴油缸压'], 'k-', alpha=0.3, label='原始数据', linewidth=1)
    axes[1, 0].plot(result['曲轴转角'], result['纯柴油缸压_SG'], 'r-', label='SG滤波', linewidth=2)
    axes[1, 0].plot(result['曲轴转角'], result['纯柴油缸压_MA'], 'g-', label='移动平均', linewidth=2)
    axes[1, 0].plot(result['曲轴转角'], result['纯柴油缸压_BW'], 'b-', label='巴特沃斯', linewidth=2)
    axes[1, 0].plot(result['曲轴转角'], result['纯柴油缸压_EWMA'], 'm-', label='指数加权', linewidth=2)
    axes[1, 0].set_title('纯柴油缸压 - 所有滤波方法')
    axes[1, 0].set_xlabel('曲轴转角 (度)')
    axes[1, 0].set_ylabel('缸压')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 所有滤波方法对比 - 双燃料
    axes[1, 1].plot(result['曲轴转角'], result['双燃料缸压'], 'k-', alpha=0.3, label='原始数据', linewidth=1)
    axes[1, 1].plot(result['曲轴转角'], result['双燃料缸压_SG'], 'r-', label='SG滤波', linewidth=2)
    axes[1, 1].plot(result['曲轴转角'], result['双燃料缸压_MA'], 'g-', label='移动平均', linewidth=2)
    axes[1, 1].plot(result['曲轴转角'], result['双燃料缸压_BW'], 'b-', label='巴特沃斯', linewidth=2)
    axes[1, 1].plot(result['曲轴转角'], result['双燃料缸压_EWMA'], 'm-', label='指数加权', linewidth=2)
    axes[1, 1].set_title('双燃料缸压 - 所有滤波方法')
    axes[1, 1].set_xlabel('曲轴转角 (度)')
    axes[1, 1].set_ylabel('缸压')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数 - 处理缸压数据
    """
    # 从Excel文件读取数据（请确保文件路径正确）
    try:
        # 如果数据在Excel文件中
        df = pd.read_excel('pressure.xlsx')
    except FileNotFoundError as e:
        # 如果数据文件不存在，提供更详细的错误信息
        print("错误：找不到pressure.xlsx文件")
        print(f"当前工作目录: {os.getcwd()}")
        print("当前目录下的文件列表:")
        for file in os.listdir('.'):
            print(f"  {file}")
        
        # 检查是否有类似名称的文件
        excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
        if excel_files:
            print("\n提示：找到以下Excel文件，检查文件名是否正确:")
            for file in excel_files:
                print(f"  {file}")
        else:
            print("\n提示：当前目录下没有找到Excel文件")
        return
    except Exception as e:
        # 其他异常情况
        print(f"读取文件时发生错误: {e}")
        print(f"当前工作目录: {os.getcwd()}")
        print("请确保文件存在且格式正确")
        return
    
    # 确保列名正确
    expected_columns = ['曲轴转角', '纯柴油缸压', '双燃料缸压']
    if not all(col in df.columns for col in expected_columns):
        print("数据列名不匹配，请检查数据格式")
        print("当前数据列名:", list(df.columns))
        print("期望的列名:", expected_columns)
        return
    
    print("原始数据统计:")
    print(df.describe())
    
    # 应用滤波
    print("\n正在应用滤波...")
    smoothed_data = smooth_cylinder_pressure(df)
    
    # 计算平滑效果指标
    print("\n纯柴油缸压平滑效果:")
    diesel_metrics = calculate_smoothness_metrics(
        smoothed_data['纯柴油缸压'], 
        smoothed_data['纯柴油缸压_SG']
    )
    for key, value in diesel_metrics.items():
        print(f"{key}: {value:.6f}")
    
    print("\n双燃料缸压平滑效果:")
    dual_fuel_metrics = calculate_smoothness_metrics(
        smoothed_data['双燃料缸压'], 
        smoothed_data['双燃料缸压_SG']
    )
    for key, value in dual_fuel_metrics.items():
        print(f"{key}: {value:.6f}")
    
    # 绘制对比图
    print("\n生成对比图表...")
    plot_comparison(smoothed_data)
    
    # 保存结果到Excel文件
    output_file = 'smoothed_pressure_data.xlsx'
    smoothed_data.to_excel(output_file, index=False)
    print(f"\n平滑后的数据已保存到: {output_file}")
    
    # 推荐最佳滤波方法
    print("\n推荐滤波方法:")
    print("1. Savitzky-Golay滤波器 - 保留峰值特征较好")
    print("2. 移动平均滤波 - 简单有效")
    print("3. 巴特沃斯滤波器 - 专业信号处理")
    
    return smoothed_data

# 运行主程序
if __name__ == "__main__":
    result_data = main()