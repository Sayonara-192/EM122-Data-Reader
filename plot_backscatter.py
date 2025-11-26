import sys
import numpy as np
import matplotlib.pyplot as plt
import read_em_122


def plot_backscatter_map(data):
    """
    绘制反向散射强度 (Backscatter) 2D 平面图
    """
    print("正在准备绘图数据...")

    lat = data["lat"]
    lon = data["lon"]
    bs = data["bs"]

    # --- 数据统计 ---
    print(f"Backscatter 统计:")
    print(f"  Min: {np.min(bs):.2f} dB")
    print(f"  Max: {np.max(bs):.2f} dB")
    print(f"  Mean: {np.mean(bs):.2f} dB")

    # --- 绘图设置 ---
    plt.figure(figsize=(12, 8), dpi=100)

    # 使用 scatter 绘制点云
    # c=bs: 根据反向散射强度着色
    # cmap='gray': 使用灰度图。
    # s=0.5: 点的大小。
    # alpha=0.8: 透明度，有助于观察重叠区域。
    scatter = plt.scatter(lon, lat, c=bs, cmap='gray', s=0.5, alpha=0.8)

    # 添加颜色条
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Backscatter Strength (dB)', rotation=270, labelpad=15)

    # 设置标题和轴标签
    plt.title(f"EM122 Backscatter Map (Points: {len(bs)})", fontsize=14)
    plt.xlabel("Longitude (degrees)", fontsize=12)
    plt.ylabel("Latitude (degrees)", fontsize=12)

    # 保持地理比例 (Aspect Ratio)
    # 这对于地图非常重要，否则圆形看起来会像椭圆
    plt.axis('equal')

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.4)

    # 自动调整布局
    plt.tight_layout()

    print("绘图完成，窗口即将弹出...")
    plt.show()


if __name__ == "__main__":
    # 1. 获取输入文件
    input_file = r"D:\MBES\data\em122_km1128\0000_20111001_182048_KM.all.mb58"

    # 2. 读取数据
    print(f"正在读取文件: {input_file}")
    data_dict = read_em_122.read_em122_wgs84(input_file)

    # 3. 绘图
    if data_dict is not None:
        plot_backscatter_map(data_dict)
    else:
        print("数据读取失败。")