# 此文件参考了 [guardiangeomatics] 的代码，项目地址：https://github.com/guardiangeomatics/pyall
import sys
import os
import time
import numpy as np
import pyall
import geodetic
import timeseries


def read_em122_wgs84(filename):
    """
    读取 EM122 .all 文件。
    功能：
    1. 提取深度(Z)、反向散射强度(BS)。
    2. 计算波束指向角(Angle)。
    3. 基于导航数据和大地测量公式，将相对偏移(Across/Along)转换为绝对坐标(WGS84 Lat/Lon)。
    4. 返回结构化的字典数据。
    """
    if not os.path.isfile(filename):
        print(f"文件未找到: {filename}")
        return None

    # 初始化读取器
    r = pyall.allreader(filename)
    print(f"正在处理文件: {filename}")

    start_time = time.time()

    # --- 第一步：预读取导航数据 (Position) ---
    print("正在加载导航数据以构建轨迹插值模型...")
    # loadnavigation 返回列表: [[timestamp, lat, lon], ...]
    navigation = r.loadnavigation()

    if not navigation:
        print("错误: 文件中未找到导航数据 (P Datagram)，无法进行坐标转换。")
        r.close()
        return None

    # 将导航数据转换为 numpy 数组
    nav_arr = np.array(navigation)

    # 创建时间序列插值器，用于获取任意时刻的船位
    # nav_arr[:, 0] 是时间戳, [:, 1] 是纬度, [:, 2] 是经度
    ts_lat = timeseries.ctimeSeries(nav_arr[:, 0], nav_arr[:, 1])
    ts_lon = timeseries.ctimeSeries(nav_arr[:, 0], nav_arr[:, 2])

    print(f"导航数据加载完成，共 {len(nav_arr)} 个定位点。")

    # --- 第二步：重置文件指针，开始读取深度数据 ---
    r.rewind()

    # 初始化数据容器 (用于分批收集)
    list_lat, list_lon, list_z = [], [], []
    list_bs, list_ang = [], []

    # 初始化索引容器
    list_beam_counts = []
    list_ping_times = []

    ping_count = 0

    print("开始读取并转换深度数据...")

    while r.moredata():
        # 读取数据报头
        typeofdatagram, datagram = r.readdatagram()

        # 处理 XYZ 88 (深度) 数据报
        if typeofdatagram == 'X':
            datagram.read()
            ping_count += 1

            # 1. 获取并转换时间戳
            dt = r.to_datetime(datagram.recorddate, datagram.time)
            ts = pyall.to_timestamp(dt)

            # 2. 插值获取当前 Ping 发射时刻船只的参考位置 (WGS84)
            ref_lat = ts_lat.getValueAt(ts)
            ref_lon = ts_lon.getValueAt(ts)

            # 如果插值失败（如时间超出范围），跳过此 Ping
            if ref_lat is None or ref_lon is None:
                continue

            # 3. 获取航向 (Heading)
            heading = datagram.heading

            # 4. 获取相对偏移数据 (转换为 numpy 数组)
            p_across = np.array(datagram.acrosstrackdistance)  # 横向距离 (Y)
            p_along = np.array(datagram.alongtrackdistance)  # 纵向距离 (X)
            p_z = np.array(datagram.depth)  # 深度 (Z)
            p_bs = np.array(datagram.reflectivity)  # 强度 (BS)

            # 5. 计算波束指向角 (Beam Angle)
            # 公式: angle = arctan(横向距离 / 深度)
            # 结果单位: 度 (degrees)
            p_ang = np.degrees(np.arctan2(p_across, p_z))

            # 6. 逐个波束进行坐标转换 (Relative -> WGS84)
            current_ping_lats = []
            current_ping_lons = []

            for i in range(len(p_z)):
                # geodetic 库计算：根据参考点、航向、纵向偏移(dy)、横向偏移(dx) 计算目标点经纬度
                # 注意参数顺序和对应关系：dx -> across, dy -> along
                b_lon, b_lat = geodetic.calculateGeographicalPositionFromBearingDxDy(
                    ref_lon,
                    ref_lat,
                    heading,
                    p_across[i],  # dx
                    p_along[i]  # dy
                )
                current_ping_lats.append(b_lat)
                current_ping_lons.append(b_lon)

            # 7. 收集数据到列表
            list_lat.append(np.array(current_ping_lats))
            list_lon.append(np.array(current_ping_lons))
            list_z.append(p_z)
            list_bs.append(p_bs)
            list_ang.append(p_ang)

            # 收集元数据
            list_beam_counts.append(len(p_z))
            list_ping_times.append(ts)

            # 进度提示 (每50个Ping刷新一次)
            if ping_count % 50 == 0:
                sys.stdout.write(f"\r已处理 Ping 数: {ping_count}")
                sys.stdout.flush()

    sys.stdout.write(f"\r已处理 Ping 数: {ping_count}")
    sys.stdout.flush()

    r.close()

    if ping_count == 0:
        print("\n警告: 文件中未找到有效的 XYZ 深度数据。")
        return None

    print(f"\n读取完成，正在构建最终结构化数组...")

    # --- 第三步：数据扁平化 (Flattening) ---
    final_lat = np.concatenate(list_lat)
    final_lon = np.concatenate(list_lon)
    final_z = np.concatenate(list_z)
    final_bs = np.concatenate(list_bs)
    final_ang = np.concatenate(list_ang)

    # --- 第四步：构建索引数组 (Indexing) ---
    final_beam_counts = np.array(list_beam_counts, dtype=np.int32)
    final_ping_times = np.array(list_ping_times, dtype=np.float64)
    # 计算每个 Ping 的起始索引位置
    final_ping_starts = np.concatenate(([0], np.cumsum(final_beam_counts)[:-1]))

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"处理耗时: {elapsed:.2f} 秒")
    print(f"总 Ping 数: {len(final_beam_counts)}")
    print(f"总 测点数: {len(final_z)}")
    print("-" * 60)

    # 返回包含所有数据的字典
    return {
        "lat": final_lat,  # [1D Array] WGS84 纬度
        "lon": final_lon,  # [1D Array] WGS84 经度
        "z": final_z,  # [1D Array] 深度 (米)
        "bs": final_bs,  # [1D Array] 反向散射强度 (dB)
        "ang": final_ang,  # [1D Array] 波束角 (度)
        "ping_counts": final_beam_counts,  # [Index] 每个 Ping 的点数
        "ping_starts": final_ping_starts,  # [Index] 每个 Ping 的起始下标
        "ping_times": final_ping_times  # [Index] 每个 Ping 的时间戳
    }

if __name__ == "__main__":
    # 默认文件或从命令行获取
    input_file = r"D:\MBES\data\em122_km1808\0000_20180515_180733_KM_EM122.all.mb58"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    # 执行读取
    data = read_em122_wgs84(input_file)

    if data:
        print("\n数据验证:")
        print(f"Lat 范围: {data['lat'].min():.5f} ~ {data['lat'].max():.5f}")
        print(f"Lon 范围: {data['lon'].min():.5f} ~ {data['lon'].max():.5f}")
        print(f"Depth 范围: {data['z'].min():.2f} ~ {data['z'].max():.2f}")
        print(f"Angle 范围: {data['ang'].min():.2f} ~ {data['ang'].max():.2f}")

