import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import joblib
from predict import predict_performance
import matplotlib.ticker as ticker
import json
# 配置中文字体，这里使用微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 保证能够在图中显示负号
plt.rcParams['axes.unicode_minus'] = False

# 假设 num_requests_list 是已经定义好的请求次数列表
num_requests_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# 历史数据
dbo_alpha_histories = {num: val for num, val in zip(num_requests_list, [12.05, 11.699, 11.372, 11.111, 11.033, 10.762, 10.754, 10.731, 10.714, 10.673, 10.661, 10.499, 10.482, 10.451])}
dbo_reuse_histories = {num: val for num, val in zip(num_requests_list, [0.75, 0.8472222222222222, 0.8777777777777778, 0.8972222222222223, 0.9055555555555556, 0.9174603174603174, 0.9185185185185185, 0.9215277777777777, 0.9234567901234568, 0.9272222222222222, 0.9287878787878788, 0.9319444444444445, 0.9320512820512821, 0.9337301587301587])}
noreuse_dbo_alpha_histories = {num: val for num, val in zip(num_requests_list, [12.222, 12.351, 12.492, 12.589, 12.703, 12.799, 12.981, 13.099, 13.211, 13.327, 13.398, 13.3999, 13.407, 13.411])}

#英文论文图
# 条形图设置
bar_width = 0.35  # 条形宽度
index = np.arange(len(num_requests_list))  # NUM_REQUESTS 的索引
# 创建图表并设置大小
plt.figure(figsize=(10, 6))
ax1 = plt.gca()  # 获取当前轴
# 绘制第一组条形图，使用 ax1 的 y 轴
bars1 = ax1.bar(index - bar_width/2, [dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='DBSSO', color="#B47B84", zorder=3)
# 绘制第二组条形图，仍然使用 ax1 的 y 轴
bars2 = ax1.bar(index + bar_width/2, [noreuse_dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='UM-DBSSO', color="#CDBBA7", zorder=3)
ax1.set_xlabel('Number of Requests', fontsize=13)
ax1.set_ylabel('Resource Consumption per Completed Service Request', fontsize=13)
# 创建第二个 y 轴
ax2 = ax1.twinx()
# 绘制折线图，使用 ax2 的 y 轴
line1, = ax2.plot(index, [dbo_reuse_histories[num] for num in num_requests_list], label='DBSSO Resource Reuse Ratio', color="#AA5656", marker='o', zorder=4)
ax2.set_ylabel('Resource Reuse Ratio', fontsize=13)
# 设置 x 轴刻度标签
ax1.set_xticks(index)
ax1.set_xticklabels(num_requests_list)
# 添加图例
legend_elements = [bars1, bars2, line1]
ax1.legend(handles=legend_elements, loc='lower right')
# 添加网格
ax1.grid(zorder=0, color='gray',alpha = 0.3)
# 保存和展示图表
plt.savefig('J_Combined_Chart.pdf', bbox_inches='tight')
plt.savefig('J_Combined_Chart.png', bbox_inches='tight', dpi=600, pad_inches=0.0)

'''
# 条形图设置
bar_width = 0.35  # 条形宽度
index = np.arange(len(num_requests_list))  # NUM_REQUESTS 的索引
# 创建图表并设置大小
plt.figure(figsize=(10, 6))
ax1 = plt.gca()  # 获取当前轴
# 绘制第一组条形图，使用 ax1 的 y 轴
bars1 = ax1.bar(index - bar_width/2, [dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='Reuse-Proposed', color="#B47B84", zorder=3)
# 绘制第二组条形图，仍然使用 ax1 的 y 轴
bars2 = ax1.bar(index + bar_width/2, [noreuse_dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='Traditional(No Reuse)', color="#CDBBA7", zorder=3)
ax1.set_xlabel('Number of Requests', fontsize=13)
ax1.set_ylabel('Resource Consumption per Completed Service Request', fontsize=13)
# 创建第二个 y 轴
ax2 = ax1.twinx()
# 绘制折线图，使用 ax2 的 y 轴
line1, = ax2.plot(index, [dbo_reuse_histories[num] for num in num_requests_list], label='DBSSO Resource Reuse Rate', color="#AA5656", marker='o', zorder=4)
ax2.set_ylabel('Resource Reuse Rate', fontsize=13)
# 设置 x 轴刻度标签
ax1.set_xticks(index)
ax1.set_xticklabels(num_requests_list)
# 添加图例
legend_elements = [bars1, bars2, line1]
ax1.legend(handles=legend_elements, loc='lower right')
# 添加网格
ax1.grid(zorder=0, color='gray',alpha = 0.3)
# 保存和展示图表
plt.savefig('Combined_Chart.pdf', bbox_inches='tight')
plt.savefig('Combined_Chart.png', bbox_inches='tight', dpi=600, pad_inches=0.0)

#中文画图
# 条形图设置
bar_width = 0.35  # 条形宽度
index = np.arange(len(num_requests_list))  # NUM_REQUESTS 的索引
# 创建图表并设置大小
plt.figure(figsize=(10, 6))
ax1 = plt.gca()  # 获取当前轴
# 绘制第一组条形图，使用 ax1 的 y 轴
bars1 = ax1.bar(index - bar_width/2, [dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='Reuse-Proposed', color="#B47B84", zorder=3)
# 绘制第二组条形图，仍然使用 ax1 的 y 轴
bars2 = ax1.bar(index + bar_width/2, [noreuse_dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='Traditional(No Reuse)', color="#CDBBA7", zorder=3)
ax1.set_xlabel('用户请求数量', fontsize=13)
ax1.set_ylabel('平均每个完成服务请求的资源消耗', fontsize=13)
# 创建第二个 y 轴
ax2 = ax1.twinx()
# 绘制折线图，使用 ax2 的 y 轴
line1, = ax2.plot(index, [dbo_reuse_histories[num] for num in num_requests_list], label='DBSSO-资源复用率', color="#AA5656", marker='o', zorder=4)
ax2.set_ylabel('资源复用率', fontsize=13)
# 设置 x 轴刻度标签
ax1.set_xticks(index)
ax1.set_xticklabels(num_requests_list)
# 添加图例
legend_elements = [bars1, bars2, line1]
ax1.legend(handles=legend_elements, loc='lower right')
# 添加网格
ax1.grid(zorder=0, color='gray',alpha = 0.3)
# 保存和展示图表
plt.savefig('Combined_Chart_Chinese.pdf', bbox_inches='tight')
plt.savefig('Combined_Chart_Chinese.png', bbox_inches='tight', dpi=600, pad_inches=0.0)'''