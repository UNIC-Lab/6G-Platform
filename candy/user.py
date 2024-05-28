import random

'''# 定义范围
Z_q_RANGE = (102400, 52428800)  # 数据大小范围 (bytes)
D_q_RANGE = (1, 200)           # 时延要求范围 (ms)
A_q_RANGE = (60, 82)           # 精度要求范围

# 生成单个用户需求，使计算资源、时延和精度成正比
def generate_user_request():
    scale = random.random()  # 生成一个随机比例因子

    # 基于比例因子和各参数范围计算值
    Z_q = round(Z_q_RANGE[0] + scale * (Z_q_RANGE[1] - Z_q_RANGE[0]))
    D_q = round(D_q_RANGE[0] + scale * (D_q_RANGE[1] - D_q_RANGE[0]), 4)
    A_q = round(A_q_RANGE[0] + scale * (A_q_RANGE[1] - A_q_RANGE[0]), 4)

    return Z_q, D_q, A_q

# 生成所有用户需求
def generate_all_user_requests(num_requests):
    return [generate_user_request() for _ in range(num_requests)]

# 示例：生成5个用户需求
user_requests = generate_all_user_requests(5)
print(user_requests)'''


'''import matplotlib.pyplot as plt
import numpy as np

# 定义用户请求的范围
num_requests_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# 假设这些是从你的算法中获得的 alpha 值
dbo_alpha_histories = {num_requests: np.random.rand() for num_requests in num_requests_list}
single_dbo_alpha_histories = {num_requests: np.random.rand() for num_requests in num_requests_list}

# 条形图设置
bar_width = 0.35  # 条形宽度
index = np.arange(len(num_requests_list))  # NUM_REQUESTS 的索引

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.bar(index - bar_width/2, [dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='Reuse-Proposed', color="#343434")
plt.bar(index + bar_width/2, [single_dbo_alpha_histories[num] for num in num_requests_list], bar_width, label='Traditional(No Reuse)', color="#E6B31E")

# 添加图表标签和标题
plt.xlabel('Number of Requests')
plt.ylabel('Resource Consumption per Completed Service Request')
plt.xticks(index, num_requests_list)  # 设置 X 轴刻度标签
plt.legend()
plt.tight_layout()  # 调整布局以适应图形
plt.show()'''

import matplotlib.pyplot as plt
import numpy as np

# 假设max_ratio_dbo_fitness等变量已经根据您的代码填充了相应的数据
ratio_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
# 下面的值仅为示例，实际应该使用您计算得到的值
max_ratio_dbo_fitness = {num_requests: np.random.rand() for num_requests in ratio_list}
max_ratio_ga_fitness = {num_requests: np.random.rand() for num_requests in ratio_list}
max_ratio_greedy_fitness = {num_requests: np.random.rand() for num_requests in ratio_list}
max_ratio_random_fitness = {num_requests: np.random.rand() for num_requests in ratio_list}

# 条形图设置
bar_width = 0.2  # 条形宽度
index = np.arange(len(ratio_list))  # NUM_REQUESTS 的索引
# 绘制条形图
plt.figure(figsize=(10, 6))
# 注意，这里我们直接使用字典的键（即 ratio_list 中的值）来获取每个比例下的最大适应度
plt.bar(index - bar_width*1.5, [max_ratio_dbo_fitness[ratio] for ratio in ratio_list], bar_width, label='DBO', color="#6DA4AA")
plt.bar(index - bar_width/2, [max_ratio_ga_fitness[ratio] for ratio in ratio_list], bar_width, label='GA', color="#7DB8DA")
plt.bar(index + bar_width/2, [max_ratio_greedy_fitness[ratio] for ratio in ratio_list], bar_width, label='Greedy', color="#82CA9D")
plt.bar(index + bar_width*1.5, [max_ratio_random_fitness[ratio] for ratio in ratio_list], bar_width, label='Random', color="#D7837F")

# 添加图表标签和标题
plt.xlabel('Ratio')
plt.ylabel('Max Fitness')
plt.xticks(index, [str(r) for r in ratio_list])  # 设置 X 轴刻度标签为字符串形式的比例值

plt.legend()
# 保存和展示图表
plt.savefig('Max_Fitness_Comparison_Bar_Chart.pdf', bbox_inches='tight')
plt.savefig('Max_Fitness_Comparison_Bar_Chart.png', bbox_inches='tight', dpi=600, pad_inches=0.0)
plt.show()
