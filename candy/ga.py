import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import joblib
from predict import predict_performance
# 配置中文字体，这里使用微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 保证能够在图中显示负号
plt.rcParams['axes.unicode_minus'] = False

# 定义参数
NUM_MODELS = 3   # 模型数量
NUM_REQUESTS = 3  # 用户请求数量

# 生成从0到1的小数
BETA_VALUES = [i / 100.0 for i in range(101)]
L_VALUES = [0, 1, 2, 3, 4]  # 部署位置基因库: 0 到 4

# 用户需求参数范围
Z_q_RANGE = (102400, 52428800)  # 数据大小范围 (bytes)
#D_q_RANGE = (10, 150)  # 时延要求范围 (ms)
#A_q_RANGE = (65, 82)  # 精度要求范围
D_q_RANGE = (1, 200)  # 时延要求范围 (ms)
A_q_RANGE = (60, 82)  # 精度要求范围

# 模型库中原始部署资源
c_md = [1.3, 4.6, 17.6]  # 对应 S, M, L 模型

# 网络参数
NETWORK_PARAMS = {
    1: {'N': 100, 'C': 5, 'r': 1250000000},    # 第1层网络参数
    2: {'N': 60, 'C': 20, 'r': 6250000000},     # 第2层网络参数
    3: {'N': 20, 'C': 80, 'r': 12500000000},    # 第3层网络参数
    4: {'N': float('inf'), 'C': float('inf'), 'r': 125000000000}  # 云（第4层）网络参数
}

# 加载模型和scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# 初始化函数(更新过)
def initialize_beta_L_matrices(num_requests):
    beta_matrix = np.zeros((NUM_MODELS, num_requests), dtype=float)
    L_matrix = np.zeros((2, num_requests), dtype=int)

    # 新的初始化过程，没有约束限制，除了要求beta_matrix每一列最多只能有两个元素有数值
    for q in range(num_requests):
        # 随机选择两行来填充非零元素
        selected_models = random.sample(range(NUM_MODELS), 2)
        # 为选中的两行随机选择基因库中的数值
        for model in selected_models:
            beta_matrix[model][q] = random.choice(BETA_VALUES)

    # 随机初始化L_matrix
    for row in range(2):
        for q in range(num_requests):
            L_matrix[row][q] = random.choice(L_VALUES)

    # 初始化beta_p, beta_r, beta_s, L_p, L_r, L_s, pre_model_index
    beta_p, beta_r, beta_s = [np.zeros((NUM_MODELS, num_requests)) for _ in range(3)]
    L_p, L_r, L_s = [np.zeros(num_requests) for _ in range(3)]
    pre_model_index = [-1 for _ in range(num_requests)]
    pos_model_index = [-1 for _ in range(num_requests)]
    sin_model_index = [-1 for _ in range(num_requests)]

    for q in range(num_requests):
        selected_models = [i for i in range(NUM_MODELS) if beta_matrix[i][q] > 0]
        if len(selected_models) == 1:
            model = selected_models[0]
            sin_model_index[q] = selected_models[0]
            beta_s[model][q] = beta_matrix[model][q]
            L_s[q] = L_matrix[0][q]  # 假设单模型部署在L_matrix的第一行
        elif len(selected_models) == 2:
            pre_model_index[q] = selected_models[0]
            pos_model_index[q] = selected_models[1]
            beta_p[selected_models[0]][q] = beta_matrix[selected_models[0]][q]
            beta_r[selected_models[1]][q] = beta_matrix[selected_models[1]][q]
            L_p[q] = L_matrix[0][q]
            L_r[q] = L_matrix[1][q]

    return beta_matrix, L_matrix, beta_p, beta_r, beta_s, L_p, L_r, L_s, pre_model_index , pos_model_index, sin_model_index

# 定义生成单个个体的函数（更新过）
def generate_individual(num_requests):
    # 使用初始化函数生成个体的各个矩阵
    beta_matrix, L_matrix, beta_p, beta_r, beta_s, L_p, L_r, L_s , pre_model_index, pos_model_index, sin_model_index= initialize_beta_L_matrices(num_requests)

    # 个体由这些矩阵组成
    individual = {
        'beta_matrix': beta_matrix,
        'L_matrix': L_matrix,
        'beta_p': beta_p,
        'beta_r': beta_r,
        'beta_s': beta_s,
        'L_p': L_p,
        'L_r': L_r,
        'L_s': L_s,
        'pre_model_index': pre_model_index,
        'pos_model_index': pos_model_index,
        'sin_model_index': sin_model_index
    }
    return individual

# 定义生成种群的函数
def generate_population(population_size, num_requests):
    return [generate_individual(num_requests) for _ in range(population_size)]

# 生成单个用户需求
def generate_user_request():
    Z_q = random.randint(*Z_q_RANGE)
    D_q = round(random.uniform(*D_q_RANGE), 4)
    A_q = round(random.uniform(*A_q_RANGE), 4)
    return Z_q, D_q, A_q

# 生成所有用户需求
def generate_all_user_requests(num_requests):
    return [generate_user_request() for _ in range(num_requests)]

# 辅助函数定义
def zeta(beta_nq):
    return 1 if beta_nq > 0 else 0

def gamma(beta_nq1, beta_nq2):
    return 1 if 0 < beta_nq1 == beta_nq2 < 1 else 0

def gamma_prime(beta_nq1, beta_nq2):
    return 1 if beta_nq1 == beta_nq2 == 1 else 0

def indicator(condition):
    return 1 if condition else 0

def phi(beta_nq_p):
    """ 计算前置模型数据变化系数 """
    return (1 - beta_nq_p) if beta_nq_p > 0 else 0

#缝合模型部署的系数判断
def coefficient_f(beta_nq_p,beta_nq_r):
    S_cof = 0
    coefficient_f_success = 0
    sumcoefficient_f = beta_nq_p + beta_nq_r
    if sumcoefficient_f == 1 :
        S_cof = 10
        coefficient_f_success = 1
    return coefficient_f_success, S_cof

#单模型部署的系数判断
def coefficient_s(beta_nq_s):
    S_cos = 0
    coefficient_s_success = 0
    sumcoefficient_s = beta_nq_s
    if sumcoefficient_s == 1:
        S_cos = 15
        coefficient_s_success = 1
    return coefficient_s_success, S_cos

#缝合模型大小判断
def model_size(pre_model_index_q,pos_model_index_q):
    mode_size_success = 0
    S_mode = 0
    if pre_model_index_q < pos_model_index_q :
        mode_size_success = 1
        S_mode = 10
    return mode_size_success, S_mode

#缝合部署位置判断
def deploy_f(L_q_p, L_q_r):
    S_depf = 0
    deploy_f_success = 0
    if 0 < L_q_p <= L_q_r :
        deploy_f_success = 1
        S_depf = 10
    return deploy_f_success, S_depf

#单部署位置判断
def deploy_s(L_q_s):
    S_deps = 0
    deploy_s_success = 0
    if L_q_s > 0 :
        deploy_s_success = 1
        S_deps = 15
    return deploy_s_success, S_deps

#是否时延、精度满足判断
def delay_and_acc_judge(D_q, A_q, t_q, a_q):
    S_delay = 0
    S_acc = 0
    delay_success = 0
    acc_success = 0
    base_score = 20
    max_extra_score = 10
    max_diff_t = 10
    max_diff_a = 3
    if t_q < D_q:
        delay_success = 1
        S_delay = base_score
        # 计算差值
        diff_t = D_q - t_q
        # 计算加分
        extra_score_t = min(max_extra_score, (diff_t / max_diff_t) * max_extra_score)
        S_delay = S_delay + extra_score_t

    if a_q > A_q:
        acc_success = 1
        S_acc = base_score
        # 计算差值
        diff_a = a_q - A_q
        # 计算加分
        extra_score_a = min(max_extra_score, (diff_a / max_diff_a) * max_extra_score)
        S_acc = S_acc + extra_score_a
    return delay_success, acc_success, S_delay, S_acc

#资源是否受限判断
def resource_judge(alpha_mp, alpha_mr, alpha_single, NETWORK_PARAMS):
    resource_success = 0
    S_resource = 0
    resource_base_score = 5
    max_extra_resource_score = 5
    max_diff_resource = 1
    floor = 0
    # 判断前三层资源限制是否满足，云层（第四层）资源是无限的，因此不考虑
    # 只检查前三层的资源限制
    for i in range(3):
        if alpha_mp[i] + alpha_mr[i] + alpha_single[i] <= NETWORK_PARAMS[i + 1]['C'] * NETWORK_PARAMS[i + 1]['N']:
            floor += 1
    if floor == 3:
        resource_success = 1
        S_resource = resource_base_score
        # 计算差值
        diff_resource = []
        for i in range(3):
            diff = ((NETWORK_PARAMS[i + 1]['C'] * NETWORK_PARAMS[i + 1]['N']) - (alpha_mp[i] + alpha_mr[i] + alpha_single[i])) / (NETWORK_PARAMS[i + 1]['C'] * NETWORK_PARAMS[i + 1]['N'])
            diff_resource.append(diff)
        # 计算加分
        ave = (diff_resource[0] + diff_resource[1] + diff_resource[2]) / 3
        extra_resource_score = min(max_extra_resource_score, (ave / max_diff_resource) * max_extra_resource_score)
        S_resource = S_resource + extra_resource_score
    return resource_success, S_resource

# AI模型部署资源计算(改过)
def calculate_deployment_cost(beta_p, beta_r, beta_s, L_p, L_r, L_s, c_md):
    alpha_mp = np.zeros(4)  # 对应四层网络的前置模型部署资源
    alpha_mr = np.zeros(4)  # 对应四层网络的后置模型部署资源
    alpha_single = np.zeros(4)  # 对应四层网络的单模型部署资源

    for i in range(1, 5):  # 遍历四层网络，层数从1到4
        alpha_single_indicator = np.zeros(NUM_MODELS)  # 对于三个模型的每个AI模型的指标值
        for n in range(NUM_MODELS):  # 遍历每个模型
            for q in range(NUM_REQUESTS):  # 遍历每个用户请求
                l_qp = 1 if L_p[q] == i else 0
                l_qr = 1 if L_r[q] == i else 0
                l_qs = 1 if L_s[q] == i else 0

                alpha_mp[i - 1] += zeta(beta_p[n][q]) * l_qp * beta_p[n][q] * c_md[n]
                alpha_mr[i - 1] += zeta(beta_r[n][q]) * l_qr * beta_r[n][q] * c_md[n]

                for q2 in range(NUM_REQUESTS):
                    if q != q2:
                        l_q2p = 1 if L_p[q2] == i else 0
                        l_q2r = 1 if L_r[q2] == i else 0
                        l_q2s = 1 if L_s[q2] == i else 0
                        alpha_mp[i - 1] -= gamma(beta_p[n][q], beta_p[n][q2]) * l_qp * l_q2p * beta_p[n][q]* c_md[n]
                        alpha_mr[i - 1] -= gamma(beta_r[n][q], beta_r[n][q2]) * l_qr * l_q2r * beta_r[n][q] * c_md[n]
                        alpha_single_indicator[n] += gamma_prime(beta_s[n][q], beta_s[n][q2]) * l_qs * l_q2s

            # 在每个模型的所有请求遍历完后，判断是否有复用
            alpha_single[i - 1] += c_md[n] if alpha_single_indicator[n] > 0 else 0

    return alpha_mp, alpha_mr, alpha_single

# 获取推理时延和精度(new)
def get_inference_time_and_accuracy(beta_p, beta_r, beta_s, c_md):
    """根据缝合系数获取推理时延和精度"""
    inference_time = []
    accuracy = []
    for q in range(NUM_REQUESTS):
        gflops = 0
        # 确定是缝合部署还是单AI部署
        if sum(beta_p[:, q]) > 0 and sum(beta_r[:, q]) > 0:  # 缝合部署
            # 计算缝合后的模型大小
            for n in range(NUM_MODELS):
                gflops += beta_p[n][q] * c_md[n] + beta_r[n][q] * c_md[n]
        else:  # 单AI模型部署
            for n in range(NUM_MODELS):
                if beta_s[n][q] > 0:
                    gflops = c_md[n]
                    break

        # 使用predict_performance函数获取精度和推理时延
        acc, inf_time = predict_performance(gflops)
        accuracy.append(acc)
        inference_time.append(inf_time)
    return inference_time, accuracy

# 对于某个用户q的请求的通信时延计算(new)
def calculate_transmission_time(q, Z_q, beta_p, L_q_s, L_q_p, L_q_r, pre_model_index_q, NETWORK_PARAMS):
    """ 计算通信时延 """
    L_q_s = int(L_q_s)  # 转换为整数
    L_q_p = int(L_q_p)  # 转换为整数
    L_q_r = int(L_q_r)  # 转换为整数
    pre_model_index_q = int(pre_model_index_q)

    # 当L_q_s、L_q_r、L_q_p都为0时，通信时延为0，代表没有部署
    if L_q_s == 0 and L_q_r == 0 and L_q_p == 0:
        return 0
    else:
        # 单独部署
        if L_q_s != 0:
            return sum(Z_q / NETWORK_PARAMS[i]['r'] for i in range(1, L_q_s + 1))
        # 缝合模型部署
        elif L_q_p > 0 and L_q_r > 0:
            # 当L_q_r < L_q_p时，返回0，代表部署失败
            if L_q_r < L_q_p:
                return 0
            else:
                delta_L_q = L_q_r - L_q_p
                Z_q_pout = phi(beta_p[pre_model_index_q][q]) * Z_q  # 使用前置模型数据变化系数
                return sum(Z_q / NETWORK_PARAMS[i]['r'] for i in range(1, L_q_p + 1)) + sum(Z_q_pout / NETWORK_PARAMS[i]['r'] for i in range(L_q_p + 1, L_q_r + 1)) * delta_L_q

    # 其他情况，默认返回0
    return 0

# 计算个体的用户满意度（加入约束）
def calculate_satisfaction(inference_time, accuracy, user_requests, beta_p, beta_r, beta_s, L_p, L_r, L_s, pre_model_index, pos_model_index, sin_model_index, NETWORK_PARAMS):
    S = []  # 用户满意度
    F = []  # 完成指标向量
    for q in range(NUM_REQUESTS):
        # 检查索引是否在列表长度范围内
        if q >= len(inference_time) or q >= len(accuracy):
            continue

        Z_q, D_q, A_q = user_requests[q]
        t_s_q = inference_time[q]  # AI模型的推理时延
        a_q = accuracy[q]
        n_pre = int(pre_model_index[q])
        n_pos = int(pos_model_index[q])
        n_sin = int(sin_model_index[q])
        L_q_p = int(L_p[q])
        L_q_r = int(L_r[q])
        L_q_s = int(L_s[q])
        S_q = 0
        success_q = 0
        # 计算传输时延
        t_trans_q = calculate_transmission_time(q, Z_q, beta_p, L_p[q], L_r[q], L_s[q], n_pre, NETWORK_PARAMS)
        if t_trans_q > 0:
            # 总时延（端到端时延）
            t_q = t_trans_q + t_s_q
            # 确定是缝合部署还是单AI部署
            if sum(beta_p[:, q]) > 0 and sum(beta_r[:, q]) > 0:  # 缝合部署
                # 计算缝合部署的用户满意度
                mode_size_success, S_mode = model_size(n_pre, n_pos)
                coefficient_f_success, S_cof = coefficient_f(beta_p[n_pre, q], beta_r[n_pos, q])
                deploy_f_success, S_depf = deploy_f(L_q_p, L_q_r)
                delay_success, acc_success, S_delay, S_acc = delay_and_acc_judge(D_q, A_q, t_q, a_q)
                S_q = S_mode + S_cof + S_depf + S_delay + S_acc
                S.append(S_q)
                success_q = mode_size_success + coefficient_f_success + deploy_f_success + delay_success + acc_success
                if success_q == 5:
                    print(f"F is updated for request {q} in Stitching Deployment")
                    F.append(1)
                else:
                    print(f"request {q} in Stitching Deployment is failed servers")
                    F.append(0)

            else:  # 单AI模型部署
                coefficient_s_success, S_cos = coefficient_s(beta_s[n_sin, q])
                deploy_s_success, S_deps = deploy_s(L_q_s)
                delay_success, acc_success, S_delay, S_acc = delay_and_acc_judge(D_q, A_q, t_q, a_q)
                S_q = S_cos + S_deps + S_delay + S_acc
                S.append(S_q)
                success_q = coefficient_s_success + deploy_s_success + delay_success + acc_success
                if success_q == 4:
                    print(f"F is updated for request {q} in Single AI Deployment")
                    F.append(1)
                else:
                    print(f"request {q} in Single AI Deployment is failed servers")
                    F.append(0)
        else:
            S.append(S_q)
            F.append(0)

    # 计算按需服务率
    theta = sum(F) / NUM_REQUESTS
    Satisfaction = np.mean(S)
    return Satisfaction, theta

# 遗传算法计算个体对于某请求q的适应度
def calculate_individual_fitness(individual, q, user_requests_q, NETWORK_PARAMS):
    Z_q, D_q, A_q = user_requests_q
    beta_p = individual['beta_p']
    beta_r = individual['beta_r']
    beta_s = individual['beta_s']
    L_p = individual['L_p']
    L_r = individual['L_r']
    L_s = individual['L_s']
    pre_model_index = individual['pre_model_index']
    pos_model_index = individual['pos_model_index']
    sin_model_index = individual['sin_model_index']
    inference_time, accuracy = get_inference_time_and_accuracy(beta_p, beta_r,beta_s, c_md)
    t_s_q = inference_time[q]  # AI模型的推理时延
    a_q = accuracy[q]
    n_pre = int(pre_model_index[q])
    n_pos = int(pos_model_index[q])
    n_sin = int(sin_model_index[q])
    L_q_p = int(L_p[q])
    L_q_r = int(L_r[q])
    L_q_s = int(L_s[q])
    S_q = 0
    success_q = 0
    # 计算传输时延
    t_trans_q = calculate_transmission_time(q, Z_q, beta_p, L_p[q], L_r[q], L_s[q], n_pre, NETWORK_PARAMS)
    if t_trans_q > 0:
        # 总时延（端到端时延）
        t_q = t_trans_q + t_s_q
        # 确定是缝合部署还是单AI部署
        if sum(beta_p[:, q]) > 0 and sum(beta_r[:, q]) > 0:  # 缝合部署
            # 计算缝合部署的用户满意度
            mode_size_success, S_mode = model_size(n_pre, n_pos)
            coefficient_f_success, S_cof = coefficient_f(beta_p[n_pre, q], beta_r[n_pos, q])
            deploy_f_success, S_depf = deploy_f(L_q_p, L_q_r)
            delay_success, acc_success, S_delay, S_acc = delay_and_acc_judge(D_q, A_q, t_q, a_q)
            S_q = (S_mode + S_cof + S_depf + S_delay + S_acc)/90
            success_q = (mode_size_success + coefficient_f_success + deploy_f_success + delay_success + acc_success)/5
        else:  # 单AI模型部署
            coefficient_s_success, S_cos = coefficient_s(beta_s[n_sin, q])
            deploy_s_success, S_deps = deploy_s(L_q_s)
            delay_success, acc_success, S_delay, S_acc = delay_and_acc_judge(D_q, A_q, t_q, a_q)
            S_q = (S_cos + S_deps + S_delay + S_acc)/90
            success_q = (coefficient_s_success + deploy_s_success + delay_success + acc_success)/4
    else:
        S_q = 0
        success_q = 0
    # 计算个体某用户的满意度
    Satisfaction_q = S_q * 50 + success_q * 50
    return Satisfaction_q

# 遗传算法计算种群的适应度
def calculate_population_fitness(population, user_requests, NETWORK_PARAMS):
    fitnesses = []
    thetas = []
    for individual in population:
        # 计算推理时延和精度
        inference_time, accuracy = get_inference_time_and_accuracy(individual['beta_p'], individual['beta_r'],
                                                                   individual['beta_s'], c_md)
        #计算个体的满意度
        Satisfaction, theta = calculate_satisfaction(inference_time, accuracy, user_requests, individual['beta_p'],
                                                     individual['beta_r'], individual['beta_s'], individual['L_p'],
                                                     individual['L_r'], individual['L_s'], individual['pre_model_index'],
                                                     individual['pos_model_index'], individual['sin_model_index'], NETWORK_PARAMS)
        # 计算个体的部署成本
        alpha_mp, alpha_mr, alpha_single = calculate_deployment_cost(individual['beta_p'], individual['beta_r'],
                                                                     individual['beta_s'], individual['L_p'],
                                                                     individual['L_r'], individual['L_s'], c_md)

        # 判断是否满足时延和精度要求
        resource_success, S_resource = resource_judge(alpha_mp, alpha_mr, alpha_single, NETWORK_PARAMS)
        if resource_success == 1:
            Satisfaction_full = Satisfaction + S_resource
        else:
            Satisfaction_full = Satisfaction
        # 计算适应度
        Satisfaction_score = (Satisfaction_full / 100) * 50
        theta_score = theta * 50
        fitness = Satisfaction_score + theta_score
        fitnesses.append(fitness)
        thetas.append(theta)
    return fitnesses, thetas

#个体更新函数
def update_individual(individual):
    beta_matrix = individual['beta_matrix']
    L_matrix = individual['L_matrix']

    for q in range(NUM_REQUESTS):
        selected_models = [i for i in range(NUM_MODELS) if beta_matrix[i][q] > 0]
        if len(selected_models) == 2:
            # 假设两个模型中较小的索引是前置模型
            individual['pre_model_index'][q] = selected_models[0]
            individual['pos_model_index'][q] = selected_models[1]
            individual['beta_p'][selected_models[0]][q] = beta_matrix[selected_models[0]][q]
            individual['beta_r'][selected_models[1]][q] = beta_matrix[selected_models[1]][q]
            individual['L_p'][q] = L_matrix[0][q]
            individual['L_r'][q] = L_matrix[1][q]
            individual['beta_s'][:, q] = 0  # 缝合模型，无单独部署
            individual['L_s'][q] = 0
        elif len(selected_models) == 1:
            individual['pre_model_index'][q] = -1  # 无前置模型
            individual['pos_model_index'][q] = -1
            individual['sin_model_index'][q] = selected_models[0]
            individual['beta_p'][:, q] = 0
            individual['beta_r'][:, q] = 0
            individual['beta_s'][selected_models[0]][q] = beta_matrix[selected_models[0]][q]
            individual['L_s'][q] = L_matrix[0][q]
            individual['L_p'][q] = 0
            individual['L_r'][q] = 0
        else:
            individual['pre_model_index'][q] = -1
            individual['pos_model_index'][q] = -1
            individual['sin_model_index'][q] = -1
            individual['beta_p'][:, q] = 0
            individual['beta_r'][:, q] = 0
            individual['beta_s'][:, q] = 0
            individual['L_p'][q] = 0
            individual['L_r'][q] = 0
            individual['L_s'][q] = 0

    return individual

# 精英选择策略函数，结合锦标赛选择
def elite_and_tournament_selection(population, fitnesses, elite_size, tournament_size):
    # 根据适应度对种群进行排序
    sorted_population = sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)

    # 选择精英个体
    elite_population = [individual for _, individual in sorted_population[:elite_size]]
    #锦标赛选择上限
    choose_num = 50
    # 锦标赛选择
    while len(elite_population) < choose_num:
        tournament = random.sample(sorted_population[elite_size:], tournament_size)
        winner = max(tournament, key=lambda pair: pair[0])[1]
        elite_population.append(winner)

    return elite_population


# 定义均匀交叉函数
def uniform_crossover(individual1, individual2, user_requests, NETWORK_PARAMS):
    new_individual1, new_individual2 = deepcopy(individual1), deepcopy(individual2)

    for q in range(NUM_REQUESTS):
        fitness1_q = calculate_individual_fitness(individual1, q, user_requests[q], NETWORK_PARAMS)
        fitness2_q = calculate_individual_fitness(individual2, q, user_requests[q], NETWORK_PARAMS)

        for n in range(NUM_MODELS):
            # 根据适应度选择交叉的基因
            if fitness1_q > fitness2_q:
                new_individual2['beta_matrix'][n][q] = individual1['beta_matrix'][n][q]
            else:
                new_individual1['beta_matrix'][n][q] = individual2['beta_matrix'][n][q]

            if random.random() < 0.5:  # 保留一定的随机性
                new_individual1['L_matrix'][0][q], new_individual2['L_matrix'][0][q] = new_individual2['L_matrix'][0][q], new_individual1['L_matrix'][0][q]
                new_individual1['L_matrix'][1][q], new_individual2['L_matrix'][1][q] = new_individual2['L_matrix'][1][q], new_individual1['L_matrix'][1][q]

    # 更新依赖矩阵
    new_individual1 = update_individual(new_individual1)
    new_individual2 = update_individual(new_individual2)

    return new_individual1, new_individual2

# 均匀变异函数
def uniform_mutation_and_update(individual, user_requests, NETWORK_PARAMS, mutation_rate, beta_mutation_intensity, L_mutation_intensity):
    new_individual = deepcopy(individual)

    for q in range(NUM_REQUESTS):
        fitness_q = calculate_individual_fitness(individual, q, user_requests[q], NETWORK_PARAMS)
        # 根据适应度调整变异概率
        adjusted_mutation_rate = mutation_rate * (1 - fitness_q / 100)

        # BETA值的变异
        for i in range(NUM_MODELS):
            if random.random() < adjusted_mutation_rate:
                # 根据变异强度选择新值
                current_beta_value = individual['beta_matrix'][i][q]
                new_beta_value = current_beta_value + random.uniform(-beta_mutation_intensity, beta_mutation_intensity)
                new_beta_value = max(0, min(new_beta_value, 1))  # 确保在有效范围内
                new_individual['beta_matrix'][i][q] = new_beta_value

        # L值的变异
        for i in range(2):
            if random.random() < adjusted_mutation_rate:
                # 计算新的L值
                current_L_value = individual['L_matrix'][i][q]
                new_L_value = current_L_value + random.randint(-L_mutation_intensity, L_mutation_intensity)
                new_L_value = max(0, min(new_L_value, max(L_VALUES)))  # 确保在有效范围内
                new_individual['L_matrix'][i][q] = new_L_value

    # 更新其他矩阵
    new_individual = update_individual(new_individual)

    return new_individual

def GA(population, population_size, num_iterations, user_requests, NETWORK_PARAMS, elite_size, tournament_size, mutation_rate, beta_mutation_intensity, L_mutation_intensity):
    # 用于记录遗传算法适应度和按需服务率
    ga_fitness_history = []
    ga_service_rate_history = []

    # 遗传算法迭代过程
    for ga_iteration in range(num_iterations):
        print(f"ga迭代 {ga_iteration + 1}/{num_iterations}...")
        iteration_start_time = time.time()

        # 计算适应度、按需服务率
        ga_fitnesses, ga_service_rates = calculate_population_fitness(population, user_requests, NETWORK_PARAMS)
        ga_fitness_history.append(ga_fitnesses)
        ga_service_rate_history.append(ga_service_rates)

        # 精英选择和锦标赛选择
        selected_population = elite_and_tournament_selection(population, ga_fitnesses, elite_size, tournament_size)

        # 交叉和变异
        new_population = selected_population.copy()  # 将精英直接复制到新种群中
        while len(new_population) < population_size:
            # 随机选择两个父代进行交叉和变异
            parent1, parent2 = random.sample(selected_population, 2)
            offspring1, offspring2 = uniform_crossover(parent1, parent2, user_requests, NETWORK_PARAMS)
            mutated_offspring1 = uniform_mutation_and_update(offspring1, user_requests, NETWORK_PARAMS, mutation_rate, beta_mutation_intensity, L_mutation_intensity)
            mutated_offspring2 = uniform_mutation_and_update(offspring2, user_requests, NETWORK_PARAMS, mutation_rate, beta_mutation_intensity, L_mutation_intensity)
            new_population.extend([mutated_offspring1, mutated_offspring2])

        population = new_population[:population_size]

        print(f"GA完成迭代 {ga_iteration + 1}，总耗时 {time.time() - iteration_start_time:.2f} 秒。\n")

    # 获取最终的最优个体
    ga_best_fitness_idx = np.argmax(ga_fitness_history[-1])
    ga_best_individual = population[ga_best_fitness_idx]
    ga_best_fitness = max(ga_fitness_history[-1])
    ga_best_service_rate = max(ga_service_rate_history[-1])

    return ga_best_fitness, ga_best_service_rate, ga_fitness_history, ga_service_rate_history


# 生成所有用户需求
print("生成用户需求...")
start_time = time.time()
user_requests = generate_all_user_requests(NUM_REQUESTS)
print(f"完成用户需求生成，耗时 {time.time() - start_time:.2f} 秒。")

# 初始化种群
population_size = 300
print("初始化种群...")
start_time = time.time()
population = generate_population(population_size, NUM_REQUESTS)
print(f"完成种群初始化，耗时 {time.time() - start_time:.2f} 秒。")

# 定义迭代次数
num_iterations = 100
#定义变异相关参数
mutation_rate = 0.3
beta_mutation_intensity = 0.01
L_mutation_intensity = 1
# 精英选择
elite_size = int(population_size * 0.3) # 选择种群中20%的个体作为精英
#锦标赛参数
tournament_size = 30

# 运行遗传算法
ga_best_fitness, ga_best_service_rate, ga_fitness_history, ga_service_rate_history = GA(population, population_size, num_iterations, user_requests, NETWORK_PARAMS, elite_size, tournament_size, mutation_rate, beta_mutation_intensity, L_mutation_intensity)
# 绘制适应度历史
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.max(ga_fitness_history, axis=1), label='Max Fitness')
plt.plot(np.mean(ga_fitness_history, axis=1), label='Average Fitness')
plt.title('Fitness over Iterations')
plt.xlabel('迭代次数')
plt.ylabel('用户满意度')
#plt.xlabel('Iteration')
#plt.ylabel('Fitness')
plt.legend()

# 绘制按需服务率历史
plt.subplot(1, 2, 2)
plt.plot(np.max(ga_service_rate_history, axis=1), label='Max Service Rate')
plt.plot(np.mean(ga_service_rate_history, axis=1), label='Average Service Rate')
plt.title('Service Rate over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Service Rate')
plt.legend()

# 保存图像
plt.tight_layout()
#plt.savefig('ga3.pdf', bbox_inches='tight')
#plt.savefig('ga3.png', bbox_inches='tight', dpi=600, pad_inches=0.0)

# 显示图像
plt.show()

# 打印最优个体信息
print("Best Individual's Fitness:", max(ga_fitness_history[-1]))
print("Best Individual's Service Rate:", max(ga_service_rate_history[-1]))

