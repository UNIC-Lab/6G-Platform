import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import joblib
from predict import predict_performance

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

# 蜣螂算法计算个体的用户满意度（加入约束）
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

# 蜣螂算法计算个体对于某请求q的适应度
def calculate_individual_q_fitness(individual, q, user_requests_q, NETWORK_PARAMS):
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

# 蜣螂算法计算个体的适应度
def dbo_calculate_individual_fitness(individual, user_requests, NETWORK_PARAMS):
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
        return fitness, theta

# 蜣螂算法计算种群的适应度
def calculate_population_fitness(population, user_requests, NETWORK_PARAMS):
    fitnesses = []
    thetas = []
    for individual in population:
        fitness, theta = dbo_calculate_individual_fitness(individual, user_requests, NETWORK_PARAMS)
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

# 滚球和跳舞行为
def ball_rolling_and_dancing_behavior(individual, GworstPosition, user_requests, NETWORK_PARAMS):
    new_individual = deepcopy(individual)
    b = 0.3  # 滚球行为的影响因子
    angle = np.random.randint(0, 360)  # 跳舞行为的角度
    theta = angle * np.pi / 180  # 转换为弧度

    # 对beta_matrix进行滚球和跳舞行为的更新
    # 对beta_matrix中的每一列进行更新
    for request_idx in range(NUM_REQUESTS):
        # 检查当前请求的beta矩阵和是否为1
        sum_beta = sum(individual['beta_matrix'][:, request_idx])
        non_zero_indices = np.where(individual['beta_matrix'][:, request_idx] > 0)[0]
        num_non_zero = len(non_zero_indices)
        or_fitness_q = calculate_individual_q_fitness(individual, request_idx, user_requests[request_idx], NETWORK_PARAMS)

        if sum_beta == 1:
            # 和为1时的特殊处理
            if num_non_zero == 1:
                # 单AI模型部署
                chosen_idx = random.choice([0, 1, 2])
                new_individual['beta_matrix'][:, request_idx] = 0
                new_individual['beta_matrix'][chosen_idx][request_idx] = 1
                new_sin_fitness_q = calculate_individual_q_fitness(new_individual, request_idx, user_requests[request_idx],NETWORK_PARAMS)
                if new_sin_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]
            elif num_non_zero == 2:
                # 缝合AI模型部署
                chosen_idx = random.choice(non_zero_indices)
                angle = np.random.randint(0, 360)
                theta = angle * np.pi / 180
                change = 0  # 默认值
                if np.random.rand() < 0.9:  # 滚球行为的概率
                    a = 1 if np.random.rand() > 0.1 else -1
                    change = b * (GworstPosition['beta_matrix'][chosen_idx][request_idx] -
                                  individual['beta_matrix'][chosen_idx][request_idx]) + a * 0.01
                else:  # 跳舞行为
                    if angle not in [0, 90, 180, 270]:
                        change = np.tan(theta) * (GworstPosition['beta_matrix'][chosen_idx][request_idx] -
                                                  individual['beta_matrix'][chosen_idx][request_idx])
                change = round(change, 2)
                new_value = individual['beta_matrix'][chosen_idx][request_idx] + change
                new_value = max(0, min(new_value, 1))
                new_individual['beta_matrix'][chosen_idx][request_idx] = new_value

                other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]
                new_individual['beta_matrix'][other_idx][request_idx] = 1 - new_value
                new_snt_fitness_q = calculate_individual_q_fitness(new_individual, request_idx, user_requests[request_idx], NETWORK_PARAMS)
                if new_snt_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]

        else:
            # 随机决定是否进行直接满足约束的更新
            if random.random() < 0.6:
                if num_non_zero == 1:
                    # 单AI模型部署
                    model_idx = non_zero_indices[0]
                    new_individual['beta_matrix'][model_idx][request_idx] = 1.0
                elif num_non_zero == 2:
                    # 缝合AI模型部署
                    sum_values = sum(individual['beta_matrix'][non_zero_indices, request_idx])
                    diff = 1 - sum_values
                    change = round(random.uniform(0, abs(diff)), 2)
                    chosen_idx = random.choice(non_zero_indices)

                    # 尝试更新chosen_idx对应的元素
                    potential_new_value = new_individual['beta_matrix'][chosen_idx][request_idx] + change
                    other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]

                    if potential_new_value > 1:
                        # 如果超过1，则将change加到另一个元素上
                        new_individual['beta_matrix'][other_idx][request_idx] += change
                        new_individual['beta_matrix'][chosen_idx][request_idx] += diff - change
                    else:
                        # 如果没有超过1，则diff - change加到另一个元素上
                        new_individual['beta_matrix'][chosen_idx][request_idx] = potential_new_value
                        new_individual['beta_matrix'][other_idx][request_idx] = 1 - potential_new_value

            else:
                "未满足约束的更新"
                if num_non_zero == 1:
                    model_idx = non_zero_indices[0]
                    angle = np.random.randint(0, 360)  # 生成一个角度用于跳舞行为
                    theta = angle * np.pi / 180  # 转换为弧度
                    change = 0  # 默认值
                    if np.random.rand() < 0.9:  # 滚球行为的概率
                        a = 1 if np.random.rand() > 0.1 else -1
                        change = b * (GworstPosition['beta_matrix'][model_idx][request_idx] -
                                      individual['beta_matrix'][model_idx][request_idx]) + a * 0.01
                        new_value = individual['beta_matrix'][model_idx][request_idx] + change
                        new_value = max(0, min(new_value, 1))  # 确保在有效范围内
                        new_individual['beta_matrix'][model_idx][request_idx] = round(new_value,2)  # 更新后的值四舍五入到最接近的0.01
                    else:  # 跳舞行为
                        if angle not in [0, 90, 180, 270]:  # 特定角度时保持位置不变
                            change = np.tan(theta) * (GworstPosition['beta_matrix'][model_idx][request_idx] -
                                                      individual['beta_matrix'][model_idx][request_idx])
                            new_value = individual['beta_matrix'][model_idx][request_idx] + change
                            new_value = max(0, min(new_value, 1))  # 确保在有效范围内
                            new_individual['beta_matrix'][model_idx][request_idx] = round(new_value,2)  # 更新后的值四舍五入到最接近的0.01
                # 如果有两个元素大于0
                if num_non_zero == 2:
                    for model_idx in non_zero_indices:
                        # 对两个非零元素应用滚球和跳舞行为
                        angle = np.random.randint(0, 360)  # 生成一个角度用于跳舞行为
                        theta = angle * np.pi / 180  # 转换为弧度
                        change = 0  # 默认值
                        if np.random.rand() < 0.9:  # 滚球行为的概率
                            a = 1 if np.random.rand() > 0.1 else -1
                            change = b * (GworstPosition['beta_matrix'][model_idx][request_idx] -
                                          individual['beta_matrix'][model_idx][request_idx]) + a * 0.01
                            new_value = individual['beta_matrix'][model_idx][request_idx] + change
                            new_value = max(0, min(new_value, 1))  # 确保在有效范围内
                            new_individual['beta_matrix'][model_idx][request_idx] = round(new_value,
                                                                                          2)  # 更新后的值四舍五入到最接近的0.01
                        else:  # 跳舞行为
                            if angle not in [0, 90, 180, 270]:  # 特定角度时保持位置不变
                                change = np.tan(theta) * (GworstPosition['beta_matrix'][model_idx][request_idx] -
                                                          individual['beta_matrix'][model_idx][request_idx])
                                new_value = individual['beta_matrix'][model_idx][request_idx] + change
                                new_value = max(0, min(new_value, 1))  # 确保在有效范围内
                                new_individual['beta_matrix'][model_idx][request_idx] = round(new_value, 2)  # 更新后的值四舍五入到最接近的0.01
        #对L矩阵的更新优化
        if num_non_zero == 1:
            # 单AI模型部署
            chosen_layer = random.choice([0, 1])
            new_individual['L_matrix'][chosen_layer][request_idx] = random.choice([1, 2, 3, 4])
            new_individual['L_matrix'][1 - chosen_layer][request_idx] = 0  # 另一行设置为0
        elif num_non_zero == 2:
            # 缝合AI模型部署
            L_values = random.sample([1, 2, 3, 4], 2)
            new_individual['L_matrix'][0][request_idx], new_individual['L_matrix'][1][request_idx] = sorted(L_values)

    # 更新依赖矩阵以反映beta_matrix和L_matrix的变化
    new_individual = update_individual(new_individual)
    # 如果新个体适应度更好或相当（允许一定的随机性以增加多样性），则接受新个体
    return new_individual

# 觅食行为
def foraging_behavior(individual, user_requests, NETWORK_PARAMS, Lbb, Ubb):
    new_individual = deepcopy(individual)

    for request_idx in range(NUM_REQUESTS):
        # 检查当前请求的beta矩阵和是否为1
        sum_beta = sum(individual['beta_matrix'][:, request_idx])
        non_zero_indices = np.where(individual['beta_matrix'][:, request_idx] > 0)[0]
        num_non_zero = len(non_zero_indices)
        or_fitness_q = calculate_individual_q_fitness(individual, request_idx, user_requests[request_idx],
                                                      NETWORK_PARAMS)

        if sum_beta == 1:
            # 和为1时的特殊处理
            if num_non_zero == 1:
                # 单AI模型部署
                chosen_idx = random.choice([0, 1, 2])
                new_individual['beta_matrix'][:, request_idx] = 0
                new_individual['beta_matrix'][chosen_idx][request_idx] = 1
                new_sin_fitness_q = calculate_individual_q_fitness(new_individual, request_idx,
                                                                   user_requests[request_idx], NETWORK_PARAMS)
                if new_sin_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]
            elif num_non_zero == 2:
                # 缝合AI模型部署
                chosen_idx = random.choice(non_zero_indices)
                change = (np.random.random()) * (individual['beta_matrix'][chosen_idx][request_idx] - Lbb['beta_matrix'][chosen_idx][request_idx]) + (
                    np.random.random()) * (individual['beta_matrix'][chosen_idx][request_idx] - Ubb['beta_matrix'][chosen_idx][request_idx])
                change = round(change, 2)
                new_value = individual['beta_matrix'][chosen_idx][request_idx] + change
                new_value = max(0, min(new_value, 1))
                new_individual['beta_matrix'][chosen_idx][request_idx] = new_value
                other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]
                new_individual['beta_matrix'][other_idx][request_idx] = 1 - new_value
                new_snt_fitness_q = calculate_individual_q_fitness(new_individual, request_idx,user_requests[request_idx], NETWORK_PARAMS)
                if new_snt_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]

        else:
            # 随机决定是否进行直接满足约束的更新
            if random.random() < 0.4:
                if num_non_zero == 1:
                    # 单AI模型部署
                    model_idx = non_zero_indices[0]
                    new_individual['beta_matrix'][model_idx][request_idx] = 1.0
                elif num_non_zero == 2:
                    # 缝合AI模型部署
                    sum_values = sum(individual['beta_matrix'][non_zero_indices, request_idx])
                    diff = 1 - sum_values
                    change = round(random.uniform(0, abs(diff)), 2)
                    chosen_idx = random.choice(non_zero_indices)

                    # 尝试更新chosen_idx对应的元素
                    potential_new_value = new_individual['beta_matrix'][chosen_idx][request_idx] + change
                    other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]

                    if potential_new_value > 1:
                        # 如果超过1，则将change加到另一个元素上
                        new_individual['beta_matrix'][other_idx][request_idx] += change
                        new_individual['beta_matrix'][chosen_idx][request_idx] += diff - change
                    else:
                        # 如果没有超过1，则diff - change加到另一个元素上
                        new_individual['beta_matrix'][chosen_idx][request_idx] = potential_new_value
                        new_individual['beta_matrix'][other_idx][request_idx] = 1 - potential_new_value

            else:
                "未满足约束的更新"
                if num_non_zero == 1:
                    model_idx = non_zero_indices[0]
                    change = (np.random.random()) * (individual['beta_matrix'][model_idx][request_idx] - Lbb['beta_matrix'][model_idx][request_idx]) + (
                        np.random.random()) * (individual['beta_matrix'][model_idx][request_idx] - Ubb['beta_matrix'][model_idx][request_idx])
                    change = round(change, 2)
                    new_value = individual['beta_matrix'][model_idx][request_idx] + change
                    new_value = max(0, min(new_value, 1))
                    new_individual['beta_matrix'][model_idx][request_idx] = new_value
                # 如果有两个元素大于0
                elif num_non_zero == 2:
                    for model_idx in non_zero_indices:
                        change = (np.random.random()) * (individual['beta_matrix'][model_idx][request_idx] - Lbb['beta_matrix'][model_idx][request_idx]) + (
                            np.random.random()) * (individual['beta_matrix'][model_idx][request_idx] - Ubb['beta_matrix'][model_idx][request_idx])
                        change = round(change, 2)
                        new_value = individual['beta_matrix'][model_idx][request_idx] + change
                        new_value = max(0, min(new_value, 1))
                        new_individual['beta_matrix'][model_idx][request_idx] = new_value

        # 对L矩阵的更新优化
        if num_non_zero == 1:
            # 单AI模型部署
            chosen_layer = random.choice([0, 1])
            new_individual['L_matrix'][chosen_layer][request_idx] = random.choice([1, 2, 3, 4])
            new_individual['L_matrix'][1 - chosen_layer][request_idx] = 0  # 另一行设置为0
        elif num_non_zero == 2:
            # 缝合AI模型部署
            L_values = random.sample([1, 2, 3, 4], 2)
            new_individual['L_matrix'][0][request_idx], new_individual['L_matrix'][1][request_idx] = sorted(L_values)

    # 更新依赖矩阵以反映beta_matrix和L_matrix的变化
    new_individual = update_individual(new_individual)
    return new_individual


# 偷窃行为
def stealing_behavior(individual, GbestPosition, GbestPosition_local, user_requests, NETWORK_PARAMS):
    new_individual = deepcopy(individual)
    s = 0.5  # 偷窃行为的影响因子

    for request_idx in range(NUM_REQUESTS):
        # 检查当前请求的beta矩阵和是否为1
        sum_beta = sum(individual['beta_matrix'][:, request_idx])
        non_zero_indices = np.where(individual['beta_matrix'][:, request_idx] > 0)[0]
        num_non_zero = len(non_zero_indices)
        or_fitness_q = calculate_individual_q_fitness(individual, request_idx, user_requests[request_idx],
                                                      NETWORK_PARAMS)

        if sum_beta == 1:
            # 和为1时的特殊处理
            if num_non_zero == 1:
                # 单AI模型部署
                chosen_idx = random.choice([0, 1, 2])
                new_individual['beta_matrix'][:, request_idx] = 0
                new_individual['beta_matrix'][chosen_idx][request_idx] = 1
                new_sin_fitness_q = calculate_individual_q_fitness(new_individual, request_idx,
                                                                   user_requests[request_idx], NETWORK_PARAMS)
                if new_sin_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]
            elif num_non_zero == 2:
                # 缝合AI模型部署
                chosen_idx = random.choice(non_zero_indices)
                new_value = GbestPosition['beta_matrix'][chosen_idx][request_idx] + (np.random.random()) * s * (
                            np.abs(individual['beta_matrix'][chosen_idx][request_idx] - GbestPosition_local['beta_matrix'][chosen_idx][request_idx])
                            + np.abs(individual['beta_matrix'][chosen_idx][request_idx] - GbestPosition['beta_matrix'][chosen_idx][request_idx]))
                new_value = round(new_value, 2)
                new_value = max(0, min(new_value, 1))
                new_individual['beta_matrix'][chosen_idx][request_idx] = new_value
                other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]
                new_individual['beta_matrix'][other_idx][request_idx] = 1 - new_value
                new_snt_fitness_q = calculate_individual_q_fitness(new_individual, request_idx,
                                                                   user_requests[request_idx], NETWORK_PARAMS)
                if new_snt_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]

        else:
            # 随机决定是否进行直接满足约束的更新
            if random.random() < 0.6:
                if num_non_zero == 1:
                    # 单AI模型部署
                    model_idx = non_zero_indices[0]
                    new_individual['beta_matrix'][model_idx][request_idx] = 1.0
                elif num_non_zero == 2:
                    # 缝合AI模型部署
                    sum_values = sum(individual['beta_matrix'][non_zero_indices, request_idx])
                    diff = 1 - sum_values
                    change = round(random.uniform(0, abs(diff)), 2)
                    chosen_idx = random.choice(non_zero_indices)

                    # 尝试更新chosen_idx对应的元素
                    potential_new_value = new_individual['beta_matrix'][chosen_idx][request_idx] + change
                    other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]

                    if potential_new_value > 1:
                        # 如果超过1，则将change加到另一个元素上
                        new_individual['beta_matrix'][other_idx][request_idx] += change
                        new_individual['beta_matrix'][chosen_idx][request_idx] += diff - change
                    else:
                        # 如果没有超过1，则diff - change加到另一个元素上
                        new_individual['beta_matrix'][chosen_idx][request_idx] = potential_new_value
                        new_individual['beta_matrix'][other_idx][request_idx] = 1 - potential_new_value

            else:
                "未满足约束的更新"
                if num_non_zero == 1:
                    model_idx = non_zero_indices[0]
                    new_value = GbestPosition['beta_matrix'][model_idx][request_idx] + (np.random.random()) * s * (
                            np.abs(individual['beta_matrix'][model_idx][request_idx] -GbestPosition_local['beta_matrix'][model_idx][request_idx])
                            + np.abs(individual['beta_matrix'][model_idx][request_idx] - GbestPosition['beta_matrix'][model_idx][request_idx]))
                    new_value = round(new_value, 2)
                    new_value = max(0, min(new_value, 1))
                    new_individual['beta_matrix'][model_idx][request_idx] = new_value
                # 如果有两个元素大于0
                elif num_non_zero == 2:
                    for model_idx in non_zero_indices:
                        new_value = GbestPosition['beta_matrix'][model_idx][request_idx] + (np.random.random()) * s * (
                            np.abs(individual['beta_matrix'][model_idx][request_idx] -GbestPosition_local['beta_matrix'][model_idx][request_idx])
                            + np.abs(individual['beta_matrix'][model_idx][request_idx] - GbestPosition['beta_matrix'][model_idx][request_idx]))
                        new_value = round(new_value, 2)
                        new_value = max(0, min(new_value, 1))
                        new_individual['beta_matrix'][model_idx][request_idx] = new_value

        # 对L矩阵的更新优化
        if num_non_zero == 1:
            # 单AI模型部署
            chosen_layer = random.choice([0, 1])
            new_individual['L_matrix'][chosen_layer][request_idx] = random.choice([1, 2, 3, 4])
            new_individual['L_matrix'][1 - chosen_layer][request_idx] = 0  # 另一行设置为0
        elif num_non_zero == 2:
            # 缝合AI模型部署
            L_values = random.sample([1, 2, 3, 4], 2)
            new_individual['L_matrix'][0][request_idx], new_individual['L_matrix'][1][request_idx] = sorted(L_values)

    # 更新依赖矩阵以反映beta_matrix和L_matrix的变化
    new_individual = update_individual(new_individual)
    return new_individual

# 繁殖行为
def egg_laying_behavior_with_uniform_mutation(individual, user_requests, NETWORK_PARAMS, GbestPosition_local, Lbstar, Ubstar):
    new_individual = deepcopy(individual)

    for request_idx in range(NUM_REQUESTS):
        # 检查当前请求的beta矩阵和是否为1
        sum_beta = sum(individual['beta_matrix'][:, request_idx])
        non_zero_indices = np.where(individual['beta_matrix'][:, request_idx] > 0)[0]
        num_non_zero = len(non_zero_indices)
        or_fitness_q = calculate_individual_q_fitness(individual, request_idx, user_requests[request_idx], NETWORK_PARAMS)

        if sum_beta == 1:
            # 和为1时的特殊处理
            if num_non_zero == 1:
                # 单AI模型部署
                chosen_idx = random.choice([0, 1, 2])
                new_individual['beta_matrix'][:, request_idx] = 0
                new_individual['beta_matrix'][chosen_idx][request_idx] = 1
                new_sin_fitness_q = calculate_individual_q_fitness(new_individual, request_idx, user_requests[request_idx],NETWORK_PARAMS)
                if new_sin_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]
            elif num_non_zero == 2:
                # 缝合AI模型部署
                chosen_idx = random.choice(non_zero_indices)
                new_value = GbestPosition_local['beta_matrix'][chosen_idx][request_idx] + (np.random.random()) * (
                        individual['beta_matrix'][chosen_idx][request_idx] - Lbstar['beta_matrix'][chosen_idx][request_idx]) + (
                    np.random.random()) * (individual['beta_matrix'][chosen_idx][request_idx] - Ubstar['beta_matrix'][chosen_idx][request_idx])
                new_value = round(new_value, 2)
                new_value = max(0, min(new_value, 1))
                new_individual['beta_matrix'][chosen_idx][request_idx] = new_value
                other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]
                new_individual['beta_matrix'][other_idx][request_idx] = 1 - new_value
                new_snt_fitness_q = calculate_individual_q_fitness(new_individual, request_idx, user_requests[request_idx], NETWORK_PARAMS)
                if new_snt_fitness_q < or_fitness_q:
                    # 如果新适应度没有改善，则撤销更新
                    new_individual['beta_matrix'][:, request_idx] = individual['beta_matrix'][:, request_idx]
                    new_individual['L_matrix'][:, request_idx] = individual['L_matrix'][:, request_idx]

        else:
            # 随机决定是否进行直接满足约束的更新
            if random.random() < 0.6:
                if num_non_zero == 1:
                    # 单AI模型部署
                    model_idx = non_zero_indices[0]
                    new_individual['beta_matrix'][model_idx][request_idx] = 1.0
                elif num_non_zero == 2:
                    # 缝合AI模型部署
                    sum_values = sum(individual['beta_matrix'][non_zero_indices, request_idx])
                    diff = 1 - sum_values
                    change = round(random.uniform(0, abs(diff)), 2)
                    chosen_idx = random.choice(non_zero_indices)

                    # 尝试更新chosen_idx对应的元素
                    potential_new_value = new_individual['beta_matrix'][chosen_idx][request_idx] + change
                    other_idx = non_zero_indices[0] if chosen_idx == non_zero_indices[1] else non_zero_indices[1]

                    if potential_new_value > 1:
                        # 如果超过1，则将change加到另一个元素上
                        new_individual['beta_matrix'][other_idx][request_idx] += change
                        new_individual['beta_matrix'][chosen_idx][request_idx] += diff - change
                    else:
                        # 如果没有超过1，则diff - change加到另一个元素上
                        new_individual['beta_matrix'][chosen_idx][request_idx] = potential_new_value
                        new_individual['beta_matrix'][other_idx][request_idx] = 1 - potential_new_value

            else:
                "未满足约束的更新"
                if num_non_zero == 1:
                    model_idx = non_zero_indices[0]
                    new_value = GbestPosition_local['beta_matrix'][model_idx][request_idx] + (np.random.random()) * (
                            individual['beta_matrix'][model_idx][request_idx] - Lbstar['beta_matrix'][model_idx][request_idx]) + (
                                    np.random.random()) * (individual['beta_matrix'][model_idx][request_idx] - Ubstar['beta_matrix'][model_idx][request_idx])
                    new_value = round(new_value, 2)
                    new_value = max(0, min(new_value, 1))
                    new_individual['beta_matrix'][model_idx][request_idx] = new_value
                # 如果有两个元素大于0
                elif num_non_zero == 2:
                    for model_idx in non_zero_indices:
                            new_value = GbestPosition_local['beta_matrix'][model_idx][request_idx] + (np.random.random()) * (
                                        individual['beta_matrix'][model_idx][request_idx] - Lbstar['beta_matrix'][model_idx][request_idx]) + (
                                            np.random.random()) * (individual['beta_matrix'][model_idx][request_idx] - Ubstar['beta_matrix'][model_idx][request_idx])

                            new_value = round(new_value, 2)
                            new_value = max(0, min(new_value, 1))
                            new_individual['beta_matrix'][model_idx][request_idx] = new_value

        #对L矩阵的更新优化
        if num_non_zero == 1:
            # 单AI模型部署
            chosen_layer = random.choice([0, 1])
            new_individual['L_matrix'][chosen_layer][request_idx] = random.choice([1, 2, 3, 4])
            new_individual['L_matrix'][1 - chosen_layer][request_idx] = 0  # 另一行设置为0
        elif num_non_zero == 2:
            # 缝合AI模型部署
            L_values = random.sample([1, 2, 3, 4], 2)
            new_individual['L_matrix'][0][request_idx], new_individual['L_matrix'][1][request_idx] = sorted(L_values)

    # 更新依赖矩阵以反映beta_matrix和L_matrix的变化
    new_individual = update_individual(new_individual)
    return new_individual
def scale_dict_values(dct, scale_factor):
    new_dict = {}
    for key, value in dct.items():
        if isinstance(value, np.ndarray):
            # 直接对 NumPy 数组进行缩放
            new_dict[key] = value * scale_factor
        elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            # 如果是数字列表，先转换为 NumPy 数组再进行缩放
            value_array = np.array(value, dtype=float)
            new_dict[key] = value_array * scale_factor
        else:
            # 如果列表不是数字（比如索引列表），保持不变
            new_dict[key] = value
    return new_dict


# 蜣螂算法种群适应度排序函数
def sort_population_by_fitness(population, fitnesses):
    # 将种群及其适应度打包成元组，然后根据适应度进行排序
    sorted_population_with_fitness = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    # 分离出排序后的种群
    sorted_population = [individual for individual, _ in sorted_population_with_fitness]
    return sorted_population

def DBO(population, population_size, num_iterations, c, d, user_requests, NETWORK_PARAMS):
    """
    :param population_size: 种群数量
    :param num_iterations: 迭代次数
    :param c: 迭代范围下界
    :param d: 迭代范围上界
    :return: 适应度值最小的值 对应得位置
    """
    P_percent = 0.2 #种群不同行为的数量
    pNum = round(population_size * P_percent)
    lb = c
    ub = d
    # 蜣螂算法的适应度和按需服务率
    dbo_fitness_history = []
    dbo_service_rate_history = []
    # 在迭代开始之前初始化 GbestPosition 和 GworstPosition
    dbo_ori_fitnesses, dbo_ori_service_rates = calculate_population_fitness(population, user_requests, NETWORK_PARAMS)
    # 记录初始适应度和按需服务率
    dbo_fitness_history.append(dbo_ori_fitnesses)
    dbo_service_rate_history.append(dbo_ori_service_rates)
    GbestPosition = population[np.argmax(dbo_ori_fitnesses)]
    GworstPosition = population[np.argmin(dbo_ori_fitnesses)]

    for t in range(num_iterations):
        print(f"dbo迭代 {t + 1}/{num_iterations}...")
        iteration_start_time = time.time()

        new_population = deepcopy(population)
        for i in range(pNum):
            individual = population[i]
            dbo_old_individual_fitness, _ = dbo_calculate_individual_fitness(individual, user_requests, NETWORK_PARAMS)
            new_individual = ball_rolling_and_dancing_behavior(individual, GworstPosition, user_requests, NETWORK_PARAMS)
            dbo_new_individual_fitness, _ = dbo_calculate_individual_fitness(new_individual, user_requests, NETWORK_PARAMS)
            if dbo_new_individual_fitness >= dbo_old_individual_fitness:
                new_population[i] = new_individual
            else:
                new_population[i] = individual

        dbo_local_fitnesses, _ = calculate_population_fitness(new_population, user_requests, NETWORK_PARAMS)
        GbestPosition_local = new_population[np.argmax(dbo_local_fitnesses)]
        R = 1 - t/num_iterations
        Lbstar = scale_dict_values(GbestPosition_local, (1 - R))
        Ubstar = scale_dict_values(GbestPosition_local, (1 + R))
        #确保Lbstar和Ubstar在[0, 1]范围内
        # 修改后
        Lbstar = {key: np.clip(value, 0, 1) if isinstance(value, np.ndarray) else value for key, value in
                  Lbstar.items()}
        Ubstar = {key: np.clip(value, 0, 1) if isinstance(value, np.ndarray) else value for key, value in
                  Ubstar.items()}
        #Lbstar = {key: max(0, min(value, 1)) for key, value in Lbstar.items()}
        #Ubstar = {key: max(0, min(value, 1)) for key, value in Ubstar.items()}

        Lbb = scale_dict_values(GbestPosition, (1 - R))
        Ubb = scale_dict_values(GbestPosition, (1 + R))  # Equation(5)
        Lbb = {key: np.clip(value, 0, 1) if isinstance(value, np.ndarray) else value for key, value in
                  Lbb.items()}
        Ubb = {key: np.clip(value, 0, 1) if isinstance(value, np.ndarray) else value for key, value in
                  Ubb.items()}
        #Lbb = {key: max(0, min(value, 1)) for key, value in Lbb.items()}
        #Ubb = {key: max(0, min(value, 1)) for key, value in Ubb.items()}

        for i in range(pNum+1, 120):      # Equation(4)
            individual = population[i]
            dbo_old_individual_fitness, _ = dbo_calculate_individual_fitness(individual, user_requests, NETWORK_PARAMS)
            new_individual = egg_laying_behavior_with_uniform_mutation(individual, user_requests, NETWORK_PARAMS, GbestPosition_local,Lbstar, Ubstar)
            dbo_new_individual_fitness, _ = dbo_calculate_individual_fitness(new_individual, user_requests, NETWORK_PARAMS)
            if dbo_new_individual_fitness >= dbo_old_individual_fitness:
                new_population[i] = new_individual
            else:
                new_population[i] = individual

        for i in range(121, 190):           # Equation(6)
            individual = population[i]
            dbo_old_individual_fitness, _ = dbo_calculate_individual_fitness(individual, user_requests, NETWORK_PARAMS)
            new_individual = foraging_behavior(individual, user_requests, NETWORK_PARAMS, Lbb, Ubb)
            dbo_new_individual_fitness, _ = dbo_calculate_individual_fitness(new_individual, user_requests, NETWORK_PARAMS)
            if dbo_new_individual_fitness >= dbo_old_individual_fitness:
                new_population[i] = new_individual
            else:
                new_population[i] = individual

        for i in range(191, population_size):           # Equation(7)
            individual = population[i]
            dbo_old_individual_fitness, _ = dbo_calculate_individual_fitness(individual, user_requests, NETWORK_PARAMS)
            new_individual = stealing_behavior(individual, GbestPosition, GbestPosition_local, user_requests, NETWORK_PARAMS)
            dbo_new_individual_fitness, _ = dbo_calculate_individual_fitness(new_individual, user_requests, NETWORK_PARAMS)
            if dbo_new_individual_fitness >= dbo_old_individual_fitness:
                new_population[i] = new_individual
            else:
                new_population[i] = individual

        # 更新种群
        population = new_population
        # 计算适应度
        dbo_fitnesses, dbo_service_rates = calculate_population_fitness(population, user_requests, NETWORK_PARAMS)
        # 获取当前最优和最差个体位置
        GbestPosition = population[np.argmax(dbo_fitnesses)]
        GworstPosition = population[np.argmin(dbo_fitnesses)]
        # 记录适应度和按需服务率
        dbo_fitness_history.append(dbo_fitnesses)
        dbo_service_rate_history.append(dbo_service_rates)
        print(f"dbo完成迭代 {t + 1}，总耗时 {time.time() - iteration_start_time:.2f} 秒。\n")
    # 在迭代结束后存储蜣螂算法的最终结果
    dbo_best_fitness_idx = np.argmax(dbo_fitness_history[-1])
    dbo_best_individual = population[dbo_best_fitness_idx]
    dbo_best_fitness = max(dbo_fitness_history[-1])
    dbo_best_service_rate = max(dbo_service_rate_history[-1])

    return dbo_best_fitness, dbo_best_service_rate, dbo_fitness_history, dbo_service_rate_history

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
c = 0
d = 1
dbo_best_fitness, dbo_best_service_rate, dbo_fitness_history, dbo_service_rate_history = DBO(population, population_size, num_iterations, c, d, user_requests, NETWORK_PARAMS)

# 绘制适应度历史
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.max(dbo_fitness_history, axis=1), label='Max Fitness')
plt.plot(np.mean(dbo_fitness_history, axis=1), label='Average Fitness')
plt.title('Fitness over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()

# 绘制按需服务率历史
plt.subplot(1, 2, 2)
plt.plot(np.max(dbo_service_rate_history, axis=1), label='Max Service Rate')
plt.plot(np.mean(dbo_service_rate_history, axis=1), label='Average Service Rate')
plt.title('Service Rate over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Service Rate')
plt.legend()

plt.tight_layout()
plt.savefig('dbo_3.pdf', bbox_inches='tight')
plt.savefig('dbo_3.png', bbox_inches='tight', dpi=600, pad_inches=0.0)
plt.show()
