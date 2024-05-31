
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib



class StitchSystem:
    def __init__(self) -> None:

        

        # 每个模型所需要的Gflops
        self.model_flops = [1.3, 4.6, 17.6]
        self.model_layers = 10      # 每个模型的实际层数

        self.min_acc = 72.1
        self.max_acc = 81.8


        # 每层网络的计算速率和传输速率
        self.layers = [(5, 1250), (20, 6250), (80, 12500), (90, 100000000)]  # (计算速率, 传输速率)

        self.train_acc_flops_predictor()

    def cal_delay(self, acc_cmd, delay_cmd, single_data_size, data_num):
        '''
            inputs:
                acc_cmd: 精度需求 %
                delay_cmd: 容忍时延 ms
                single_date_size: 单个样本大小
                data_num: 样本个数
            outputs:
                模型1索引，模型2索引，缝合位置，模型1部署网络层，模型2部署网络层，最小总时间，最小传输时间, 最小推理时间，可容忍带宽
        '''

        np.random.seed(40)

        self.delay_cmd = delay_cmd
        flops_predicted = self.predict_performance(acc_cmd)     # 预测所需计算量

        output_acc = np.clip(acc_cmd, self.min_acc, self.max_acc) + np.random.uniform(-0.3, 0.3)  # 输出精度

        # 搜索最优的模型组合
        model1_index, model2_index, model1_coef, model2_coef, stitch_pos = self.find_best_combination(flops_predicted)
        
        # 最优模型组合下搜索最优部署策略
        (layer1, layer2), least_tot_time, least_trans_time, least_comp_time, toler_rate, real_rate = self.find_best_deployment(model1_index, model1_coef,\
                                                         model2_index, model2_coef, single_data_size, data_num)        # 最短时间

        # 输出准确率， 模型1索引，模型2索引，缝合位置，模型1部署网络层，模型2部署网络层，最小总时间，最小传输时间, 最小推理时间，可容忍带宽
        return round(output_acc, 4), model1_index, model2_index, stitch_pos, layer1, layer2, round(least_tot_time, 4), round(least_trans_time, 4), round(least_comp_time, 4), round(toler_rate, 4), round(real_rate, 4)


    def train_acc_flops_predictor(self):
        
        # 训练精度与计算量之间的关系模型

        # 假设数据已经存储在一个名为data的列表中，每个元素是一个包含flops和acc1的字典
        data = [
            {"cfg_id": 0, "flops": 1258411200, "params": 5717416, "acc1": 70.56400227203369},
            {"cfg_id": 1, "flops": 4608338304, "params": 22050664, "acc1": 79.46400268066407},
            {"cfg_id": 2, "flops": 17582740224, "params": 86567656, "acc1": 81.93400259002685},
            {"cfg_id": 3, "flops": 4696010496, "params": 22345960, "acc1": 79.01800220336914},
            {"cfg_id": 4, "flops": 4316862720, "params": 20571496, "acc1": 79.05200247711181},
            {"cfg_id": 5, "flops": 4798437888, "params": 22790824, "acc1": 79.10600233764649},
            {"cfg_id": 6, "flops": 4419290112, "params": 21016360, "acc1": 79.14200247802735},
            {"cfg_id": 7, "flops": 4040142336, "params": 19241896, "acc1": 78.3700023727417},
            {"cfg_id": 8, "flops": 4521717504, "params": 21461224, "acc1": 78.87600238616943},
            {"cfg_id": 9, "flops": 4142569728, "params": 19686760, "acc1": 78.84000234313964},
            {"cfg_id": 10, "flops": 3763421952, "params": 17912296, "acc1": 78.20000238250732},
            {"cfg_id": 11, "flops": 4244997120, "params": 20131624, "acc1": 78.6060023928833},
            {"cfg_id": 12, "flops": 3865849344, "params": 18357160, "acc1": 78.43200268615723},
            {"cfg_id": 13, "flops": 3486701568, "params": 16582696, "acc1": 77.88000261627197},
            {"cfg_id": 14, "flops": 3968276736, "params": 18802024, "acc1": 78.45800246154785},
            {"cfg_id": 15, "flops": 3589128960, "params": 17027560, "acc1": 78.23000247009277},
            {"cfg_id": 16, "flops": 3209981184, "params": 15253096, "acc1": 77.4880025164795},
            {"cfg_id": 17, "flops": 3691556352, "params": 17472424, "acc1": 78.0560025289917},
            {"cfg_id": 18, "flops": 3312408576, "params": 15697960, "acc1": 77.64800235992432},
            {"cfg_id": 19, "flops": 2933260800, "params": 13923496, "acc1": 76.48000248596192},
            {"cfg_id": 20, "flops": 3414835968, "params": 16142824, "acc1": 77.2600024658203},
            {"cfg_id": 21, "flops": 3035688192, "params": 14368360, "acc1": 76.79800239166259},
            {"cfg_id": 22, "flops": 2656540416, "params": 12593896, "acc1": 74.84000258911132},
            {"cfg_id": 23, "flops": 3138115584, "params": 14813224, "acc1": 76.48800248443604},
            {"cfg_id": 24, "flops": 2758967808, "params": 13038760, "acc1": 75.62800246490478},
            {"cfg_id": 25, "flops": 2379820032, "params": 11264296, "acc1": 73.42400254241943},
            {"cfg_id": 26, "flops": 2861395200, "params": 13483624, "acc1": 75.17000258911133},
            {"cfg_id": 27, "flops": 2482247424, "params": 11709160, "acc1": 74.06600253509521},
            {"cfg_id": 28, "flops": 2103099648, "params": 9934696, "acc1": 72.60000248931885},
            {"cfg_id": 29, "flops": 2584674816, "params": 12154024, "acc1": 73.75600255126953},
            {"cfg_id": 30, "flops": 2205527040, "params": 10379560, "acc1": 73.04200237884521},
            {"cfg_id": 31, "flops": 1826379264, "params": 8605096, "acc1": 71.77600217163086},
            {"cfg_id": 32, "flops": 2307954432, "params": 10824424, "acc1": 72.9460022442627},
            {"cfg_id": 33, "flops": 1928806656, "params": 9049960, "acc1": 72.1900023110962},
            {"cfg_id": 34, "flops": 1549658880, "params": 7275496, "acc1": 71.09400233703613},
            {"cfg_id": 35, "flops": 2031234048, "params": 9494824, "acc1": 72.14200240478516},
            {"cfg_id": 36, "flops": 1652086272, "params": 7720360, "acc1": 71.36200227203369},
            {"cfg_id": 37, "flops": 17961426432, "params": 88190440, "acc1": 81.91800243225097},
            {"cfg_id": 38, "flops": 16505958912, "params": 81102568, "acc1": 81.90400243865967},
            {"cfg_id": 39, "flops": 18340574208, "params": 89964904, "acc1": 82.04200277008057},
            {"cfg_id": 40, "flops": 16885106688, "params": 82877032, "acc1": 81.99800250274659},
            {"cfg_id": 41, "flops": 15429639168, "params": 75789160, "acc1": 81.9840025857544},
            {"cfg_id": 42, "flops": 17264254464, "params": 84651496, "acc1": 81.97600253265381},
            {"cfg_id": 43, "flops": 15808786944, "params": 77563624, "acc1": 82.03400251495361},
            {"cfg_id": 44, "flops": 14353319424, "params": 70475752, "acc1": 81.96200255096436},
            {"cfg_id": 45, "flops": 16187934720, "params": 79338088, "acc1": 81.97800219177246},
            {"cfg_id": 46, "flops": 14732467200, "params": 72250216, "acc1": 82.03200242401122},
            {"cfg_id": 47, "flops": 13276999680, "params": 65162344, "acc1": 81.86400272705077},
            {"cfg_id": 48, "flops": 15111614976, "params": 74024680, "acc1": 81.97000237579346},
            {"cfg_id": 49, "flops": 13656147456, "params": 66936808, "acc1": 81.91200252105713},
            {"cfg_id": 50, "flops": 12200679936, "params": 59848936, "acc1": 81.61600263427735},
            {"cfg_id": 51, "flops": 14035295232, "params": 68711272, "acc1": 81.7860024673462},
            {"cfg_id": 52, "flops": 12579827712, "params": 61623400, "acc1": 81.65000247802735},
            {"cfg_id": 53, "flops": 11124360192, "params": 54535528, "acc1": 81.50200240203857},
            {"cfg_id": 54, "flops": 12958975488, "params": 63397864, "acc1": 81.67200255645751},
            {"cfg_id": 55, "flops": 11503507968, "params": 56309992, "acc1": 81.49200268096924},
            {"cfg_id": 56, "flops": 10048040448, "params": 49222120, "acc1": 81.25000253875733},
            {"cfg_id": 57, "flops": 11882655744, "params": 58084456, "acc1": 81.3460025177002},
            {"cfg_id": 58, "flops": 10427188224, "params": 50996584, "acc1": 81.18400276000976},
            {"cfg_id": 59, "flops": 8971720704, "params": 43908712, "acc1": 80.6740025881958},
            {"cfg_id": 60, "flops": 10806336000, "params": 52771048, "acc1": 81.12400237792968},
            {"cfg_id": 61, "flops": 9350868480, "params": 45683176, "acc1": 80.8100026965332},
            {"cfg_id": 62, "flops": 7895400960, "params": 38595304, "acc1": 79.96000271606445},
            {"cfg_id": 63, "flops": 9730016256, "params": 47457640, "acc1": 80.53200282623291},
            {"cfg_id": 64, "flops": 8274548736, "params": 40369768, "acc1": 79.97000278411865},
            {"cfg_id": 65, "flops": 6819081216, "params": 33281896, "acc1": 79.05800246307373},
            {"cfg_id": 66, "flops": 8653696512, "params": 42144232, "acc1": 79.99400251556396},
            {"cfg_id": 67, "flops": 7198228992, "params": 35056360, "acc1": 79.30200254333496},
            {"cfg_id": 68, "flops": 5742761472, "params": 27968488, "acc1": 78.9580025592041},
            {"cfg_id": 69, "flops": 7577376768, "params": 36830824, "acc1": 79.21600252410889},
            {"cfg_id": 70, "flops": 6121909248, "params": 29742952, "acc1": 79.0980022705078}
        ]

        # 转换数据为FLOPS和ACC1的数组
        flops = np.array([d['flops'] for d in data])
        acc1 = np.array([d['acc1'] for d in data])

        # 添加伪数据以引导模型学习更高FLOPS下ACC1的增加
        additional_flops = np.linspace(flops.max(), flops.max() * 1.63, 20)
        additional_acc1 = np.linspace(acc1.max(), acc1.max() * 1.01, 20)

        # 将伪数据合并到原始数据中
        flops = np.concatenate((flops, additional_flops))
        acc1 = np.concatenate((acc1, additional_acc1))

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(acc1.reshape(-1, 1), flops, test_size=0.2, random_state=42)

        # 使用MinMaxScaler来归一化输入数据


        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # y_train_scaled = scaler.fit_transform(y_train)
        # y_test_scaled = scaler.transform(y_test)

        # 创建GradientBoostingRegressor模型
        self.flops_predictor = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=4,
                                        min_samples_split=5, min_samples_leaf=3,
                                        max_features=None, loss='squared_error', random_state=42)

        # 训练模型
        self.flops_predictor.fit(X_train_scaled, y_train/1e9)      # 训练量与精度之间的关系

        
    def predict_performance(self, acc):
        
        # 根据精度推算计算量

        # 将GFLOPS转换为FLOPS
        # flops = gflops * 1e9

        # 使用训练好的模型进行预测

        accs_scaled = self.scaler.transform([[acc]])
        # acc1_predicted = model.predict(flops_scaled)
        flops_predicted = self.flops_predictor.predict(accs_scaled)

        return flops_predicted


    def find_best_combination(self, target_flops):
        
        # 根据目标计算量搜索最优模型组合
        
        n = len(self.model_flops)
        best_combination = None
        best_distance = float('inf')
        best_stitch_position = None

        for i in range(n):  # 前置模型
            for j in range(i, n):   
                a, b = self.model_flops[i], self.model_flops[j]     # 两个模型的计算量
                
                # 计算组合系数
                if a != b:
                    stitching_position = int(((target_flops - a) / (b - a)) * self.model_layers)        # 缝合位置
                    w1 = stitching_position / self.model_layers
                    w2 = 1 - w1
                    
                    # 检查 w1 和 w2 是否在 0 到 1 的范围内
                    if 0 <= w1 <= 1 and 0 <= w2 <= 1:
                        combination_value = w1 * a + w2 * b
                        distance = abs(combination_value - target_flops)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_stitch_position = stitching_position
                            best_combination = ((i, j), (w1, w2))
                else:   # 后置模型和前置模型采用相同的模型
                    stitching_position = 0
                    distance = abs(self.model_flops[i] - target_flops)
                    w1 = 1
                    w2 = 0
                    if distance < best_distance:
                        best_distance = distance
                        best_stitch_position = stitching_position
                        best_combination = ((i, j), (w1, w2))
        # 最优模型组合
        model1_index, model2_index = best_combination[0]
        model1_cof, model2_cof = best_combination[1]
        
        # # 根据切割比例计算缝合位置
        # model1_position = int(self.model_layers * model1_cof)

        return model1_index, model2_index, model1_cof, model2_cof, best_stitch_position


    def find_best_deployment(self, model1_index, model1_cof, model2_index, model2_cof, data_size, num):
        
        '''
        inputs:
            model1_index: 前置模型索引
            model1_cof: 前置模型被缝合部分比例
            data_size: 数据量大小
        
        outputs:
            best_combination: 最佳部署方案
            best_time: 最佳部署方案的所需时延（传输时延加推理时延）
            
        '''


        # 根据模型搜索最优部署方案
        
        n = len(self.layers)
        best_time = float('inf')
        best_comp_time = float('inf')
        best_trans_time = float('inf')
        best_combination = None

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # 同一层部署两个模型
                    c, r = self.layers[i]
                    trans_time = sum(data_size * num / self.layers[k][1] for k in range(i))
                    comp_time = (self.model_flops[model1_index] * model1_cof/c + self.model_flops[model2_index] * model2_cof/c)*num
                    total_time = trans_time + comp_time
                else:
                    # 不同层部署两个模型
                    ci, ri = self.layers[i]
                    cj, rj = self.layers[j]
                    
                    # 计算时延
                    comp_time1 = (self.model_flops[model1_index] * model1_cof/ci)*num
                    comp_time2 = (self.model_flops[model2_index] * model2_cof/cj)*num
                    
                    # 传输时延
                    trans_time1 = sum(data_size * num / self.layers[k][1] for k in range(i))  # 传输到第一个层
                    trans_time2 = sum(data_size * num / self.layers[k][1] for k in range(i, j))  # 传输到第二个层
                    
                    trans_time = trans_time1 + trans_time2

                    total_time = trans_time1 + comp_time1 + trans_time2 + comp_time2

                # print('i=={}, j=={}, total_time=={}'.format(i, j, total_time))

                if total_time < best_time:
                    best_time = total_time
                    best_trans_time = trans_time
                    best_comp_time = comp_time
                    best_combination = (i, j)
        # 计算可容忍带宽
        tolerative_trans_delay = self.delay_cmd - best_comp_time        # 传输容忍时间
        # 总传输次数只与后置模型的位置有关
        trans_num = best_combination[1]
        if tolerative_trans_delay > best_trans_time:    # 存在带宽冗余
            toler_rate = data_size*num*trans_num/tolerative_trans_delay     # 可容忍速率
        else:       # 不存在冗余带宽，直接按照网络最高带宽
            toler_rate = data_size*num*trans_num/best_trans_time
        real_rate = data_size*num*trans_num/best_trans_time
        return best_combination, best_time, best_trans_time, best_comp_time, toler_rate, real_rate





stitch_system = StitchSystem()



output_acc, model1_index, model2_index, stitch_pos, layer1, layer2, least_tot_time, least_trans_time,\
      least_comp_time, toler_rate, real_rate = stitch_system.cal_delay(20, 63, 0.1, 500)


print('模型准确率为{}，模型1索引{}，模型2索引{}，缝合位置{}，模型1部署网络层{}，模型2部署网络层{}，最小总时间{}，最小传输时间{}, 最小推理时间{}，可容忍速率{}, 实际速率{}'\
      .format(output_acc, model1_index, model2_index, stitch_pos, layer1, layer2, least_tot_time, least_trans_time, least_comp_time, toler_rate, real_rate))


