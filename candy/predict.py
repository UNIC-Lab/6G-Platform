from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

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
X_train, X_test, y_train, y_test = train_test_split(flops.reshape(-1, 1), acc1, test_size=0.2, random_state=42)

# 使用MinMaxScaler来归一化输入数据
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建GradientBoostingRegressor模型
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=4,
                                  min_samples_split=5, min_samples_leaf=3,
                                  max_features=None, loss='ls', random_state=42)

# 训练模型
model.fit(X_train_scaled, y_train)


# 定义预测函数
def predict_performance(gflops):
    # 将GFLOPS转换为FLOPS
    flops = gflops * 1e9

    # 使用训练好的模型进行预测
    flops_scaled = scaler.transform([[flops]])
    acc1_predicted = model.predict(flops_scaled)[0]

    # 峰值FLOPS（假设为336 GFLOPS）转换为FLOPS
    peak_flops = 336 * 1e9

    # 计算推理时间，单位为秒
    inference_time_seconds = flops / peak_flops

    # 将推理时间转换为毫秒
    inference_time_milliseconds = inference_time_seconds * 1000

    return round(acc1_predicted, 4), round(inference_time_milliseconds, 4)

'''# 保存模型和scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')'''
# 现在您可以直接使用predict_performance函数进行预测
gflops_value = 1.0  # 示例的1 GFLOPS
predicted_acc1, predicted_inference_time = predict_performance(gflops_value)
print(f"Predicted ACC1: {predicted_acc1}")
print(f"Predicted Inference Time (ms): {predicted_inference_time}")

