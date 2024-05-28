import numpy as np
import json
'''a = {
    "0": [
        43.611757598775974,
        43.86660660902081,
        45.236606388888895,
        47.11804595513555,
        47.11804595513555,
        48.65388227545408,
        48.966759977076364,
        49.308152393618556,
        49.64361438207465,
        49.670046909722224,
        50.06028460916737,
        50.06028460916737,
        50.29252704641236,
        50.34777796522293,
        50.40593478793655,
        50.435966388888886,
        50.44190631944444,
        50.44190631944444,
        50.44567555182544,
        50.459653402777775,
        50.46245979166667,
        50.46245979166667,
        50.466948116111816,
        50.467982630000705,
        50.47186625,
        50.47186625,
        50.473312595278486,
        50.47588628472222,
        50.47825260416667,
        50.479712847222224,
        50.480480729166665,
        50.48121597222222,
        50.48188454861111,
        50.48287534722222,
        50.48292899305555,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225,
        50.482976909722225
    ],
    "0.2": [
        41.354903430256975,
        43.52075624800571,
        43.99679047162853,
        45.99259292934091,
        46.433829814702364,
        46.67603266663831,
        46.84214266705399,
        47.29104277087871,
        47.29104277087871,
        47.48876964630972,
        47.65154851460142,
        47.844067274250854,
        47.913660812388905,
        47.93304506453262,
        47.954592429515586,
        47.97970352819526,
        47.97970352819526,
        47.97970352819526,
        47.98546911286827,
        48.003881074082045,
        48.014682520298656,
        48.01824399427164,
        48.02933929113199,
        48.03088502029866,
        48.03561793822468,
        48.03777301910608,
        48.04449227139768,
        48.04795618836839,
        48.04795618836839,
        48.04795618836839,
        48.04841206660169,
        48.05082806336839,
        48.05082806336839,
        48.051603176982084,
        48.05207512142653,
        48.05207512142653,
        48.052515760547806,
        48.05258516943405,
        48.052991223090615,
        48.053423341146164,
        48.05349486892395,
        48.05355806336839,
        48.05355806336839,
        48.05360320225728,
        48.05366639670172,
        48.05366639670172,
        48.05366639670172,
        48.05366946649074,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726,
        48.053991396701726
    ],
    "0.4": [
        37.80704790828824,
        41.67765131965156,
        43.16220991769893,
        44.99197855735733,
        44.99197855735733,
        47.30204282919743,
        48.34067704299112,
        48.85491876464385,
        49.5205070613636,
        50.62477195701588,
        51.40839271933832,
        51.54719599499872,
        51.54719599499872,
        51.80264813325917,
        51.84477703090869,
        52.017599117997406,
        52.06693996349564,
        52.205020968979355,
        52.22292553451048,
        52.237528129750004,
        52.28293584734645,
        52.28293584734645,
        52.28867666002341,
        52.298419706877425,
        52.32051452249617,
        52.32051452249617,
        52.335254723850994,
        52.33912541878327,
        52.343468396091865,
        52.350129713715546,
        52.35042169470297,
        52.354530904406545,
        52.35838388977999,
        52.35962444607321,
        52.36216173773988,
        52.36335041846178,
        52.364070177121434,
        52.36418076739922,
        52.36471604517699,
        52.36472323578464,
        52.36526340628811,
        52.365523406288105,
        52.36610604846108,
        52.366273579899214,
        52.366273579899214,
        52.36629217553663,
        52.36629217553663,
        52.36629217553663,
        52.36629217553663,
        52.366295234199825,
        52.366295234199825,
        52.36852410073254,
        52.368527159395754,
        52.368527159395754,
        52.368527159395754,
        52.368527159395754,
        52.36852773589885,
        52.36853079456205,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415,
        52.368892263562415
    ],
    "0.6": [
        37.84636398034251,
        38.936692149565026,
        42.27094242558942,
        44.07264030942224,
        45.0344898067094,
        46.19336496054165,
        48.63960151969944,
        48.63960151969944,
        48.63960151969944,
        49.62544425950438,
        49.62544425950438,
        51.04460526313676,
        51.04460526313676,
        51.2167044441737,
        51.22516390979035,
        51.34587740203692,
        51.448854933339476,
        51.60582386878555,
        51.61516404787395,
        51.61559389295586,
        51.61602718237367,
        51.6405256280989,
        51.644604423912426,
        51.65522562905599,
        51.65554229864277,
        51.66665899429161,
        51.66938401358955,
        51.68720474815531,
        51.68725609988522,
        51.695403567636234,
        51.70414229937771,
        51.70414229937771,
        51.70501209510684,
        51.707656891783984,
        51.7077147870277,
        51.71157131335637,
        51.71477715484589,
        51.71582021467292,
        51.719265875305226,
        51.719265875305226,
        51.72131277094621,
        51.725136952848004,
        51.727670298357936,
        51.72784712840618,
        51.72912720797025,
        51.73206424860699,
        51.73500475946567,
        51.73578274962941,
        51.73677836922083,
        51.73677836922083,
        51.73681612209013,
        51.736860464928824,
        51.73692055672083,
        51.736930394923576,
        51.73698561417271,
        51.73705368726547,
        51.737817528671584,
        51.737817528671584,
        51.73782049620277,
        51.737823681720826,
        51.73807628155241,
        51.74066263170671,
        51.74066263170671,
        51.74066263170671,
        51.740673850996856,
        51.74106091321856,
        51.74107213250869,
        51.74107213250869,
        51.74122820889758,
        51.74122820889758,
        51.74124345004593,
        51.74124345004593,
        51.74124345004593,
        51.74151935473091,
        51.74151935473091,
        51.74156362556424,
        51.74156362556424,
        51.74156362556424,
        51.74157313368116,
        51.74157313368116,
        51.742136442808246,
        51.742136442808246,
        51.742136442808246,
        51.742136442808246,
        51.742136442808246,
        51.74214595092516,
        51.74214595092516,
        51.742162950552526,
        51.742162950552526,
        51.742162950552526,
        51.74216683976999,
        51.74216683976999,
        51.74393757602694,
        51.74393757602694,
        51.74393757602694,
        51.74394146524439,
        51.74394146524439,
        51.743942925789,
        51.74395114002549,
        51.74395114002549
    ],
    "0.8": [
        33.420960158447045,
        38.43495623028005,
        39.12875132758874,
        40.913993533910975,
        41.14566925001737,
        43.56709742581295,
        43.56709742581295,
        43.87754599395203,
        45.55701605215435,
        45.734663338733036,
        46.035195105258886,
        46.257886298407016,
        46.262477035544336,
        46.426978623359396,
        46.44526597793203,
        46.60924658470576,
        46.622541118692325,
        46.622541118692325,
        46.63927649269499,
        46.652800099982336,
        46.65741378693314,
        46.65741378693314,
        46.665619895992286,
        46.67625381834526,
        46.67792585134622,
        46.693779498693466,
        46.720806070753135,
        46.720806070753135,
        46.72268631103381,
        46.72809849311567,
        46.732922917804345,
        46.73676479745776,
        46.74238267370164,
        46.743436757927455,
        46.75008163399448,
        46.75461307990733,
        46.75461307990733,
        46.75979866080248,
        46.75979866080248,
        46.76039161228107,
        46.76204590605795,
        46.763053035496995,
        46.764586989746945,
        46.76499225963002,
        46.76499225963002,
        46.76659165390958,
        46.7670411831433,
        46.767130617029004,
        46.76751406155247,
        46.76800275967169,
        46.76950291035128,
        46.76985406645061,
        46.77027271105977,
        46.77083130240818,
        46.771635955137434,
        46.77309129210859,
        46.77309129210859,
        46.77443251717355,
        46.77443251717355,
        46.77444389559825,
        46.77444487926158,
        46.774575233691394,
        46.774575233691394,
        46.77473645914968,
        46.77473645914968,
        46.77476731391937,
        46.774900676813644,
        46.774900676813644,
        46.774900676813644,
        46.77493197499655,
        46.77493197499655,
        46.77493197499655,
        46.77493217289613,
        46.77497259999655,
        46.77501009999655,
        46.77502364166321,
        46.775064266663215,
        46.775064266663215,
        46.775064266663215,
        46.775064266663215,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275,
        46.77508494578275
    ],
    "1": [
        33.96375931955467,
        37.368069366810666,
        39.74722806634492,
        39.85672236593142,
        42.58427805489616,
        43.08070853496324,
        44.083784271679725,
        44.116049122498026,
        44.85462661502105,
        45.345959835927495,
        45.47633792812907,
        46.034302862227065,
        46.034302862227065,
        46.034302862227065,
        46.11099439095398,
        46.31883666745971,
        46.34806380717008,
        46.50732972151844,
        46.55011481228516,
        46.68550865332334,
        46.70022118172869,
        46.72391383202707,
        46.72917579808783,
        46.72917579808783,
        46.73207151215801,
        46.73905447670931,
        46.741083242017154,
        46.76338160774462,
        46.76611774030543,
        46.76840487277023,
        46.779786747435566,
        46.779786747435566,
        46.781621491533905,
        46.789108759657296,
        46.79638425959583,
        46.80489649803352,
        46.806107087281625,
        46.81240035003682,
        46.81333389127448,
        46.81876488225651,
        46.819341543815085,
        46.82865312486389,
        46.82865312486389,
        46.83340520686227,
        46.83391927154959,
        46.83620750493123,
        46.83718821032455,
        46.83977985648277,
        46.84241047155546,
        46.84465228295979,
        46.844797282959796,
        46.846387389393314,
        46.846812253553935,
        46.84937428253004,
        46.84937428253004,
        46.85042345489491,
        46.851898069862045,
        46.85261869350361,
        46.85438531040105,
        46.855116933862924,
        46.855116933862924,
        46.85634477433935,
        46.85634477433935,
        46.85753924143867,
        46.85822260238608,
        46.85980664815868,
        46.86053286343646,
        46.86164649459268,
        46.861727330535786,
        46.864403798703336,
        46.86475707619379,
        46.86540645119379,
        46.865415873292214,
        46.86551034729904,
        46.866082393300374,
        46.866206941184146,
        46.866267165291816,
        46.86627440718926,
        46.86676898329138,
        46.86685718376909,
        46.86685718376909,
        46.86685718376909,
        46.86695155441622,
        46.86695831207821,
        46.86713931219486,
        46.86714597886153,
        46.86714597886153,
        46.86892544885593,
        46.86893218376909,
        46.86922097886153,
        46.869365978861524,
        46.86944603870732,
        46.87138630808893,
        46.87138630808893,
        46.87140630808893,
        46.871466367934715,
        46.87148964142226,
        46.87188365273127,
        46.872582974755595,
        46.872582974755595
    ]
}
averages = {key: np.mean(values) for key, values in a.items()}
print(averages)'''

# 保存为 JSON 文件
def save_as_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# 定义读取 JSON 数据的函数
def load_json_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
A=[]
'''a = load_json_data('D:\\pydoc\\candy\\main\\algorithm_5_dbo_fitness_history_0.json')
b = load_json_data('D:\\pydoc\\candy\\main\\algorithm_5_dbo_fitness_history_1.json')
c = np.mean(a, axis = 1)
d = np.max(b, axis = 1)'''

#averages = [sum(position_values) / len(position_values) for position_values in zip(*A)]
#print(averages)
a = [3,6,2,9,1]
b = [8,5,2,8,6]
A.append(a)
A.append(b)
max_values = [max(position_values) for position_values in zip(*A)]
print(max_values)