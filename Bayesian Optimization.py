import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import warnings  # 过滤警告信息
warnings.filterwarnings("ignore")

# 读取训练集
train_per = pd.read_csv('per_train_data.csv')
train_sel = pd.read_csv('sel_train_data.csv')

# 指纹列、分类列、数值列定义
fp_columns = [c for c in train_per.columns if c.startswith(('Monomer_A_FP', 'Monomer_B_FP', 'Additive_FP'))]
categorical_features = train_per.select_dtypes(include=['object']).columns.tolist()
numeric_features_per = [c for c in train_per.columns
                        if c not in fp_columns + ['Permeability'] + categorical_features]
numeric_features_sel = [c for c in train_sel.columns
                        if c not in fp_columns + ['Selectivity'] + categorical_features]

# 通用预处理函数
def conv_data_pd(data, data_fit, encoder, scaler, fp_columns, numeric_features):
    fp_pd = data[fp_columns].copy()
    X_fit = data_fit.drop(columns=[encoder.target_col] + fp_columns)
    y_fit = data_fit[encoder.target_col]
    X_enc = encoder.fit_transform(X_fit, y_fit)              # 分类特征处理
    scaler.fit(X_enc[numeric_features])                      # 数值特征处理
    X_all = data.drop(columns=[encoder.target_col] + fp_columns)
    X_all_enc = encoder.transform(X_all)                     # 完全转换
    col_pd = X_all_enc.drop(columns=numeric_features)
    num_pd = pd.DataFrame(scaler.transform(X_all_enc[numeric_features]), columns=numeric_features)
    return pd.concat([fp_pd, col_pd, num_pd], axis=1)

# 加载模型
per_model = CatBoostRegressor()
per_model.load_model('per_model.model')
sel_model = XGBRegressor()
sel_model.load_model('sel_model.model')
# 固定特征
fixed_feats = {
    'Monomer A1 concentration': 0, 'Substrate': 'PES', 'MWCO': 150, 'Test Temperature': 25,
    'Filtration Area': 24, 'Salt': 'Na2SO4', 'Salt Concentration': 0.2,
    'Average hydration radius': 0.369, 'OSM': 'CR', 'OSM Concentration': 0.01,
    'Mw': 696.68, 'Charge': -2, 'Log D': 2.362
}
# 固定单体
fp_path = r"D:\PyCharm\Py.Projects\Permeability\Morgan_2048.xlsx"
naoh_fp = pd.read_excel(fp_path, sheet_name='Sheet1', index_col='Molecule').loc['NaOH'].to_dict()
rt_fp = pd.read_excel(fp_path, sheet_name='Sheet2', index_col='Molecule').loc['RT'].to_dict()
tmc_fp = pd.read_excel(fp_path, sheet_name='Sheet3', index_col='Molecule').loc['TMC'].to_dict()
fixed_fp = {**naoh_fp, **rt_fp, **tmc_fp}                   # 合并单体摩根指纹信息

# 两套 encoder/scaler
en_per = ce.BaseNEncoder(cols=categorical_features); en_per.target_col= 'Permeability'
sc_per = StandardScaler()
en_sel = ce.sum_coding.SumEncoder(cols=categorical_features); en_sel.target_col= 'Selectivity'
sc_sel = StandardScaler()

# Optuna多目标函数预测
def objective(trial):
    var = {
      'Additive concentration':    trial.suggest_float('Ad conc', 0, 2, step=0.01, log=False),
      'Monomer A2 concentration':  trial.suggest_float('A2 conc', 0, 2, step=0.01, log=False),
      'Monomer B concentration':   trial.suggest_float('B conc', 0, 1, step=0.01, log=False),
      'Curing Temperature':        trial.suggest_float('CT', 25, 100, step=1, log=False),
      'Curing Time':               trial.suggest_float('CTime', 1, 25, step=1, log=False),
      'Soaking Time':              trial.suggest_float('STime', 1, 10, step=0.1, log=False),
      'Reaction Time':             trial.suggest_float('RTime', 0, 5, step=0.1, log=False),
      'Operation Pressure':        trial.suggest_float('OP', 1, 6, step=0.5, log=False)
    }
    sample = pd.DataFrame([{**fixed_feats, **fixed_fp, **var}])

    # Permeability
    comb_per = pd.concat([train_per, sample], ignore_index=True)
    x_per_pd = conv_data_pd(comb_per, train_per, en_per, sc_per, fp_columns, numeric_features_per)
    p = per_model.predict(x_per_pd.tail(1).values)[0]

    # Selectivity
    comb_sel = pd.concat([train_sel, sample], ignore_index=True)
    x_sel_pd = conv_data_pd(comb_sel, train_sel, en_sel, sc_sel, fp_columns, numeric_features_sel)
    s_log = sel_model.predict(x_sel_pd.tail(1).values)[0]
    s = np.exp(s_log)                                        # Selectivity还原原始尺度

    # 返回负值以 maximize 原值
    return -p, -s

study = optuna.create_study(directions=['minimize','minimize'])
study.optimize(objective, n_trials=1000)

# 保存并显示 Pareto 前沿
pareto = study.best_trials
rows = []
for t in pareto:
    p_val = -t.values[0]
    s_val = -t.values[1]
    rows.append({**t.params, 'Permeability': p_val, 'Selectivity': s_val})
df = pd.DataFrame(rows)
df.to_excel('NaOH-RT-TMC-CR&Na2SO4.xlsx', index=False)