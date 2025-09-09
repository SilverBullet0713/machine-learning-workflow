# 导入所需库
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer  # 数值特征标准化归一化
import category_encoders as ce
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.impute import IterativeImputer
from catboost import CatBoostRegressor  # 典型模型导入与扩展
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # 模型评估RMSE、R2、MAE
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK  # 超参数调整
import shap
import pickle
import csv
import ast
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 配置与全局变量
# 文件路径配置
DATA_OLD_PATH = r"data/data_per_0401.xlsx"
DATA_FP_PATH = r"data/fp_file.xlsx"
TRAIN_DATA_PATH = 'data/per_train_data.csv'
TEST_DATA_PATH = 'data/per_test.csv'
MODEL_SELECTION_RESULTS_PATH = r'results/per_model_selection.xlsx'
HYPEROPT_LOG_PATH = 'results/hyperopt_log.csv'
TRIALS_PATH = r"results/hyperopt_trials.p"
SAVED_MODEL_PATH = 'saved_models/per_model.model'

# 特征列定义
FP_COLUMNS = []
CATEGORICAL_FEATURES = []
NUMERIC_FEATURES = []
TARGET_COLUMN = 'Permeability'

# 全局迭代计数器 (用于超参数优化)
ITERATION_COUNTER = 0

# 1.数据预处理 (Data Preprocessing)
def run_data_preprocessing():
    # 读取数据
    data_old = pd.read_excel(DATA_OLD_PATH)
    data_fp = pd.read_excel(DATA_FP_PATH)

    # 动态定义全局特征列
    global FP_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    FP_COLUMNS = list(data_fp.columns)
    data = pd.concat([data_old, data_fp], axis=1)

    CATEGORICAL_FEATURES = data.select_dtypes(include=['object']).columns.tolist()
    columns_to_exclude = [TARGET_COLUMN] + FP_COLUMNS
    NUMERIC_FEATURES = [col for col in data.select_dtypes(include=['int64', 'float64']).columns if
                        col not in columns_to_exclude]

    # 划分训练集和测试集
    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
    train_index, test_index = next(sss.split(data))
    train_data = data.iloc[train_index].reset_index(drop=True)
    test_data = data.iloc[test_index].reset_index(drop=True)

    # 对数值特征使用MICE插补法 (IterativeImputer)
    cols_positive = ['MWCO', 'Soaking Time', 'Filtration Area']
    imputer = IterativeImputer(random_state=10)

    # 为保证插补值为正，先对数变换，插补后再指数变换恢复
    for col in cols_positive:
        train_data[col] = np.log(train_data[col].clip(lower=1e-6))
        test_data[col] = np.log(test_data[col].clip(lower=1e-6))

    train_data[NUMERIC_FEATURES] = imputer.fit_transform(train_data[NUMERIC_FEATURES])
    test_data[NUMERIC_FEATURES] = imputer.transform(test_data[NUMERIC_FEATURES])

    for col in cols_positive:
        train_data[col] = np.exp(train_data[col])
        test_data[col] = np.exp(test_data[col])

    # 保存处理后的数据
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    print(f"预处理完成，训练数据已保存至: {TRAIN_DATA_PATH}")
    print(f"预处理完成，测试数据已保存至: {TEST_DATA_PATH}")
    print("-" * 30 + "\n")

    # 对给定的数据集进行特征编码和缩放
def preprocess_features(data_to_transform, data_to_fit_on, encoder, scaler):
    # 数据划分
    fp_df = data_to_transform[FP_COLUMNS].copy()
    X_fit = data_to_fit_on.drop(columns=[TARGET_COLUMN] + FP_COLUMNS)
    y_fit = data_to_fit_on[TARGET_COLUMN]

    # 训练编码器和缩放器
    X_fit_encoded = encoder.fit_transform(X_fit, y_fit)
    scaler.fit(X_fit_encoded[NUMERIC_FEATURES])

    # 应用于待转换数据
    X_transform = data_to_transform.drop(columns=[TARGET_COLUMN] + FP_COLUMNS)
    y_transform = data_to_transform[TARGET_COLUMN].values

    X_transform_encoded = encoder.transform(X_transform)

    # 分离编码后的类别和数值部分
    cat_df = X_transform_encoded.drop(columns=NUMERIC_FEATURES)
    num_array = scaler.transform(X_transform_encoded[NUMERIC_FEATURES])
    num_df = pd.DataFrame(data=num_array, columns=NUMERIC_FEATURES, index=data_to_transform.index)

    # 合并最终特征矩阵
    final_X = pd.concat([fp_df, cat_df, num_df], axis=1)

    return final_X, y_transform

# 2.模型筛选 (Model Selection)
def run_model_selection():
    # 通过5折交叉验证，系统地评估不同模型、编码器和缩放器的组合。
    print("--- 开始执行: 模型筛选 ---")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    # 定义候选模型、编码器和缩放器
    models = [
        CatBoostRegressor(verbose=False, random_state=10),
        XGBRegressor(random_state=10),
        SVR(),
        RandomForestRegressor(random_state=10),
        GradientBoostingRegressor(random_state=10),
        LinearRegression(),
        DecisionTreeRegressor(random_state=10),
        AdaBoostRegressor(random_state=10)
    ]
    encoders = [
        ce.BackwardDifferenceEncoder(cols=CATEGORICAL_FEATURES), ce.BaseNEncoder(cols=CATEGORICAL_FEATURES),
        ce.BinaryEncoder(cols=CATEGORICAL_FEATURES), ce.HelmertEncoder(cols=CATEGORICAL_FEATURES),
        ce.JamesSteinEncoder(cols=CATEGORICAL_FEATURES), ce.OneHotEncoder(cols=CATEGORICAL_FEATURES),
        ce.MEstimateEncoder(cols=CATEGORICAL_FEATURES), ce.SumEncoder(cols=CATEGORICAL_FEATURES)
    ]
    scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), PowerTransformer()]

    results_list = []

    for model in models:
        for en in encoders:
            for sc in scalers:
                metrics = {'t_rmse': [], 't_r2': [], 't_mae': [], 'v_rmse': [], 'v_r2': [], 'v_mae': []}

                for train_idx, val_idx in kf.split(train_data):
                    train_fold = train_data.iloc[train_idx].reset_index(drop=True)
                    val_fold = train_data.iloc[val_idx].reset_index(drop=True)

                    x_train_fold, y_train_fold = preprocess_features(train_fold, train_fold, en, sc)
                    x_val_fold, y_val_fold = preprocess_features(val_fold, train_fold, en, sc)

                    model.fit(x_train_fold, y_train_fold)

                    y_train_pred = model.predict(x_train_fold)
                    y_val_pred = model.predict(x_val_fold)

                    metrics['t_rmse'].append(np.sqrt(mean_squared_error(y_train_fold, y_train_pred)))
                    metrics['t_r2'].append(r2_score(y_train_fold, y_train_pred))
                    metrics['t_mae'].append(mean_absolute_error(y_train_fold, y_train_pred))
                    metrics['v_rmse'].append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))
                    metrics['v_r2'].append(r2_score(y_val_fold, y_val_pred))
                    metrics['v_mae'].append(mean_absolute_error(y_val_fold, y_val_pred))

                results_list.append({
                    'train_rmse': np.mean(metrics['t_rmse']), 'test_rmse': np.mean(metrics['v_rmse']),
                    'train_r2': np.mean(metrics['t_r2']), 'test_r2': np.mean(metrics['v_r2']),
                    'train_mae': np.mean(metrics['t_mae']), 'test_mae': np.mean(metrics['v_mae']),
                    'name': model.__class__.__name__, 'encoder': en.__class__.__name__, 'scaler': sc.__class__.__name__
                })

    results = pd.DataFrame(results_list)
    results.sort_values('test_rmse', ascending=True, inplace=True)
    results.to_excel(MODEL_SELECTION_RESULTS_PATH, index=False)

    print(f"模型筛选完成，结果已保存至: {MODEL_SELECTION_RESULTS_PATH}")
    print("表现最佳的前5个组合:")
    print(results.head(5))
    print("-" * 30 + "\n")
    return results.iloc[0]

# 3.模型优化 (Model Optimization)
def run_hyperparameter_optimization(best_encoder_name, best_scaler_name):
    print("--- 开始执行: 模型超参数优化 ---")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    # 根据模型筛选结果，实例化最佳的编码器和缩放器
    sc = PowerTransformer()
    en = ce.BackwardDifferenceEncoder(cols=CATEGORICAL_FEATURES)

    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'depth': hp.quniform('depth', 3, 6, 1),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 3),
        'iterations': hp.quniform('iterations', 600, 800, 10),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 10, 1),
        'random_strength': hp.uniform('random_strength', 0, 5),
        'bagging_temperature': hp.uniform('bagging_temperature', 0.5, 1.0),
    }

    def objective(params):
        global ITERATION_COUNTER
        ITERATION_COUNTER += 1

        # 确保整数参数类型正确
        for name in ['depth', 'iterations', 'min_data_in_leaf']:
            params[name] = int(params[name])

        model = CatBoostRegressor(**params, random_state=10, verbose=False)

        val_rmses = []
        for train_idx, val_idx in kf.split(train_data):
            train_fold = train_data.iloc[train_idx].reset_index(drop=True)
            val_fold = train_data.iloc[val_idx].reset_index(drop=True)

            x_train_fold, y_train_fold = preprocess_features(train_fold, train_fold, en, sc)
            x_val_fold, y_val_fold = preprocess_features(val_fold, train_fold, en, sc)

            model.fit(x_train_fold, y_train_fold)
            y_val_pred = model.predict(x_val_fold)
            val_rmses.append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))

        loss = np.mean(val_rmses)

        # 记录日志
        with open(HYPEROPT_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([loss, params, ITERATION_COUNTER])

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    # 初始化日志文件
    with open(HYPEROPT_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'params', 'iteration'])

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100,
                trials=trials, rstate=np.random.default_rng(50))

    # 保存Trials对象以便后续分析
    with open(TRIALS_PATH, "wb") as f:
        pickle.dump(trials, f)

    # 读取并排序结果
    result = pd.read_csv(HYPEROPT_LOG_PATH)
    result.sort_values('loss', ascending=True, inplace=True)
    best_params_str = result.iloc[0]['params']
    best_params = ast.literal_eval(best_params_str)

    print(f"超参数优化完成，日志已保存至: {HYPEROPT_LOG_PATH}")
    print("找到的最优超参数:")
    print(best_params)
    print("-" * 30 + "\n")
    return best_params

# 4.模型评估 (Model Evaluation)
def train_and_evaluate_final_model(best_params):
    print("--- 开始执行: 最终模型训练与评估 ---")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    # 使用最优的预处理器
    final_scaler = PowerTransformer()
    final_encoder = ce.BackwardDifferenceEncoder(cols=CATEGORICAL_FEATURES)

    x_train, y_train = preprocess_features(train_data, train_data, final_encoder, final_scaler)
    x_test, y_test = preprocess_features(test_data, train_data, final_encoder, final_scaler)

    # 确保整数参数类型正确
    for name in ['depth', 'iterations', 'min_data_in_leaf']:
        best_params[name] = int(best_params[name])

    final_model = CatBoostRegressor(**best_params, random_state=10, verbose=False)
    final_model.fit(x_train, y_train)

    # 保存模型
    final_model.save_model(SAVED_MODEL_PATH)
    print(f"最终模型已训练并保存至: {SAVED_MODEL_PATH}")

    # 在训练集和测试集上评估
    y_train_pred = final_model.predict(x_train)
    y_test_pred = final_model.predict(x_test)

    print("\n最终模型性能评估:")
    print(
        f"  训练集 R2: {r2_score(y_train, y_train_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}, MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(
        f"  测试集 R2: {r2_score(y_test, y_test_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}, MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    print("-" * 30 + "\n")

    return final_model, final_encoder, final_scaler

# 5.模型解释 (Model Interpretation)
def run_model_interpretation(model, encoder, scaler):
    print("--- 开始执行: 模型解释 ---")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    # 准备完整数据集用于解释
    x_train, _ = preprocess_features(train_data, train_data, encoder, scaler)
    x_test, _ = preprocess_features(test_data, train_data, encoder, scaler)
    x_all = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)

    # SHAP分析
    print("正在计算SHAP值...")
    explainer = shap.Explainer(model, x_train)
    shap_values_all = explainer(x_all)

    # 特征过滤
    exclude_prefixes = ["OSM_", "Salt_"]
    exceptions = ["OSM Concentration", "Salt Concentration"]

    keep_idx = []
    selected_features = []
    for idx, col in enumerate(x_all.columns):
        if any(col.startswith(prefix) for prefix in exclude_prefixes) and (col not in exceptions):
            continue
        else:
            keep_idx.append(idx)
            selected_features.append(col)

    shap_values_filtered = shap_values_all.values[:, keep_idx]
    x_all_filtered = x_all[selected_features]

    print("SHAP分析完成，正在生成图表...")

    # SHAP摘要图 (点图)
    plt.figure()
    shap.summary_plot(shap_values_filtered, x_all_filtered, plot_type='dot', show=False)
    plt.title("SHAP Feature Summary Plot")
    plt.tight_layout()
    plt.show()

    # SHAP重要性条形图
    plt.figure()
    shap.summary_plot(shap_values_filtered, x_all_filtered, plot_type='bar', show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    # PDP & ICE 分析 (以 'Monomer A2 concentration' 为例)
    print("\n正在为 'Monomer A2 concentration' 计算PDP/ICE数据...")

    feature_to_plot = 'Monomer A2 concentration'
    if feature_to_plot not in NUMERIC_FEATURES:
        print(f"特征 '{feature_to_plot}' 不在数值特征列表中，跳过PDP/ICE分析。")
        print("-" * 30 + "\n")
        return

    raw_all = pd.concat([train_data, test_data], axis=0)
    raw_medians = train_data[NUMERIC_FEATURES].median()

    subsample_size = min(200, len(x_all))
    X_sample = x_all.sample(subsample_size, random_state=10).reset_index(drop=True)

    raw_min, raw_max = raw_all[feature_to_plot].min(), raw_all[feature_to_plot].max()
    grid_points = np.linspace(raw_min, raw_max, 50)

    grid_df = pd.DataFrame(np.repeat(raw_medians.values[np.newaxis, :], 50, axis=0), columns=NUMERIC_FEATURES)
    grid_df[feature_to_plot] = grid_points

    scaled_grid_array = scaler.transform(grid_df)
    scaled_grid_vals = scaled_grid_array[:, NUMERIC_FEATURES.index(feature_to_plot)]

    ice_curves = np.zeros((len(X_sample), 50))
    X_sample.columns = X_sample.columns.astype(str)  # 确保列名为字符串
    for i, val in enumerate(scaled_grid_vals):
        X_modified = X_sample.copy()
        X_modified[feature_to_plot] = val
        ice_curves[:, i] = model.predict(X_modified)

    # 中心化ICE和PDP曲线
    ice_centered = ice_curves - ice_curves.mean(axis=1, keepdims=True)
    pdp_centered = ice_centered.mean(axis=0)

    print("PDP/ICE数据计算完成，正在生成图表...")

    fig, ax = plt.subplots()
    for curve in ice_centered:
        ax.plot(grid_points, curve, color='lightblue', alpha=0.5)
    ax.plot(grid_points, pdp_centered, color='red', linestyle='--', linewidth=2, label='Centered PDP')

    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel("Centered Prediction")
    ax.set_title(f"ICE and PDP for {feature_to_plot}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("-" * 30 + "\n")

# 6.主函数入口
if __name__ == '__main__':
    run_data_preprocessing()
    # run_model_selection()

    best_params = run_hyperparameter_optimization(
        best_encoder_name='BackwardDifferenceEncoder',
        best_scaler_name='PowerTransformer'
    )

    final_model, final_encoder, final_scaler = train_and_evaluate_final_model(best_params)

    run_model_interpretation(final_model, final_encoder, final_scaler)

    print("所有流程执行完毕")



