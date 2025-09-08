# 导入所需库
import pandas as pd
import numpy as np
import category_encoders as ce
import shap
import pickle
import csv
import ast
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

warnings.filterwarnings("ignore")

# 配置与全局变量
# 文件路径配置
DATA_OLD_PATH = r"data/data_sel_0406.xlsx"
DATA_FP_PATH = r"data/fp_file.xlsx"
TRAIN_DATA_PATH = 'data/sel_train_data.csv'
TEST_DATA_PATH = 'data/sel_test.csv'
MODEL_SELECTION_RESULTS_PATH = r'results/sel_model_selection.xlsx'
HYPEROPT_LOG_PATH = 'results/sel_hyperopt_log.csv'
TRIALS_PATH = r"results/sel_hyperopt_trials.p"
SAVED_MODEL_PATH = 'saved_models/sel_model.model'

# 特征列定义
FP_COLUMNS = []
CATEGORICAL_FEATURES = []
NUMERIC_FEATURES = []
TARGET_COLUMN = 'Selectivity'

# 全局迭代计数器 (用于超参数优化)
ITERATION_COUNTER = 0


# 1.数据预处理 (Data Preprocessing)
def run_data_preprocessing():
    print("--- 开始执行: 数据预处理 ---")
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
    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=50)
    train_index, test_index = next(sss.split(data))
    train_data = data.iloc[train_index].reset_index(drop=True)
    test_data = data.iloc[test_index].reset_index(drop=True)

    # 对数值特征使用MICE插补法
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
    # 对目标值进行对数变换
    y_transform = np.log(data_to_transform[TARGET_COLUMN].values)

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
                metrics = {'v_rmse': []}
                for train_idx, val_idx in kf.split(train_data):
                    train_fold = train_data.iloc[train_idx].reset_index(drop=True)
                    val_fold = train_data.iloc[val_idx].reset_index(drop=True)

                    x_train_fold, y_train_fold = preprocess_features(train_fold, train_fold, en, sc)
                    x_val_fold, y_val_fold = preprocess_features(val_fold, train_fold, en, sc)

                    model.fit(x_train_fold, y_train_fold)
                    y_val_pred = model.predict(x_val_fold)
                    metrics['v_rmse'].append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))

                results_list.append({
                    'test_rmse': np.mean(metrics['v_rmse']),
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
def run_hyperparameter_optimization():
    print("--- 开始执行: 模型超参数优化 ---")
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    # 根据模型筛选结果，实例化最佳的编码器和缩放器
    sc = MaxAbsScaler()
    en = ce.MEstimateEncoder(cols=CATEGORICAL_FEATURES)

    # 定义XGBoost的超参数空间
    space = {
        'max_delta_step': hp.uniform('max_delta_step', 0, 10),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.quniform('max_depth', 4, 7, 1),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'subsample': hp.uniform('subsample', 0.7, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0, 2),
        'gamma': hp.uniform('gamma', 0, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 50)
    }

    def objective(params):
        global ITERATION_COUNTER
        ITERATION_COUNTER += 1

        # 确保整数参数类型正确
        for name in ['max_depth', 'n_estimators']:
            params[name] = int(params[name])

        model = XGBRegressor(**params, random_state=10)
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
        with open(HYPEROPT_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([loss, params, ITERATION_COUNTER])
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    with open(HYPEROPT_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'params', 'iteration'])
    trials = Trials()
    fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100,
         trials=trials, rstate=np.random.default_rng(50))
    with open(TRIALS_PATH, "wb") as f:
        pickle.dump(trials, f)
    result = pd.read_csv(HYPEROPT_LOG_PATH)
    result.sort_values('loss', ascending=True, inplace=True)
    best_params = ast.literal_eval(result.iloc[0]['params'])

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

    final_scaler = MaxAbsScaler()
    final_encoder = ce.MEstimateEncoder(cols=CATEGORICAL_FEATURES)
    x_train, y_train = preprocess_features(train_data, train_data, final_encoder, final_scaler)
    x_test, y_test = preprocess_features(test_data, train_data, final_encoder, final_scaler)

    for name in ['max_depth', 'n_estimators']:
        best_params[name] = int(best_params[name])
    final_model = XGBRegressor(**best_params, random_state=10)
    final_model.fit(x_train, y_train)
    final_model.save_model(SAVED_MODEL_PATH)
    print(f"最终模型已训练并保存至: {SAVED_MODEL_PATH}")

    y_train_pred = final_model.predict(x_train)
    y_test_pred = final_model.predict(x_test)
    print("\n最终模型性能评估 (基于对数变换后的值):")
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

    x_train, _ = preprocess_features(train_data, train_data, encoder, scaler)
    x_test, _ = preprocess_features(test_data, train_data, encoder, scaler)
    x_all = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)

    print("正在计算SHAP值...")
    explainer = shap.Explainer(model, x_train)
    shap_values_all = explainer(x_all)

    exclude_prefixes = ["OSM", "Salt"]
    exceptions = ["OSM Concentration", "Salt Concentration"]
    keep_idx, selected_features = [], []
    for idx, col in enumerate(x_all.columns):
        if any(col.startswith(prefix) for prefix in exclude_prefixes) and (col not in exceptions): continue
        selected_features.append(col)
        keep_idx.append(idx)
    shap_values_filtered = shap_values_all.values[:, keep_idx]
    x_all_filtered = x_all[selected_features]

    print("SHAP分析完成，正在生成图表...")
    plt.figure()
    shap.summary_plot(shap_values_filtered, x_all_filtered, plot_type='dot', show=False)
    plt.title("SHAP Feature Summary Plot")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values_filtered, x_all_filtered, plot_type='bar', show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    print("\n正在计算PDP/ICE数据...")
    features_to_plot = [
        'Monomer A2 concentration', 'Monomer B concentration', 'Additive concentration',
        'Curing Temperature', 'Curing Time', 'Reaction Time', 'Operation Pressure',
        'Filtration Area', 'OSM Concentration'
    ]
    raw_all = pd.concat([train_data, test_data], axis=0)
    raw_medians = train_data[NUMERIC_FEATURES].median()
    subsample_size = min(200, len(x_all))
    X_sample = x_all.sample(subsample_size, random_state=10).reset_index(drop=True)
    X_sample.columns = X_sample.columns.astype(str)

    for feature_to_plot in features_to_plot:
        if feature_to_plot not in NUMERIC_FEATURES: continue
        print(f"  - 正在为 '{feature_to_plot}' 生成图表...")
        raw_min, raw_max = raw_all[feature_to_plot].min(), raw_all[feature_to_plot].max()
        grid_points = np.linspace(raw_min, raw_max, 50)
        grid_df = pd.DataFrame(np.repeat(raw_medians.values[np.newaxis, :], 50, axis=0), columns=NUMERIC_FEATURES)
        grid_df[feature_to_plot] = grid_points
        scaled_grid_array = scaler.transform(grid_df)
        scaled_grid_vals = scaled_grid_array[:, NUMERIC_FEATURES.index(feature_to_plot)]
        ice_curves = np.zeros((len(X_sample), 50))
        for i, val in enumerate(scaled_grid_vals):
            X_modified = X_sample.copy()
            X_modified[feature_to_plot] = val
            ice_curves[:, i] = model.predict(X_modified)
        ice_centered = ice_curves - ice_curves.mean(axis=1, keepdims=True)
        pdp_centered = ice_centered.mean(axis=0)

        fig, ax = plt.subplots()
        for curve in ice_centered: ax.plot(grid_points, curve, color='lightblue', alpha=0.3)
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

    # 模型筛选过程耗时较长，建议在第一次运行后注释掉
    # run_model_selection()

    best_params = run_hyperparameter_optimization()
    final_model, final_encoder, final_scaler = train_and_evaluate_final_model(best_params)
    run_model_interpretation(final_model, final_encoder, final_scaler)

    print("所有流程执行完毕")