# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:33:19 2025

@author: HUAWEI
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib

# ============== EDA + 可视化函数 ==============

def plot_correlation_matrix(df, outdir):
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    path = os.path.join(outdir, "correlation_matrix.png")
    plt.savefig(path)
    plt.close()

def plot_target_distribution(y, outdir):
    plt.figure(figsize=(6,4))
    sns.histplot(y, bins=30, kde=True)
    plt.title("Target Distribution")
    plt.xlabel("Purchase Amount (USD)")
    path = os.path.join(outdir, "target_distribution.png")
    plt.savefig(path)
    plt.close()

def plot_actual_vs_pred(y_true, y_pred, outdir):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    path = os.path.join(outdir, "actual_vs_predicted.png")
    plt.savefig(path)
    plt.close()

def plot_residuals(y_true, y_pred, outdir):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution")
    path = os.path.join(outdir, "residuals.png")
    plt.savefig(path)
    plt.close()

def plot_feature_importances(model, feature_names, outdir):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(8,6))
        plt.barh(range(len(indices)), importances[indices][::-1], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
        plt.title("Feature Importances")
        plt.tight_layout()
        path = os.path.join(outdir, "feature_importances.png")
        plt.savefig(path)
        plt.close()

# ============== 训练和调参函数 ==============

def train_and_evaluate(X, y, outdir):
    # 拆分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数值型 & 分类型特征
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 预处理
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 模型 + 超参数
    models = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 5, 10]
            }
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5, 7]
            }
        )
    }

    metrics = []
    best_models = {}

    for name, (model, params) in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        if params:
            search = GridSearchCV(pipe, param_grid=params, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"{name} 最优参数: {search.best_params_}")
        else:
            best_model = pipe.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics.append({"model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        best_models[name] = best_model

        # 保存模型
        model_path = os.path.join(outdir, f"{name}.joblib")
        joblib.dump(best_model, model_path)

        if name != "LinearRegression":  # 绘制特征重要性
            feature_names = numeric_features + list(best_model.named_steps["preprocessor"]
                                                   .transformers_[1][1]
                                                   .named_steps["onehot"]
                                                   .get_feature_names_out(categorical_features))
            plot_feature_importances(best_model.named_steps["model"], feature_names, outdir)

    # 保存评估结果
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(outdir, "model_metrics.csv"), index=False)
    print(metrics_df)

    # 绘制预测图（用最佳模型）
    best_model_name = metrics_df.sort_values("RMSE").iloc[0]["model"]
    best_model = best_models[best_model_name]
    y_pred = best_model.predict(X_test)
    plot_actual_vs_pred(y_test, y_pred, outdir)
    plot_residuals(y_test, y_pred, outdir)

    return metrics_df

# ============== 主函数 ==============

def run(data_path, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    df = pd.read_csv(data_path)

    # 选择重点特征 + 目标
    features = ["Age", "Review Rating", "Previous Purchases", "Frequency of Purchases"]
    target = "Purchase Amount (USD)"

    X = df[features]
    y = df[target]

    # EDA
    plot_correlation_matrix(df, outdir)
    plot_target_distribution(y, outdir)

    # 模型训练 + 调参
    metrics = train_and_evaluate(X, y, outdir)

    print("分析完成，结果已保存到：", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="shopping_trends.csv")
    parser.add_argument("--outdir", type=str, default="./results_tuned")
    args = parser.parse_args()

    run(args.data, args.outdir)
