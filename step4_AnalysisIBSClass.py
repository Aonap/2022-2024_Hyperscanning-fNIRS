# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:09:12 2024

@author: Administrator
"""
import os
import pickle as pkl
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
def Savepkl(file, filename, data):
    # save file as .pickle. file:file's name
    filesave = file + filename
    if os.path.exists(filesave):
        print(filesave)
    else:
        file = open(filesave, 'wb')
        pkl.dump(data, file)
        file.close()
def Loadpkl(file, filename):
    # load file.pickle. file:file's name
    file = file + filename
    if os.path.exists(file):
        with open(file, 'rb') as file:
            data = pkl.load(file)
            return data
    else:
        print(file)     
def Savepkl(file, filename, data):
    # save file as .pickle. file:file's name
    filesave = file + filename
    if os.path.exists(filesave):
        print(filesave)
    else:
        file = open(filesave, 'wb')
        pkl.dump(data, file)
        file.close()
def Loadpkl(file, filename):
    # load file.pickle. file:file's name
    file = file + filename
    if os.path.exists(file):
        with open(file, 'rb') as file:
            data = pkl.load(file)
            return data
    else:
        print(file)
def extract_lower_triangle(data_dict):
    X = []
    y = []
    for key, matrix in data_dict.items():
        # 只提取下三角部分
        data = matrix.flatten()
        X.append(data)
        # 提取标签
        if "block1" in key and key[10]=='2':
            y.append(0)
        elif "block1" in key and key[10]=='3':
            y.append(1)
        elif "block2" in key and key[10]=='2':
            y.append(2)
        elif "block2" in key and key[10]=='3':
            y.append(3)

    return np.array(X), np.array(y)     
wd = r'F:\Chen\Friend2' #E
os.chdir(wd)
result_save = '3Result\\'
file_pic = '3result\\plot\\'
ch_name = pd.read_excel(result_save+'ch_names.xlsx')
ch_name =[i for i in ch_name[0]]
dt_sys = pd.read_excel('2code\\'+'10-20MNI.xlsx')
Fc_ccor = Loadpkl(result_save,'/step3_FC-ccor.pkl')
Fc_ccor_Roi = Loadpkl(result_save,'/step4_FCRoi.pkl')
Fc_coh = Loadpkl(result_save,'/step3_FC-coh.pkl')

Fc = Fc_ccor.copy()
f_names = [k for k in Fc.keys()]
Fc = {k:Fc[k].copy()[0:40:2, 40:80:2] for k in f_names}
Cor = {k:Fc[k].copy() for k in f_names if k[10]=='2'}
# 提取下三角部分并构建数据集
X, Y = extract_lower_triangle(Cor)
# ROI
Fc = Fc_ccor_Roi.copy()
f_names = [k for k in Fc_ccor.keys() if k[10]=='2']
X = pd.DataFrame({r:[Fc[r][k] for k in f_names] for r in Fc.keys()})
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # 对每列单独进行 MinMax 归一化
Y = np.array([k[5] for k in f_names])
# 留一交叉验证
loo = LeaveOneOut()
# 使用 SMOTE 增强样本
smote = SMOTE(random_state=42)
# 将数据分为训练集和测试集（为保证公平性，SMOTE 仅在训练集上使用）
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=42)
# 增强训练集
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
# 定义线性 SVM 分类器
classifier = SVC(kernel='linear', C=1,probability=True)
# 使用增强后的数据进行 SFFS 特征选择
sffs = SFS(
    classifier,
    k_features='best',  # 自动选择最优特征数量
    forward=True,
    floating=True,
    scoring='accuracy',
    cv=loo,
    n_jobs=-1)
# 在增强后的数据上进行特征选择
sffs = sffs.fit(X_train_smote, Y_train_smote)
# 获取最优特征子集
optimal_features = sffs.k_feature_idx_
optimal_feature_name = [X_normalized.columns[i] for i in optimal_features]
print(f"Optimal feature subset after augmentation: {optimal_features}")
# 提取增强数据中的最优特征
X_train_selected = X_train_smote.to_numpy()[:, optimal_features]
X_test_selected = X_test.to_numpy()[:, optimal_features]
# 网格搜索优化 SVM 参数
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_selected, Y_train_smote)
# 打印最佳参数
print("Best parameters:", grid_search.best_params_)
# 使用 LOOCV 在增强后的数据上进行评估
y_true, y_pred = [], []
for train_idx, test_idx in loo.split(X_train_selected):
    X_train_loo, X_test_loo = X_train_selected[train_idx], X_train_selected[test_idx]
    Y_train_loo, Y_test_loo = Y_train_smote[train_idx], Y_train_smote[test_idx]
    model = SVC(kernel=grid_search.best_params_['kernel'], 
                C=grid_search.best_params_['C'], 
                gamma=grid_search.best_params_['gamma'],probability=True)
    model.fit(X_train_loo, Y_train_loo)
    y_pred.append(model.predict(X_test_loo)[0])
    y_true.append(Y_test_loo[0])
# 计算增强数据下的准确率
final_accuracy = accuracy_score(Y_train_smote, y_pred)
print(f"Final LOOCV accuracy with augmentation: {final_accuracy:.2f}")
# 计算精确率、召回率和F1分数
precision = precision_score(y_true, y_pred, average='binary', pos_label='2')  # 假设 1 为正类
recall = recall_score(y_true, y_pred, average='binary', pos_label='2')  # 假设 1 为正类
f1 = f1_score(y_true, y_pred, average='binary', pos_label='2')  # 假设 1 为正类
# 输出相关指标
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
# 计算ROC曲线和AUC
y_true_numeric = [0 if label == '1' else 1 for label in y_true]
y_pred_numeric = [0 if label == '1' else 1 for label in y_pred]
# 计算 ROC AUC
roc_auc = roc_auc_score(y_true_numeric, y_pred_numeric)
fpr, tpr, thresholds = roc_curve(y_true_numeric, y_pred_numeric)
# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 随机分类器的参考线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
# 输出AUC
print(f"ROC AUC: {roc_auc:.2f}")
# 输出混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# 输出详细的分类报告（包括每个类别的指标）
print("Classification Report:")
print(classification_report(y_true, y_pred))
# %% from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 定义十折交叉验证 (10-Fold CV)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 存储每折的评估指标
accuracies, precisions, recalls, f1_scores = [], [], [], []
for train_idx, test_idx in kf.split(X_train_selected, Y_train_smote):
    # 获取当前折的训练集和测试集
    X_train_kf, X_test_kf = X_train_selected[train_idx], X_train_selected[test_idx]
    Y_train_kf, Y_test_kf = Y_train_smote[train_idx], Y_train_smote[test_idx]
    # 使用最佳参数训练 SVM
    model = SVC(kernel=grid_search.best_params_['kernel'], 
                C=grid_search.best_params_['C'], 
                gamma=grid_search.best_params_['gamma'], 
                probability=True)
    model.fit(X_train_kf, Y_train_kf)
    # 预测当前折的测试集
    Y_pred_kf = model.predict(X_test_kf)
    # 计算评估指标
    accuracies.append(accuracy_score(Y_test_kf, Y_pred_kf))
    precisions.append(precision_score(Y_test_kf, Y_pred_kf, average='weighted'))
    recalls.append(recall_score(Y_test_kf, Y_pred_kf, average='weighted'))
    f1_scores.append(f1_score(Y_test_kf, Y_pred_kf, average='weighted'))

# 打印十折交叉验证结果
print("10-Fold Cross-Validation Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
# %%
# 12. SHAP 模型解释
# 使用 SHAP 计算特征的 SHAP 值
explainer = shap.KernelExplainer(model.predict_proba, X_train_selected)
shap_values = explainer.shap_values(X_train_selected)

# 13. 查看每个特征的 SHAP 值
shap_df_class0 = pd.DataFrame(shap_values[0], columns=optimal_feature_name)  # 对应类别0
shap_df_class1 = pd.DataFrame(shap_values[1], columns=optimal_feature_name)  # 对应类别1

print("SHAP values for Class 0:")
print(shap_df_class0.mean(axis=0))  # 每个特征在类别 0 上的平均影响

print("SHAP values for Class 1:")
print(shap_df_class1.mean(axis=0))  # 每个特征在类别 1 上的平均影响

# 14. 绘制 SHAP summary plot
shap.summary_plot(shap_values, X_train_selected)

# %% 使用 PCA 后的数据进行 LOOCV 分析
pca = PCA(n_components=0.98)  # 保留 95% 的累计方差
X_pca = pca.fit_transform(X_normalized)
# 查看累计方差解释比例
explained_variance = pca.explained_variance_ratio_
print(f"累计解释方差比例: {explained_variance.cumsum()}")

loo = LeaveOneOut()
y_true, y_pred = [], []
Y = np.array(Y)
for train_idx, test_idx in loo.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    # SVM 模型
    model = SVC(kernel='linear', C=10)  # 使用线性核
    model.fit(X_train, Y_train)
    y_pred.append(model.predict(X_test)[0])
    y_true.append(Y_test[0])

# 计算准确率
final_accuracy = accuracy_score(y_true, y_pred)
print(f"使用 PCA 后的 LOOCV 准确率: {final_accuracy:.2f}")
# %% network
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 假设你的相关性字典是这样的
# cor_dict = { 'subject1_condition1': matrix1, 'subject1_condition2': matrix2, ... }

# 特征提取函数
def extract_graph_features(connectivity_matrix):
    G = nx.from_numpy_array(connectivity_matrix)
    features = {
        'degree': np.mean(np.array(list(dict(G.degree()).values()))),
        'clustering': np.mean(np.array(list(nx.clustering(G).values()))),
        'betweenness': np.mean(np.array(list(nx.betweenness_centrality(G).values()))),
        'density': nx.density(G),
    }
    return features

# 初始化存储特征的列表
features_condition1 = []
features_condition2 = []
cor_dict = Cor.copy()
# 遍历字典提取特征
for key, matrix in cor_dict.items():
    features = extract_graph_features(matrix)
    if 'block1' in key:
        features_condition1.append(features)
    elif 'block2' in key:
        features_condition2.append(features)

# 将特征列表转换为DataFrame
features_df_condition1 = pd.DataFrame(features_condition1)
features_df_condition2 = pd.DataFrame(features_condition2)

# 计算平均特征
mean_features_condition1 = features_df_condition1.mean(axis=0)
mean_features_condition2 = features_df_condition2.mean(axis=0)

# 创建一个包含平均特征的字典
mean_features = {
    'mean_degree_condition1': mean_features_condition1['degree'],
    'mean_clustering_condition1': mean_features_condition1['clustering'],
    'mean_betweenness_condition1': mean_features_condition1['betweenness'],
    'mean_degree_condition2': mean_features_condition2['degree'],
    'mean_clustering_condition2': mean_features_condition2['clustering'],
    'mean_betweenness_condition2': mean_features_condition2['betweenness'],
}

# 准备标签
labels = [0] * len(features_df_condition1) + [1] * len(features_df_condition2)

# 将所有平均特征整合为一个数据框
mean_feature_matrix = pd.DataFrame([mean_features])

# 划分训练集和测试集
data = pd.concat((features_df_condition1,features_df_condition2),axis=0)
data['condition'] = ['不牵手孤独'] * len(features_df_condition1) + ['牵手孤独'] * len(features_df_condition2)

# 对不同条件下的特征进行t检验
from scipy.stats import ttest_ind
results = {}
for column in data.columns[:-1]:  # 除去最后一列'condition'
    condition1_data = data[data['condition'] == '牵手孤独'][column]
    condition2_data = data[data['condition'] == '不牵手孤独'][column]
    
    t_stat, p_value = ttest_ind(condition1_data, condition2_data, equal_var=False)  # Welch's t-test
    results[column] = {'t_statistic': t_stat, 'p_value': p_value}

