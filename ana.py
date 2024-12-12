# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:53:03 2024

@author: ZKHY
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm
wd = r'D:\Study\research\1three-pic\\'

os.chdir(wd)
dt_fnirs = pd.read_csv('selected_data.csv')
dt_hbo = dt_fnirs[dt_fnirs['channel']=='S8_D6 hbo']
dt_hbo.index = range(dt_hbo.shape[0])
ID_hbo = [dt_hbo['subj'][i] for i in range(dt_hbo.shape[0]) if dt_hbo['marker'][i]==1]
dt_Ques = pd.read_excel('反向计分近红外量表total.xlsx')
dt_Img = pd.read_excel('总图片评分.xlsx')
Img = pd.read_excel('female.xlsx')['Imge']
# t-test
# 量表 需要进行 t 检验的变量列
dt = dt_Ques.copy()
variables = [i for i in dt.columns]
# 要删除的变量
variables_to_remove = ['ID', 'group','age_bl','gender_body']
# 删除指定的变量
variables = [var for var in variables if var not in variables_to_remove]
# 进行 t 检验并存储结果
results = {}
for var in variables:
    group_A = dt[dt['group'] == 0][var]
    group_B = dt[dt['group'] == 1][var]
    # 删除空值
    group_A = group_A.dropna()
    group_B = group_B.dropna()
    # 独立样本 t 检验
    t_stat, p_value = ttest_ind(group_A, group_B)
    
    # 存储结果
    results[var] = {'t_stat': t_stat, 'p_value': p_value,
                    'mean0':group_A.mean(),'std0':group_A.std(),
                    'mean1':group_B.mean(),'std1':group_B.std()}
results_df = pd.DataFrame(results).transpose()
results_df.to_excel('Ques_t-test_Group.xlsx')
dt_cor = dt.corr()
dt_cor.to_excel('Ques_.xlsx')
# 图片评分
dt = dt_Img.copy()
dt = dt[dt['Order'].isin(Img)]
variables = [i for i in dt.columns]
# 要删除的变量
variables = ['Class','rg_aro_m', 'rg_ple_m','nrg_aro_m','nrg_ple_m','arom','plem']

# 进行 t 检验并存储结果
results = {}
for var in variables:
    group_A = dt[dt['Class'] == 1][var]
    group_B = dt[dt['Class'] == 2][var]
    # 删除空值
    group_A = group_A.dropna()
    group_B = group_B.dropna()
    # 独立样本 t 检验
    t_stat, p_value = ttest_ind(group_A, group_B)
    
    # 存储结果
    results[var] = {'t_stat': t_stat, 'p_value': p_value,
                    'mean0':group_A.mean(),'std0':group_A.std(),
                    'mean1':group_B.mean(),'std1':group_B.std()}
results_df = pd.DataFrame(results).transpose()
results_df.to_excel('Image-Select_t-test_Group.xlsx')
dt_cor = dt.corr()
# 回归分析
df = pd.DataFrame(dt)
df = df[df['ID'].isin(ID_hbo)]
df['ID'] = pd.Categorical(df['ID'], categories=ID_hbo, ordered=True)
df = df.sort_values('ID')
df.index = range(df.shape[0])

# 准备数据
k = 1 # mark 1,2,3 gay les mask
y_n = 'HBO_homo'
value =  dt_hbo[dt_hbo['marker']==k]
value.index = range(value.shape[0])
df[y_n] = value['value']
X = df[['group', 'sexual_orientation', 'gender_psy','at']]  # 独立变量
X = sm.add_constant(X)  # 添加常数项
y = df[y_n]  # 因变量
# 构建回归模型
model = sm.OLS(y, X).fit()
# 打印回归结果
print(model.summary())

# 计算 VIF 的函数
def calculate_vif(X):
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [sm.OLS(X[col], X.loc[:, X.columns != col]).fit().rsquared for col in X.columns]
    vif['VIF'] = 1 / (1 - vif['VIF'])
    return vif

# 假设 X 是你的自变量 DataFrame
vif_results = calculate_vif(X)
print(vif_results)

# 假设 df 是你的 DataFrame，'Y' 是因变量名
y = df[y_n]  # 因变量
X_columns = X.columns  # 自变量的列名
# 计算每个自变量与因变量之间的相关系数
from scipy.stats import pearsonr
# 假设 X 是你的自变量 DataFrame
X_columns = X.columns  # 自变量的列名
results = {}

# 计算每对自变量之间的相关系数
for i in range(len(X_columns)):
    for j in range(i + 1, len(X_columns)):
        col1 = X_columns[i]
        col2 = X_columns[j]
        corr_coef, p_value = pearsonr(X[col1], X[col2])
        results[(col1, col2)] = {'correlation_coefficient': corr_coef, 'p_value': p_value}

# 将结果转换为 DataFrame 以便于查看
correlation_results = pd.DataFrame(results).T
correlation_results.columns = ['correlation_coefficient', 'p_value']
print(correlation_results)

# 假设 df 是你的 DataFrame，'y_n' 是因变量的列名
y = df[y_n]  # 因变量
X_columns = X.columns  # 自变量的列名

# 计算因变量与每个自变量之间的相关系数
results_y = {}
for col in X_columns:
    corr_coef, p_value = pearsonr(df[col], y)
    results_y[col] = {'correlation_coefficient': corr_coef, 'p_value': p_value}

# 将结果转换为 DataFrame 以便于查看
correlation_results_y = pd.DataFrame(results_y).T
correlation_results_y.columns = ['correlation_coefficient', 'p_value']
print(correlation_results_y)
