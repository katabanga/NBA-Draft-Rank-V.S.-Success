import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
import statsmodels.api as sm

file_list = []
df_lists = []
season_lists = []
df_season_lists = []

for i in range(7,21):
  file_list.append("draft/"+str(i)+".csv")

for i in range(7,8):
    season_lists.append("season/"+str(i)+".csv")

def draft_file_reader(file_list):
    for file in file_list:
        df = pd.read_csv(file)
        clean_data_draft(df)

# remove all the rows that have NaN values in the WS/48 or VORP columns
def clean_data_draft(df):
    df = df.dropna(subset=['WS/48', 'VORP'])
    df_lists.append(df)


def draft_analzyer():
    plt.figure(1)
    # Concatenate all the dataframes into one
    df = pd.concat(df_lists, ignore_index=True)

    model_1 = LinearRegression()
    model_1.fit(df['Pk'].values.reshape(-1, 1), df['VORP'])
    y_pred_1 = model_1.predict(df['Pk'].values.reshape(-1, 1))
    plt.plot(df['Pk'], y_pred_1, color='red')

    # model_ws = sm.formula.ols('VORP ~ Pk', data=df).fit()
    # model_vorp = sm.formula.ols('WS/48 ~ Pk', data=df).fit()
    # print(model_ws.summary())
    # print(model_vorp.summary())
    # print('WS/48 p-value:', model_ws.pvalues['Pk'])
    # print('VORP p-value:', model_vorp.pvalues['Pk'])
    
    plt.scatter(df['Pk'], df['VORP'], alpha=0.5)
    plt.xlabel("Draft Pick")
    plt.ylabel("Value Over Replacement Player")
    plt.title("VORP vs. Pk")
    plt.savefig("VORP.png")

    plt.figure(2)
    model_2 = LinearRegression()
    model_2.fit(df['Pk'].values.reshape(-1, 1), df['WS/48'])
    y_pred_2 = model_2.predict(df['Pk'].values.reshape(-1, 1))
    plt.plot(df['Pk'], y_pred_2, color='red')

    plt.scatter(df['Pk'], df['WS/48'], alpha=0.5)
    plt.xlabel("Draft Pick")
    plt.ylabel("Win Shares per 48 Minutes")
    plt.title("WS vs. Pk")
    plt.savefig("WS48.png")

    # Group by pick number and calculate mean WS/48 and VORP
    grouped = df.groupby('Pk').agg({'PTS.1': 'mean', 'PTS': 'mean','AST.1': 'mean', 'WS/48': 'mean', 'VORP': 'mean'}).reset_index()

    
    # Split the data into two groups based on the Pk value
    top_picks = grouped[grouped['Pk'] <= 5]
    other_picks = grouped[(df['Pk'] > 5) & (df['Pk'] <= 10)]

    # Perform a two-sample t-test on the WS/48 values
    ws_ttest = ttest_ind(top_picks['WS/48'], other_picks['WS/48'])

    # Print the p-value
    print('WS/48 p-value:', ws_ttest.pvalue)

    # Perform a two-sample t-test on the VORP values
    vorp_ttest = ttest_ind(top_picks['VORP'], other_picks['VORP'])

    # Print the p-value
    print('VORP p-value:', vorp_ttest.pvalue)

    print(grouped)


draft_file_reader(file_list)  
draft_analzyer()


    