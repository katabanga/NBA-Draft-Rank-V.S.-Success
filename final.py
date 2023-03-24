import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from scipy.stats import spearmanr
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
    df = df.dropna(subset=['WS/48', 'VORP', 'PTS'])
    df_lists.append(df)

def draft_analzyer():
    plt.figure(1)
    # Concatenate all the dataframes into one
    df = pd.concat(df_lists, ignore_index=True)

    model_1 = LinearRegression()
    model_1.fit(df['Pk'].values.reshape(-1, 1), df['VORP'])
    y_pred_1 = model_1.predict(df['Pk'].values.reshape(-1, 1))
    plt.plot(df['Pk'], y_pred_1, color='red')

    # X = df['Pk'].values.reshape(-1, 1)
    # X = sm.add_constant(X)
    # Y = df['VORP']
    # model = sm.OLS(Y, X).fit()
    # print(model.summary())
    # plt.plot(X, model.predict(X), color='blue')
    
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

    plt.figure(3)
    model_3 = LinearRegression()
    model_3.fit(df['Pk'].values.reshape(-1, 1), df['PTS'])
    y_pred_3 = model_3.predict(df['Pk'].values.reshape(-1, 1))
    plt.plot(df['Pk'], y_pred_3, color='red')

    plt.scatter(df['Pk'], df['PTS'], alpha=0.5)
    plt.xlabel("Pk")
    plt.ylabel("Points per game")
    plt.title("PTS vs. Pk")
    plt.savefig("PTS.png")

    # Group by pick number and calculate mean WS/48 and VORP
    grouped = df.groupby('Pk').agg({'PTS.1': 'mean', 'PTS': 'mean','AST.1': 'mean', 'WS/48': 'mean', 'VORP': 'mean'}).reset_index()

    # calculate the correlation matrix between draft pick and performance metrics
    correlation_matrix = grouped[['Pk', 'PTS', 'WS/48', 'VORP']].corr()

    # print the correlation matrix
    print(correlation_matrix)

    # e.g. Null Hypothesis: There is no significant difference in the average WS/48 and VORP 
    # between players selected in the top 5 picks (Pk <= 5) and players selected outside 
    # the top 10 picks (Pk > 5 and Pk <= 10). If null hypothesis is greater than 0.05,
    # then we can not reject the null hypothesis.

    for i in range(0, 11):
    # Split the data into two groups based on the Pk value
        top_picks = grouped[(grouped['Pk'] > i * 5) & (grouped['Pk'] <= (i + 1) * 5)]
        other_picks = grouped[(grouped['Pk'] > (i+1)*5) & (grouped['Pk'] <= (i+2)*5)]

        print("ttest group  1 Pk:", i*5, "-", (i+1)*5, "vs" , "group 2, pk: ", (i+1)*5, "-", (i+2)*5)

        # print(top_picks['WS/48'])
        # Perform a two-sample t-test on the WS/48 values
        ws_ttest = ttest_ind(top_picks['WS/48'], other_picks['WS/48'])

        # Print the p-value
        print('WS/48 p-value:', ws_ttest.pvalue)

        # Perform a two-sample t-test on the VORP values
        vorp_ttest = ttest_ind(top_picks['VORP'], other_picks['VORP'])

        # Print the p-value
        print('VORP p-value:', vorp_ttest.pvalue)

        # Perform a two-sample t-test on the PTS values
        pts_ttest = ttest_ind(top_picks['PTS'], other_picks['PTS'])

        # Print the p-value
        print('PTS p-value:', pts_ttest.pvalue)



draft_file_reader(file_list)  
draft_analzyer()


    