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

    # model_1 = LinearRegression()
    # model_1.fit(df['Pk'].values.reshape(-1, 1), df['VORP'])
    # y_pred_1 = model_1.predict(df['Pk'].values.reshape(-1, 1))
    # plt.plot(df['Pk'], y_pred_1, color='red')

    model_ws = sm.formula.ols('VORP ~ Pk', data=df).fit()
    model_vorp = sm.formula.ols('WS/48 ~ Pk', data=df).fit()
    print(model_ws.summary())
    print(model_vorp.summary())
    print('WS/48 p-value:', model_ws.pvalues['Pk'])
    print('VORP p-value:', model_vorp.pvalues['Pk'])
    
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

    # print(grouped)

def season_file_reader(season_lists):
    for file in season_lists:
        df = pd.read_csv(file)
        clean_season_data(df)

def clean_season_data(df):
    # check if the player occurs in the df multiple times, take the first occurence
    df = df.drop_duplicates(subset=['Player'], keep='first')
    df_season_lists.append(df)

def form_player_growth():
    # read the draft dataset
    draft_df = pd.concat(df_lists, ignore_index=True)

    # read the performance dataset
    performance_df = pd.concat(df_season_lists, ignore_index=True)

    # merge the datasets based on player names
    merged_df = pd.merge(draft_df, performance_df, on='Player', how='outer')
    print(draft_df)
    player_dict = {}

    for index, row in merged_df.iterrows():
        # get the player name
        player_name = row['Player']
        
        # check if the player is in the draft
        if player_name in draft_df['Player'].values:
            # create a new entry in the dictionary for the player
            if player_name not in player_dict:
                player_dict[player_name] = []
            
            # add the player's performance for the season
            # print( player_dict[player_name])
            if not math.isnan(row['PTS_x']):
                continue
            else:
                player_dict[player_name].append(row['PTS_y'])
            # player_dict[player_name]['AST'] += row['AST']
            # player_dict[player_name]['TRB'] += row['TRB']
    return player_dict



draft_file_reader(file_list)  
draft_analzyer()

season_file_reader(season_lists)
print(form_player_growth())
   


    