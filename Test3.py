import pandas as pd
import glob
import streamlit as st


def truemetrics(truevalues):
    truevalues['Ave'] = truevalues['Runs Scored'] / truevalues['Out']
    truevalues['SR'] = (truevalues['Runs Scored'] / truevalues['BF'] * 100)

    truevalues['Expected Ave'] = truevalues['Expected Runs'] / truevalues['Expected Outs']
    truevalues['Expected SR'] = (truevalues['Expected Runs'] / truevalues['BF'] * 100)

    # Calculate 'True Ave' and 'True SR' for the final results
    truevalues['True Ave'] = (truevalues['Ave'] - truevalues['Expected Ave'])
    truevalues['True SR'] = (truevalues['SR'] - truevalues['Expected SR'])

    truevalues['Out Ratio'] = (truevalues['Expected Outs'] / truevalues['Out'])

    return truevalues


def truemetrics2(truevalues):
    ball_bins = [0, 6, 11, 16, 20]
    ball_labels = ['1 to 6','7 to 11','12 to 16','17 to 20']
    truevalues['phase'] = pd.cut(truevalues['Over'], bins=ball_bins, labels=ball_labels, include_lowest=True, right=True)
    truevalues2 = truevalues.groupby(['Player', 'phase'])[['Runs Scored', 'BF', 'Out']].sum().reset_index()
    truevalues3 = truevalues.groupby(['phase'])[['Runs Scored', 'BF', 'Out']].sum().reset_index()
    truevalues3.columns= ['phase','Mean Runs','Mean BF','Mean Outs']
    truevalues4 = pd.merge(truevalues2, truevalues3,on=['phase'], how='left')
    truevalues4['SR'] =  truevalues4['Runs Scored']/ truevalues4['BF'] * 100
    truevalues4['Mean SR'] =  truevalues4['Mean Runs']/ truevalues4['Mean BF'] * 100
    return truevalues4

def truemetrics3(truevalues):
    truevalues2 = truevalues.groupby(['Player', 'phase'])[['Runs Scored', 'BF', 'Out']].sum().reset_index()
    truevalues3 = truevalues.groupby(['phase'])[['Runs Scored', 'BF', 'Out']].sum().reset_index()
    truevalues3.columns= ['phase','Mean Runs','Mean BF','Mean Outs']
    truevalues4 = pd.merge(truevalues2, truevalues3,on=['phase'], how='left')
    truevalues4['SR'] =  truevalues4['Runs Scored']/ truevalues4['BF'] * 100
    truevalues4['Mean SR'] =  truevalues4['Mean Runs']/ truevalues4['Mean BF'] * 100
    return truevalues4

def calculate_entry_point_all_years(data):
    # Identifying the first instance each batter faces a delivery in each match

    first_appearance = data.drop_duplicates(subset=['match_id', 'innings', 'striker'])
    first_appearance = first_appearance.copy()

    # Converting overs and deliveries into a total delivery count
    first_appearance.loc[:, 'total_deliveries'] = first_appearance['ball'].apply(
        lambda x: int(x) * 6 + int((x - int(x)) * 10))
    # Calculating the average entry point for each batter in total deliveries
    avg_entry_point_deliveries = first_appearance.groupby('striker')['total_deliveries'].median().reset_index()

    # Converting the average entry point from total deliveries to overs and balls and rounding to 1 decimal place
    avg_entry_point_deliveries['average_over'] = avg_entry_point_deliveries['total_deliveries'].apply(
        lambda x: round((x // 6) + (x % 6) / 10, 1))

    return avg_entry_point_deliveries[['striker', 'average_over']], first_appearance


def calculate_first_appearance(data):
    # Identifying the first instance each batter faces a delivery in each match
    first_appearance = data.drop_duplicates(subset=['match_id', 'innings', 'striker'])
    # Converting overs and deliveries into a total delivery count
    first_appearance.loc[:, 'total_deliveries'] = first_appearance['ball'].apply(
        lambda x: int(x) * 6 + int((x - int(x)) * 10))

    # Calculating the average entry point for each batter in total deliveries
    avg_entry_point_deliveries = first_appearance.groupby(['striker', 'year', 'batting_team'])[
        'total_deliveries'].median().reset_index()

    # Converting the average entry point from total deliveries to overs and balls
    avg_entry_point_deliveries['average_over'] = (
        avg_entry_point_deliveries['total_deliveries'].apply(lambda x: (x // 6) + (x % 6) / 10)).round(1)

    return avg_entry_point_deliveries[['striker', 'average_over']]


def analyze_data_for_year2(data):
    # Filter the data for the specified year
    year_data = data.copy()

    # Calculate the first appearance of each batter in each match for the year
    first_appearance_data = calculate_first_appearance(year_data)

    # Calculate the average entry point for each batter

    # Assuming other analysis results are in a DataFrame named 'analysis_results'
    if 'analysis_results' in locals() or 'analysis_results' in globals():
        # Merge the average entry point data with other analysis results
        analysis_results = pd.merge(year_data, first_appearance_data, on=['striker'],
                                    how='left')
    else:
        # Use average entry point data as the primary analysis result
        analysis_results = first_appearance_data

    return analysis_results


def analyze_data_for_year3(year2, data2):
    combineddata2 = data2[data2['innings'] < 3].copy()
    combineddata = combineddata2[combineddata2['year'] == year2].copy()
    inns = combineddata.groupby(['striker', 'match_id'])[['runs_off_bat']].sum().reset_index()
    inns['I'] = 1
    inns2 = inns.groupby(['striker'])[['I']].sum().reset_index()
    inns2.columns = ['Player', 'I']
    inns3 = inns.copy()
    inns['CI'] = inns.groupby(['striker'],as_index=False)[['I']].cumsum()
    analysis_results = analyze_data_for_year2(combineddata)
    analysis_results.columns = ['Player', 'Median Entry Point']

    # Filter out rows where a player was dismissed
    dismissed_data = combineddata[combineddata['player_dismissed'].notnull()]
    dismissed_data = dismissed_data[dismissed_data['wicket_type'] != 'retired hurt']
    dismissed_data['Out'] = 1

    combineddata2 = pd.merge(combineddata, dismissed_data[['match_id', 'innings', 'player_dismissed', 'over', 'Out']],
                             on=['match_id', 'innings', 'player_dismissed', 'over'], how='left')
    combineddata2['Out'].fillna(0, inplace=True)
    combineddata = combineddata2.copy()

    player_outs = dismissed_data.groupby(['player_dismissed', 'venue', 'over'])[['Out']].sum().reset_index()
    player_outs.columns = ['Player', 'Venue', 'Over', 'Out']

    over_outs = dismissed_data.groupby(['venue', 'over'])[['Out']].sum().reset_index()
    over_outs.columns = ['Venue', 'Over', 'Outs']

    # Group by player and aggregate the runs scored
    player_runs = combineddata.groupby(['striker', 'venue', 'over'])[['runs_off_bat', 'B']].sum().reset_index()
    # Rename the columns for clarity
    player_runs.columns = ['Player', 'Venue', 'Over', 'Runs Scored', 'BF']

    # Display the merged DataFrame
    over_runs = combineddata.groupby(['venue', 'over'])[['runs_off_bat', 'B']].sum().reset_index()
    over_runs.columns = ['Venue', 'Over', 'Runs', 'B']
    # Merge the two DataFrames on the 'Player' column

    combined_df = pd.merge(player_runs, player_outs, on=['Player', 'Venue', 'Over'], how='left')
    # Merge the two DataFrames on the 'Player' column
    combined_df2 = pd.merge(over_runs, over_outs, on=['Venue', 'Over'], how='left')
    # Calculate BSR and OPB for each ball at each venue
    combined_df2['BSR'] = combined_df2['Runs'] / combined_df2['B']
    combined_df2['OPB'] = combined_df2['Outs'] / combined_df2['B']

    combined_df3 = pd.merge(combined_df, combined_df2, on=['Venue', 'Over'], how='left')
    combined_df3['Expected Runs'] = combined_df3['BF'] * combined_df3['BSR']
    combined_df3['Expected Outs'] = combined_df3['BF'] * combined_df3['OPB']

    truevalues = combined_df3.groupby(['Player'])[
        ['Runs Scored', 'BF', 'Out', 'Expected Runs', 'Expected Outs']].sum()

    final_results = truemetrics(truevalues)

    players_years = combineddata[['striker', 'batting_team', 'year']].drop_duplicates()
    players_years.columns = ['Player', 'Team', 'Year']
    final_results2 = pd.merge(inns2, final_results, on='Player', how='left')
    final_results3 = pd.merge(players_years, final_results2, on='Player', how='left')
    final_results4 = pd.merge(final_results3, analysis_results, on='Player', how='left')
    print(combined_df3.columns)
    truevalues = truemetrics2(combined_df3)
    return final_results4.round(2)

def analyze_data_for_year4(year2, data2):
    combineddata2 = data2[data2['innings'] < 3].copy()
    combineddata = combineddata2[combineddata2['year'] == year2].copy()
    inns = combineddata.groupby(['striker', 'match_id','phase'])[['runs_off_bat']].sum().reset_index()
    inns['I'] = 1
    inns2 = inns.groupby(['striker','phase'])[['I']].sum().reset_index()
    inns2.columns = ['Player','phase', 'I']
    inns3 = inns.copy()
    inns['CI'] = inns.groupby(['striker'],as_index=False)[['I']].cumsum()

    # Filter out rows where a player was dismissed
    dismissed_data = combineddata[combineddata['player_dismissed'].notnull()]
    dismissed_data = dismissed_data[dismissed_data['wicket_type'] != 'retired hurt']
    dismissed_data['Out'] = 1

    combineddata2 = pd.merge(combineddata, dismissed_data[['match_id', 'innings', 'player_dismissed', 'over', 'Out']],
                             on=['match_id', 'innings', 'player_dismissed', 'over'], how='left')
    combineddata2['Out'].fillna(0, inplace=True)
    combineddata = combineddata2.copy()

    player_outs = dismissed_data.groupby(['player_dismissed', 'venue', 'over','phase'])[['Out']].sum().reset_index()
    player_outs.columns = ['Player', 'Venue', 'Over','phase', 'Out']

    over_outs = dismissed_data.groupby(['venue', 'over','phase'])[['Out']].sum().reset_index()
    over_outs.columns = ['Venue', 'Over','phase', 'Outs']

    # Group by player and aggregate the runs scored
    player_runs = combineddata.groupby(['striker', 'venue', 'over','phase'])[['runs_off_bat', 'B']].sum().reset_index()
    # Rename the columns for clarity
    player_runs.columns = ['Player', 'Venue', 'Over','phase', 'Runs Scored', 'BF']

    # Display the merged DataFrame
    over_runs = combineddata.groupby(['venue', 'over','phase'])[['runs_off_bat', 'B']].sum().reset_index()
    over_runs.columns = ['Venue', 'Over','phase', 'Runs', 'B']
    # Merge the two DataFrames on the 'Player' column

    combined_df = pd.merge(player_runs, player_outs, on=['Player', 'Venue', 'Over','phase'], how='left')
    # Merge the two DataFrames on the 'Player' column
    combined_df2 = pd.merge(over_runs, over_outs, on=['Venue', 'Over','phase'], how='left')
    # Calculate BSR and OPB for each ball at each venue
    combined_df2['BSR'] = combined_df2['Runs'] / combined_df2['B']
    combined_df2['OPB'] = combined_df2['Outs'] / combined_df2['B']

    combined_df3 = pd.merge(combined_df, combined_df2, on=['Venue', 'Over','phase'], how='left')
    combined_df3['Expected Runs'] = combined_df3['BF'] * combined_df3['BSR']
    combined_df3['Expected Outs'] = combined_df3['BF'] * combined_df3['OPB']

    truevalues = combined_df3.groupby(['Player','phase'])[
        ['Runs Scored', 'BF', 'Out', 'Expected Runs', 'Expected Outs']].sum()

    final_results = truemetrics(truevalues)

    players_years = combineddata[['striker', 'batting_team', 'year','phase']].drop_duplicates()
    players_years.columns = ['Player', 'Team', 'Year','phase']
    final_results2 = pd.merge(inns2, final_results, on=['Player','phase'], how='left')
    final_results3 = pd.merge(players_years, final_results2, on=['Player','phase'], how='left')
    return final_results3.round(2)

all_data = []
all_data2 = []
all_data3 = []

# Load the data
@st.cache
def load_data(filename):
    data = pd.read_csv(filename, low_memory=False)
    return data

# The main app function
def main():
    st.title('IPL Batting True Values')
    
    # Load your data
    data =  pd.read_csv('all_matches.csv', low_memory=False)
    data['B'] = 1

    # Set 'B' to 0 for deliveries that are wides
    # Assuming 'wides' column exists and is non-zero for wide balls
    data.loc[data['wides'] > 0, 'B'] = 0
    
    data['wides'].fillna(0, inplace=True)
    data['noballs'].fillna(0, inplace=True)
    
    data['RC'] = data['wides'] + data['noballs'] + data['runs_off_bat']
    
    # Extract the year from the 'start_date' column
    
    data['year'] = pd.to_datetime(data['start_date'], format='mixed').dt.year
    
    # Remove any potential duplicate rows
    data = data.drop_duplicates()

    
    data['ball2'] = pd.to_numeric(data['ball'], errors='coerce')
    data['over'] = data['ball2'] // 1 + 1
    
    # Selectors for user input
    start_year, end_year = st.slider('Select Years Range:', min_value=2008, max_value=2024, value=(2008, 2024))
    start_over, end_over = st.slider('Select Overs Range:', min_value=1, max_value=20, value=(1, 20))
    filtered_data = data[(data['over'] >= start_over) & (data['over'] <= end_over)]
    filtered_data2 = filtered_data[(filtered_data['year'] >= start_year) & (filtered_data['year'] <= end_year)]
    x = filtered_data2
    # A button to trigger the analysis
    if st.button('Analyze'):
        # Call a hypothetical function to analyze data
        all_data = []

# Analyze data and save results for each year
        for year in filtered_data2['year'].unique():
            results = analyze_data_for_year3(year, filtered_data2)
            all_data.append(results)

        combined_data = pd.concat(all_data, ignore_index=True)
        most_frequent_team = combined_data.groupby('Player')['Team'].agg(lambda x: x.mode().iat[0]).reset_index()

        truevalues = combined_data.groupby(['Player'])[
            ['I', 'Runs Scored', 'BF', 'Out', 'Expected Runs', 'Expected Outs']].sum()
        final_results = truemetrics(truevalues)

        final_results2 = pd.merge(most_frequent_team, final_results, on='Player', how='left')

        final_results3, f = calculate_entry_point_all_years(x)
        final_results3.columns = ['Player', 'Median Entry Point']

        final_results4 = pd.merge(final_results3, final_results2, on='Player', how='left').reset_index()
        final_results4 = final_results4.sort_values(by=['Runs Scored'], ascending=False)

        # Display the results
        st.dataframe(final_results4[['Player', 'Median Entry Point','Team','I', 'Runs Scored', 'BF', 'Out','Ave','SR','Expected Ave','Expected SR','True Ave','True SR']].round(2))

# Run the main function
if __name__ == '__main__':
    main()
