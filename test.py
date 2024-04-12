import math

import pandas as pd
import glob

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#
# file_pattern = '*.csv'
# file_paths = [file for file in glob.glob(file_pattern) if "_info" not in file]
#
# all_data = []
# for file in file_paths:
#     df = pd.read_csv(file, low_memory=False)
#     all_data.append(df)
#
# combined_data = pd.concat(all_data, ignore_index=True)
combined_data = pd.read_csv('all_matches.csv', low_memory=False)

combined_data['B'] = 1

# Set 'B' to 0 for deliveries that are wides
# Assuming 'wides' column exists and is non-zero for wide balls
combined_data.loc[combined_data['wides'] > 0, 'B'] = 0

# Create the 'Out' column based on 'wicket_type'
# Assuming a player is out if 'wicket_type' is not null or not empty
# combined_data['Wicket'] = combined_data['wicket_type'].apply(lambda x: 0 if pd.isna(
#     x) or x == '' or x == 'retired hurt' or x == 'run out' or x == 'retired out' or x == 'obstructing the field' else 1)

combined_data['wides'].fillna(0, inplace=True)
combined_data['noballs'].fillna(0, inplace=True)

combined_data['RC'] = combined_data['wides'] + combined_data['noballs'] + combined_data['runs_off_bat']

# Extract the year from the 'start_date' column

combined_data['year'] = pd.to_datetime(combined_data['start_date'], format='mixed').dt.year

# Remove any potential duplicate rows
combined_data = combined_data.drop_duplicates()

# Define the years of interest
years_of_interest = list(range(2008, 2025))

combined_data['ball2'] = pd.to_numeric(combined_data['ball'], errors='coerce')

# Extract the over number using integer division
combined_data['over'] = combined_data['ball2'] // 1 + 1

over = list(range(1, 21))
combined_data2 = combined_data[combined_data['over'].isin(over)].copy()
combined_data = combined_data2.copy()

# Remove any potential duplicate rows
combined_data = combined_data.drop_duplicates()

# combined_data.to_csv('left.csv')
# combined_data = combined_data[combined_data['BatType'] == 'L']

# combined_data = combined_data[combined_data['bowler']=='Rashid Khan']

ball_bins = [0, 6, 11, 16, 20]
print(ball_bins)
ball_labels = ['1 to 6','7 to 11','12 to 16','17 to 20']
print(ball_labels)
combined_data['phase'] = pd.cut(combined_data['over'], bins=ball_bins, labels=ball_labels, include_lowest=True, right=True)
print(combined_data)

x =  combined_data[combined_data['year'].isin(years_of_interest)].copy()

def truemetrics(truevalues):
    truevalues['ER'] = truevalues['Runs Conceded'] / (truevalues['B'] / 6)
    truevalues['Expected ER'] = truevalues['Expected Runs Conceded'] / (truevalues['B'] / 6)
    truevalues['True ER'] = (truevalues['Expected Runs Conceded'] / (truevalues['B'] / 6) - truevalues['Runs Conceded'] / (truevalues['B'] / 6))
    truevalues['True Wickets'] = (truevalues['Wicket'] - truevalues['Expected Wickets'])
    truevalues['True W/ 4 overs'] = truevalues['True Wickets'] / (truevalues['B'] / 24)
    return truevalues


def calculate_entry_point_all_years(data):
    # Identifying the first instance each batter faces a delivery in each match

    first_appearance = data.drop_duplicates(subset=['match_id', 'innings', 'bowler'])
    first_appearance = first_appearance.copy()

    # Converting overs and deliveries into a total delivery count
    first_appearance.loc[:, 'total_deliveries'] = first_appearance['ball'].apply(
        lambda x: int(x) * 6 + int((x - int(x)) * 10))

    # Calculating the average entry point for each batter in total deliveries
    avg_entry_point_deliveries = first_appearance.groupby('bowler')['total_deliveries'].median().reset_index()

    # Converting the average entry point from total deliveries to overs and balls and rounding to 1 decimal place
    avg_entry_point_deliveries['average_over'] = avg_entry_point_deliveries['total_deliveries'].apply(
        lambda x: round((x // 6) + (x % 6) / 10, 1))

    return avg_entry_point_deliveries[['bowler', 'average_over']]


def calculate_first_appearance(data):
    # Identifying the first instance each batter faces a delivery in each match
    first_appearance = data.drop_duplicates(subset=['match_id', 'innings', 'bowler'])

    # Converting overs and deliveries into a total delivery count
    first_appearance.loc[:, 'total_deliveries'] = first_appearance['ball'].apply(
        lambda x: int(x) * 6 + int((x - int(x)) * 10))

    # Calculating the average entry point for each batter in total deliveries
    avg_entry_point_deliveries = first_appearance.groupby(['bowler', 'year', 'bowling_team'])[
        'total_deliveries'].median().reset_index()

    # Converting the average entry point from total deliveries to overs and balls
    avg_entry_point_deliveries['average_over'] = (
        avg_entry_point_deliveries['total_deliveries'].apply(lambda x: (x // 6) + (x % 6) / 10)).round(1)

    return avg_entry_point_deliveries[['bowler', 'average_over']]


def analyze_data_for_year2(data):
    # Filter the data for the specified year
    year_data = data.copy()

    # Calculate the first appearance of each batter in each match for the year
    first_appearance_data = calculate_first_appearance(year_data)

    # Calculate the average entry point for each batter

    # Assuming other analysis results are in a DataFrame named 'analysis_results'
    if 'analysis_results' in locals() or 'analysis_results' in globals():
        # Merge the average entry point data with other analysis results
        analysis_results = pd.merge(year_data, first_appearance_data, on=['bowler'],
                                    how='left')
    else:
        # Use average entry point data as the primary analysis result
        analysis_results = first_appearance_data

    return analysis_results


def analyze_data_for_year(year, data):
    # Filter data for the specific year
    combineddata2 = data[data['innings'] < 3].copy()
    data_year = combineddata2[combineddata2['year'] == year].copy()

    analysis_results = analyze_data_for_year2(data_year)
    analysis_results.columns = ['Player', 'Median Entry Point']

    # Ensure the 'ball' column is treated as a string
    # Filter out rows where a player was dismissed
    valid = ['retired hurt', 'run out' , 'retired out' , 'hit wicket' ,'obstructing the field']

    dismissed_data = data_year[data_year['player_dismissed'].notnull()]
    dismissed_data = dismissed_data[~dismissed_data['wicket_type'].isin(valid)]
    dismissed_data['Wicket'] = 1

    combineddata3 = pd.merge(data_year, dismissed_data[['match_id', 'innings', 'ball', 'bowler','Wicket']],
                             on=['match_id', 'innings', 'bowler','ball'], how='left')
    combineddata3['Wicket'].fillna(0, inplace=True)

    player_outs = combineddata3.groupby(['bowler', 'venue', 'over'])[['Wicket']].sum().reset_index()
    player_outs.columns = ['Player', 'Venue', 'Over', 'Wicket']

    over_outs = combineddata3.groupby(['venue', 'over'])[['Wicket']].sum().reset_index()
    over_outs.columns = ['Venue', 'Over', 'Wickets']

    temp = combineddata3.copy()
    temp['Runs'] = temp.groupby(['bowler', 'match_id', 'innings'], as_index=False)['RC'].cumsum()
    temp['Balls'] = temp.groupby(['bowler', 'match_id', 'innings'], as_index=False)['B'].cumsum()
    temp['Wickets'] = temp.groupby(['bowler', 'match_id', 'innings'], as_index=False)['Wicket'].cumsum()


    temp2 = temp[['match_id', 'innings', 'bowler', 'over','Runs', 'Balls']]
    temp2 = temp2.drop_duplicates()
    temp2.round(2).to_csv(f'ballbyballbowling{year}.csv')
    # Group by player and aggregate the runs scored
    player_runs = combineddata3.groupby(['bowler', 'venue', 'over'])[['RC', 'B']].sum().reset_index()
    # Rename the columns for clarity
    player_runs.columns = ['Player', 'Venue', 'Over', 'Runs Conceded', 'B']

    # Display the merged DataFrame
    over_runs = combineddata3.groupby(['venue', 'over'])[['RC', 'B']].sum().reset_index()
    over_runs.columns = ['Venue', 'Over', 'Runs', 'Balls']

    combined_df = pd.merge(player_runs, player_outs, on=['Player', 'Venue', 'Over'], how='left')

    # Merge the two DataFrames on the 'Player' column
    combined_df2 = pd.merge(over_runs, over_outs, on=['Venue', 'Over'], how='left')

    # Calculate BER and OPB for each ball
    combined_df2['BER'] = combined_df2['Runs'] / combined_df2['Balls']
    combined_df2['OPB'] = combined_df2['Wickets'] / combined_df2['Balls']

    # Merge the grouped data with the original data
    merged_data = pd.merge(combined_df, combined_df2, on=['Venue', 'Over'], how='left').reset_index()
    print(merged_data.columns)

    # Calculate Expected RC and Expected Wickets for each row
    merged_data['Expected Runs Conceded'] = merged_data['B'] * merged_data['BER']
    merged_data['Expected Wickets'] = merged_data['B'] * merged_data['OPB']

    players_years = combineddata3[['bowler', 'bowling_team', 'year']].drop_duplicates()
    players_years.columns = ['Player', 'Team', 'Year']

    # Group by bowler and sum the columns for final results
    truevalues = merged_data.groupby(['Player'])[
        ['B', 'Runs Conceded', 'Wicket', 'Expected Runs Conceded', 'Expected Wickets']].sum().reset_index()
    ball_bins = [0, 6, 11, 16, 20]
    ball_labels = ['1 to 6','7 to 11','12 to 16','17 to 20']
    merged_data['phase'] = pd.cut(merged_data['Over'], bins=ball_bins, labels=ball_labels, include_lowest=True, right=True)

    truevalues2 = merged_data.groupby(['Player','phase'])[
        ['B', 'Runs Conceded', 'Wicket', 'Expected Runs Conceded', 'Expected Wickets']].sum().reset_index()
    final_results = truemetrics(truevalues)
    final_results3 = pd.merge(analysis_results, final_results, on='Player', how='left')
    final_results4 = pd.merge(players_years, final_results3, on='Player', how='left')
    return final_results4.round(2),truevalues2


# Remove any potential duplicate rows
combined_data = combined_data.drop_duplicates()

all_data = []
all_data2 = []
# Analyze data and save results for each year
for year in years_of_interest:
    if year in combined_data['year'].unique():
        results,results2 = analyze_data_for_year(year, combined_data)
        all_data2.append(results2)
        results2.round(2).to_csv(f'overbyoverbowling{year}.csv')
        output_file_path = f'Bowling_{year}.csv'  # Adjust the path as needed
        results = results.sort_values(by=['Wicket'],ascending=False)
        all_data.append(results)
        results.to_csv(output_file_path)
        print(f"Data for year {year} saved to {output_file_path}")
    else:
        print(f"No data found for year {year}")




# Combine all data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)
combined_data12 = pd.concat(all_data2, ignore_index=True)

combined_data12.round(2).to_csv('overbyoverbowling.csv')

truevalues2 = combined_data12.groupby(['Player','phase'])[
    ['B', 'Runs Conceded', 'Wicket', 'Expected Runs Conceded', 'Expected Wickets']].sum().reset_index()

# Creating a pivot table with Balls as the index and players as columns
pivot_table = truevalues2.pivot_table(values='B', index='Player', columns='phase')

pivot_table.round(2).to_csv('overbyoverbowling2.csv')

output_file_path = 'bowler_data_summary2.csv'  # Adjust the path as needed


combined_data.to_csv(output_file_path)

# Drop duplicates to prevent double counting
combined_data.drop_duplicates(inplace=True)

truevalues = combined_data.groupby('Player')[['B', 'Runs Conceded', 'Wicket', 'Expected Runs Conceded', 'Expected Wickets']].sum()

final_results = truemetrics(truevalues)

most_frequent_team = combined_data.groupby('Player')['Team'].agg(lambda x: x.mode().iat[0]).reset_index()

final_results2 = pd.merge(most_frequent_team, final_results, on='Player', how='left')

analysis_results = calculate_entry_point_all_years(x)
analysis_results.columns = ['Player', 'Median Entry Point']

final_results3 = pd.merge(analysis_results, final_results2, on='Player', how='left')
output_file_path = 'Bowling_Overall.csv'  # Adjust the path as needed

final_results3 = final_results3.sort_values(by=['Wicket'],ascending=False)
final_results3.round(2).to_csv(output_file_path)
