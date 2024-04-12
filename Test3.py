import pandas as pd
import glob

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

# combined_data.to_csv('all_matches.csv')

def battingpositions(combineddata):
    # Initialize a column for the batting position in the original DataFrame
    grouped_data = combineddata.groupby(['match_id', 'batting_team'])
    combineddata['batting_position'] = 0

    # Iterate over each group (unique match_id and batting_team)
    for (match_id, team), group in grouped_data:
        # Initialize a dictionary to track positions
        position_counter = {}
        # Get the first row of the group to identify initial striker and non-striker
        first_row = group.iloc[0]
        initial_striker = first_row['striker']
        initial_non_striker = first_row['non_striker']

        # Assign position 1 to the initial striker and 2 to the initial non-striker
        position_counter[initial_striker] = 1
        position_counter[initial_non_striker] = 2

        # Iterate through each row of the group to assign positions
        for index, row in group.iterrows():
            striker = row['striker']
            if striker not in position_counter:
                # Assign the next position number to new strikers
                position_counter[striker] = len(position_counter) + 1
            # Update the DataFrame with the batting position
            combineddata.at[index, 'batting_position'] = position_counter[striker]
    return combineddata


combined_data2 = battingpositions(combined_data)
combined_data = combined_data2.copy()

combined_data['B'] = 1

# Set 'B' to 0 for deliveries that are wides
# Assuming 'wides' column exists and is non-zero for wide balls
combined_data.loc[combined_data['wides'] > 0, 'B'] = 0

combined_data['wides'].fillna(0, inplace=True)
combined_data['noballs'].fillna(0, inplace=True)

combined_data['RC'] = combined_data['wides'] + combined_data['noballs'] + combined_data['runs_off_bat']

# Extract the year from the 'start_date' column

combined_data['year'] = pd.to_datetime(combined_data['start_date'], format='mixed').dt.year

# Remove any potential duplicate rows
combined_data = combined_data.drop_duplicates()

# Define the years of interest
years_of_interest = list(range(2008, 2025))

pos = list(range(1, 12))
combined_data2 = combined_data[combined_data['batting_position'].isin(pos)].copy()
combined_data = combined_data2.copy()

combined_data['ball2'] = pd.to_numeric(combined_data['ball'], errors='coerce')
combined_data['over'] = combined_data['ball2'] // 1 + 1

# Remove any potential duplicate rows
combined_data = combined_data.drop_duplicates()

over = list(range(1, 21))
combined_data2 = combined_data[combined_data['over'].isin(over)].copy()
combined_data = combined_data2.copy()
# Remove any potential duplicate rows
combined_data = combined_data.drop_duplicates()

# combined_data.to_csv('left.csv')
# combined_data[combined_data['BatType'] == 'R'].to_csv('right.csv')

# combined_data = combined_data[combined_data['bowler']=='Rashid Khan']

ball_bins = [0, 6, 11, 16, 20]
ball_labels = ['1 to 6','7 to 11','12 to 16','17 to 20']
combined_data['phase'] = pd.cut(combined_data['over'], bins=ball_bins, labels=ball_labels, include_lowest=True, right=True)


x = combined_data[combined_data['year'].isin(years_of_interest)].copy()

#
#
# temp = combined_data[combined_data['striker'] == 'V Kohli']
# temp2 = temp['match_id'].unique().tolist()
# temp3 = combined_data[combined_data['striker'] == 'V Kohli']
# temp4 = temp3['match_id'].unique().tolist()
#
# combined_data2 = combined_data[combined_data['venue'] == 'Narendra Modi Stadium, Ahmedabad'].copy()
# combined_data = combined_data2.copy()



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

    temp = combineddata.copy()
    temp['Runs'] = temp.groupby(['striker', 'match_id', 'innings'], as_index=False)['runs_off_bat'].cumsum()
    temp['Balls'] = temp.groupby(['striker', 'match_id', 'innings'], as_index=False)['B'].cumsum()
    temp['Wickets'] = temp.groupby(['striker', 'match_id', 'innings'], as_index=False)['Out'].cumsum()

    temp.to_csv('asdkasda.csv')

    temp2 = temp[['match_id', 'innings', 'striker', 'over','Runs', 'Balls']]
    temp2 = temp2.drop_duplicates()
    temp2.round(2).to_csv(f'ballbyball{year2}.csv')
    temp2['count'] = 1
    temp2.loc[temp2['Balls'] == 0, 'count'] = 0
    temp3 = temp2.groupby(['striker', 'Balls'])[['Runs', 'count']].sum().reset_index()
    temp3['Ave Runs'] = temp3['Runs'] / temp3['count']
    temp3['SR'] = temp3['Ave Runs'] / temp3['Balls'] * 100
    temp3.round(2).to_csv(f'averagesr{year2}.csv')

    # Creating a pivot table with Balls as the index and players as columns
    pivot_table = temp3.pivot_table(values='Ave Runs', index='Balls', columns='striker')

    # Saving the pivot table to a CSV file
    pivot_table.round(2).to_csv(f'player_runs_by_balls{year2}.csv')

    # Create a pivot table with Balls as the index and strikers as columns with their average runs
    pivot_table = temp3.pivot_table(values='SR', index='Balls', columns='striker')

    # Saving the strike rates to a CSV file
    pivot_table.round(2).to_csv(f'strike_rate_by_balls{year2}.csv')

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
    return final_results4.round(2), truevalues

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

# Analyze data and save results for each year
for year in years_of_interest:
    if year in combined_data['year'].unique():
        results, results2 = analyze_data_for_year3(year, combined_data)
        output_file_path = f'Batting_{year}.csv'  # Adjust the path as needed
        results2.round(2).to_csv(f'overbyover{year}.csv')
        results = results.sort_values(by=['Runs Scored'], ascending=False)
        results.to_csv(output_file_path)
        all_data.append(results)
        all_data2.append(results2)
        print(f"Data for year {year} saved to {output_file_path}")
        results = analyze_data_for_year4(year,combined_data)
        output_file_path2 = f'Battingbyphase_{year}.csv'  # Adjust the path as needed
        results = results.sort_values(by=['Runs Scored'], ascending=False)
        results.to_csv(output_file_path2)
        all_data3.append(results)
    else:
        print(f"No data found for year {year}")
#
# # Read each file and store them in a list
# for file in file_paths:
#     df = pd.read_csv(file, low_memory=False)
#     all_data.append(df)
# Combine all data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)
combined_data2 = pd.concat(all_data2, ignore_index=True)
combined_data9 = pd.concat(all_data3, ignore_index=True)

output_file_path = 'batter_data_summary2.csv'  # Adjust the path as needed
combined_data.to_csv(output_file_path)

# Drop duplicates to prevent double counting
combined_data.drop_duplicates(inplace=True)

most_frequent_team = combined_data.groupby('Player')['Team'].agg(lambda x: x.mode().iat[0]).reset_index()

truevalues = combined_data.groupby(['Player'])[
    ['I', 'Runs Scored', 'BF', 'Out', 'Expected Runs', 'Expected Outs']].sum()
truevalues2 = combined_data9.groupby(['Player','phase'])[
    ['I', 'Runs Scored', 'BF', 'Out', 'Expected Runs', 'Expected Outs']].sum()

final_results = truemetrics(truevalues)

output_file_path = 'Batting_Overall.csv'  # Adjust the path as needed

final_results2 = pd.merge(most_frequent_team, final_results, on='Player', how='left')

final_results3, f = calculate_entry_point_all_years(x)
final_results3.columns = ['Player', 'Median Entry Point']

final_results4 = pd.merge(final_results3, final_results2, on='Player', how='left')
final_results4 = final_results4.sort_values(by=['Runs Scored'], ascending=False)
final_results4.round(2).to_csv(output_file_path)

truemetrics3(combined_data2).round(2).to_csv('overbyover.csv')

most_frequent_team = combined_data9.groupby(['Player','phase'])['Team'].agg(lambda x: x.mode().iat[0]).reset_index()
final_results = truemetrics(truevalues2)
final_results2 = pd.merge(most_frequent_team, final_results, on=['Player','phase'], how='left')
final_results2.round(2).to_csv('Battingbyphase.csv')

combined_data9.to_csv('Battingbyphase2.csv')
