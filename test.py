import math

import pandas as pd
import streamlit as st

# Extract the year from the 'start_date' column

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
    # combineddata2 = data[data['innings'] < 3].copy()
    data_year = data[data['year'] == year].copy()

    # analysis_results = analyze_data_for_year2(data_year)
    # analysis_results.columns = ['Player', 'Median Entry Point']

    # Ensure the 'ball' column is treated as a string
    # Filter out rows where a player was dismissed
    valid = ['retired hurt', 'run out' , 'retired out' , 'hit wicket' ,'obstructing the field']

    dismissed_data = data_year[data_year['player_dismissed'].notnull()]
    dismissed_data = dismissed_data[~dismissed_data['wicket_type'].isin(valid)]
    dismissed_data['Wicket'] = 1

    combineddata3 = pd.merge(data_year, dismissed_data[['match_id', 'innings', 'ball', 'bowler','League','Wicket']],
                             on=['match_id', 'innings', 'bowler','ball','League'], how='left')
    combineddata3['Wicket'].fillna(0, inplace=True)

    player_outs = combineddata3.groupby(['bowler', 'venue', 'over','League'])[['Wicket']].sum().reset_index()
    player_outs.columns = ['Player', 'Venue', 'Over', 'Wicket','League']

    over_outs = combineddata3.groupby(['venue', 'over','League'])[['Wicket']].sum().reset_index()
    over_outs.columns = ['Venue', 'Over', 'Wickets','League']

    # Group by player and aggregate the runs scored
    player_runs = combineddata3.groupby(['bowler', 'venue', 'over','League'])[['RC', 'B']].sum().reset_index()
    # Rename the columns for clarity
    player_runs.columns = ['Player', 'Venue', 'Over', 'Runs Conceded', 'B','League']

    # Display the merged DataFrame
    over_runs = combineddata3.groupby(['venue', 'over','League'])[['RC', 'B']].sum().reset_index()
    over_runs.columns = ['Venue', 'Over', 'Runs', 'Balls','League']

    combined_df = pd.merge(player_runs, player_outs, on=['Player', 'Venue', 'Over','League'], how='left')

    # Merge the two DataFrames on the 'Player' column
    combined_df2 = pd.merge(over_runs, over_outs, on=['Venue', 'Over','League'], how='left')

    # Merge the grouped data with the original data
    merged_data = pd.merge(combined_df, combined_df2, on=['Venue', 'Over','League'], how='left').reset_index()
    merged_data['Wickets'].fillna(0, inplace=True)
    merged_data['Wicket'].fillna(0, inplace=True)

    merged_data['Over_Runs'] =merged_data['Runs'] - merged_data['Runs Conceded']
    merged_data['Over_B'] =merged_data['Balls'] - merged_data['B']
    merged_data['Over_Wickets'] =merged_data['Wickets'] - merged_data['Wicket']

    merged_data['BER'] = merged_data['Over_Runs'] / merged_data['Over_B']
    merged_data['OPB'] = merged_data['Over_Wickets'] / merged_data['Over_B']

    # Calculate Expected RC and Expected Wickets for each row
    merged_data['Expected Runs Conceded'] = merged_data['B'] * merged_data['BER']
    merged_data['Expected Wickets'] = merged_data['B'] * merged_data['OPB']

    # Calculate Expected RC and Expected Wickets for each row
    merged_data['Expected Runs Conceded'] = merged_data['B'] * merged_data['BER']
    merged_data['Expected Wickets'] = merged_data['B'] * merged_data['OPB']

    players_years = combineddata3[['bowler', 'bowling_team', 'year','League']].drop_duplicates()
    players_years.columns = ['Player', 'Team', 'Year','League']

    # Group by bowler and sum the columns for final results
    truevalues = merged_data.groupby(['Player','League'])[
        ['B', 'Runs Conceded', 'Wicket', 'Expected Runs Conceded', 'Expected Wickets']].sum().reset_index()

    final_results = truemetrics(truevalues)
    # final_results3 = pd.merge(analysis_results, final_results, on='Player', how='left')
    final_results4 = pd.merge(players_years, final_results, on=['Player','League'], how='left')
    return final_results4.round(2)


# Load the data
@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename, low_memory=False)
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
    return data

# The main app function

def main():
    st.title('Bowling True Values')

    league = st.multiselect('Choose an option:', ['IPL','PSL','SA20','T20I (test playing nations only)', 'T20 WCs', 'CPL','LPL'])
    dic = {'IPL':'all_matches.csv','PSL':'PSL.csv','SA20':'SA20.csv','T20I (test playing nations only)':'testplayingnations.csv', 'T20 WCs':'t20wcs.csv', 'CPL':'CPL.csv','LPL':'LPL.csv'}
    #
    leagues = []
    for l in league:
        d = load_data(dic[l])
        d['League'] = l
        leagues.append(d)

    if league:
        data =pd.concat(leagues, ignore_index=True)

        years = data['year'].unique()
        # Load your data

        # Selectors for user input
        options = ['Overall Stats', 'Season By Season']
        # Create a select box
        choice = st.selectbox('Select your option:', options)
        choice2 = st.selectbox('Individual Player or Everyone:', ['Individual','Everyone'])
        start_year, end_year = st.slider('Select Years Range:', min_value=min(years), max_value=max(years), value=(min(years), max(years)))
        start_over, end_over = st.slider('Select Overs Range:', min_value=1, max_value=20, value=(1, 20))
        start_runs,end_runs = st.slider('Select Minimum Wickets:', min_value=0, max_value=300, value=(0, 500))
        start_runs1,end_runs1 = st.slider('Select Minimum Balls Bowled:', min_value=1, max_value=10000, value=(1, 10000))

        inns = [1,2]

        if choice2 == 'Individual':
            players = data['bowler'].unique()
            player = st.multiselect("Select Players:", players)
            # name = st.selectbox('Choose the Player From the list', data['striker'].unique())

        inn = st.multiselect("Select innings:", inns)

        # A button to trigger the analysis
        if st.button('Analyse'):
            # Call a hypothetical function to analyze data
            all_data = []

            filtered_data = data[(data['over'] >= start_over) & (data['over'] <= end_over)]
            filtered_data2 = filtered_data[(filtered_data['year'] >= start_year) & (filtered_data['year'] <= end_year)]

            # Analyze data and save results for each year
            for year in filtered_data2['year'].unique():
                results = analyze_data_for_year(year, filtered_data2)
                all_data.append(results)

            combined_data = pd.concat(all_data, ignore_index=True)
            most_frequent_team = combined_data.groupby('Player')['Team'].agg(lambda x: x.mode().iat[0]).reset_index()

            truevalues = combined_data.groupby('Player')[['B', 'Runs Conceded', 'Wicket', 'Expected Runs Conceded', 'Expected Wickets']].sum()

            final_results = truemetrics(truevalues)

            final_results2 = pd.merge(most_frequent_team, final_results, on='Player', how='left')


            final_results4 = final_results2.sort_values(by=['Wicket'],ascending=False)
            final_results4 = final_results4[(final_results4['Wicket'] >= start_runs) & (final_results4['Wicket'] <= end_runs)]
            final_results4 = final_results4[(final_results4['B'] >= start_runs1) & (final_results4['B'] <= end_runs1)]
            if choice == 'Overall Stats':
                # Display the results
                if choice2 == 'Individual':
                    temp = []
                    for i in player:
                        if i in final_results4['Player'].unique():
                            temp.append(i)
                        else:
                            st.subheader(f'{i} not in this list')
                    final_results4 = final_results4[final_results4['Player'].isin(temp)]
                final_results4 = final_results4.sort_values(by=['Wicket'], ascending=False)
                st.dataframe(final_results4.round(2))

            elif choice == 'Season By Season':
                temp = []
                for i in player:
                    if i in combined_data['Player'].unique():
                        temp.append(i)
                    else:
                        st.subheader(f'{i} not in this list')
                combined_data = combined_data[combined_data['Player'].isin(temp)]
                combined_data = combined_data.sort_values(by=['Wicket'], ascending=False)
                combined_data = combined_data[(combined_data['Wicket'] >= start_runs) & (combined_data['Wicket'] <= end_runs)]
                combined_data = combined_data[(combined_data['B'] >= start_runs1) & (combined_data['B'] <= end_runs1)]
                st.dataframe(combined_data)



# Run the main function
if __name__ == '__main__':
    main()
