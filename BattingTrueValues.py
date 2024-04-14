import pandas as pd
import streamlit as st


# combined_data = combined_data[combined_data['BowlCat'] == 'S']
# combined_data = combined_data[combined_data['BatType'] == 'R']

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
    truevalues2 = truevalues.groupby(['Player', 'Over'])[['Runs Scored', 'BF', 'Out']].sum().reset_index()
    truevalues3 = truevalues.groupby(['Over'])[['Runs Scored', 'BF', 'Out']].sum().reset_index()
    truevalues3.columns = ['Over', 'Mean Runs', 'Mean BF', 'Mean Outs']
    truevalues4 = pd.merge(truevalues2, truevalues3, on=['Over'], how='left')
    truevalues4['SR'] = truevalues4['Runs Scored'] / truevalues4['BF'] * 100
    truevalues4['Mean SR'] = truevalues4['Mean Runs'] / truevalues4['Mean BF'] * 100
    return truevalues4


def calculate_entry_point_all_years(data):
    # Identifying the first instance each batter faces a delivery in each match
    print(data.columns)
    first_appearance = data.drop_duplicates(subset=['MatchNum', 'MatchInn', 'Batter', 'Types', ])
    first_appearance = first_appearance.copy()

    # Converting overs and deliveries into a total delivery count
    first_appearance.loc[:, 'total_deliveries'] = first_appearance['Over'].apply(
        lambda x: int(x) * 6 + int((x - int(x)) * 10))
    # Calculating the average entry point for each batter in total deliveries
    avg_entry_point_deliveries = first_appearance.groupby(['Batter', 'Types', ])[
        'total_deliveries'].median().reset_index()

    # Converting the average entry point from total deliveries to overs and Overs and rounding to 1 decimal place
    avg_entry_point_deliveries['average_over'] = avg_entry_point_deliveries['total_deliveries'].apply(
        lambda x: round((x // 6) + (x % 6) / 10, 1))

    return avg_entry_point_deliveries[['Batter', 'Types', 'average_over']], first_appearance


def calculate_first_appearance(data):
    # Identifying the first instance each batter faces a delivery in each match
    first_appearance = data.drop_duplicates(subset=['MatchNum', 'MatchInn', 'Batter', 'Types', ])
    # Converting overs and deliveries into a total delivery count
    first_appearance.loc[:, 'total_deliveries'] = first_appearance['Over'].apply(
        lambda x: int(x) * 6 + int((x - int(x)) * 10))

    # Calculating the average entry point for each batter in total deliveries
    avg_entry_point_deliveries = first_appearance.groupby(['Batter', 'year', 'Types', ])[
        'total_deliveries'].median().reset_index()

    # Converting the average entry point from total deliveries to overs and Overs
    avg_entry_point_deliveries['average_over'] = (
        avg_entry_point_deliveries['total_deliveries'].apply(lambda x: (x // 6) + (x % 6) / 10)).round(1)

    return avg_entry_point_deliveries[['Batter', 'Types', 'average_over']]


def analyze_data_for_year2(data):
    # Filter the data for the specified year
    year_data = data.copy()

    # Calculate the first appearance of each batter in each match for the year
    first_appearance_data = calculate_first_appearance(year_data)

    # Calculate the average entry point for each batter

    # Assuming other analysis results are in a DataFrame named 'analysis_results'
    if 'analysis_results' in locals() or 'analysis_results' in globals():
        # Merge the average entry point data with other analysis results
        analysis_results = pd.merge(year_data, first_appearance_data, on=['Batter', 'Types', ],
                                    how='left')
    else:
        # Use average entry point data as the primary analysis result
        analysis_results = first_appearance_data

    return analysis_results


def analyze_data_for_year3(year2, data2):
    combineddata2 = data2[data2['MatchInn'] < 3].copy()
    combineddata = combineddata2[combineddata2['year'] == year2].copy()
    inns = combineddata.groupby(['Batter', 'MatchNum', 'Types'])[['Runs']].sum().reset_index()
    inns['I'] = 1
    inns2 = inns.groupby(['Batter', 'Types'])[['I']].sum().reset_index()
    inns2.columns = ['Player', 'Types', 'I']
    inns3 = inns.copy()
    inns['CI'] = inns.groupby(['Batter', 'Types'], as_index=False)[['I']].cumsum()
    analysis_results = analyze_data_for_year2(combineddata)
    analysis_results.columns = ['Player', 'Types', 'Median Entry Point']

    valid = ['X', 'WX']
    # Filter out rows where a player was dismissed
    dismissed_data = combineddata[combineddata['Notes'].isin(valid)]
    dismissed_data = dismissed_data[dismissed_data['LongDis'] != 'run out']
    dismissed_data['Out'] = 1

    combineddata2 = pd.merge(combineddata, dismissed_data[['MatchNum', 'MatchInn', 'Batter', 'Notes', 'over', 'Out']],
                             on=['MatchNum', 'MatchInn', 'Batter', 'Notes', 'over'], how='left')
    combineddata2['Out'].fillna(0, inplace=True)
    combineddata = combineddata2.copy()

    player_outs = dismissed_data.groupby(['Batter', 'Venue', 'over', 'Types'])[['Out']].sum().reset_index()
    player_outs.columns = ['Player', 'Venue', 'Over', 'Types', 'Out']

    over_outs = dismissed_data.groupby(['Venue', 'over', 'Types'])[['Out']].sum().reset_index()
    over_outs.columns = ['Venue', 'Over', 'Types', 'Outs']

    # Group by player and aggregate the runs scored
    player_runs = combineddata.groupby(['Batter', 'Venue', 'over', 'Types'])[['Runs', 'B']].sum().reset_index()
    # Rename the columns for clarity
    player_runs.columns = ['Player', 'Venue', 'Over', 'Types', 'Runs Scored', 'BF']

    # Display the merged DataFrame
    over_runs = combineddata.groupby(['Venue', 'over', 'Types'])[['Runs', 'B']].sum().reset_index()
    over_runs.columns = ['Venue', 'Over', 'Types', 'Runs', 'B']
    # Merge the two DataFrames on the 'Player' column

    combined_df = pd.merge(player_runs, player_outs, on=['Player', 'Venue', 'Over', 'Types'], how='left')
    # Merge the two DataFrames on the 'Player' column
    combined_df2 = pd.merge(over_runs, over_outs, on=['Venue', 'Over', 'Types'], how='left')
    # Calculate BSR and OPB for each ball at each Venue
    combined_df2['BSR'] = combined_df2['Runs'] / combined_df2['B']
    combined_df2['OPB'] = combined_df2['Outs'] / combined_df2['B']

    combined_df3 = pd.merge(combined_df, combined_df2, on=['Venue', 'Over', 'Types'], how='left')
    combined_df3['Expected Runs'] = combined_df3['BF'] * combined_df3['BSR']
    combined_df3['Expected Outs'] = combined_df3['BF'] * combined_df3['OPB']

    truevalues = combined_df3.groupby(['Player', 'Types'])[
        ['Runs Scored', 'BF', 'Out', 'Expected Runs', 'Expected Outs']].sum()

    final_results = truemetrics(truevalues)

    players_years = combineddata[['Batter', 'year', 'Types']].drop_duplicates()
    players_years.columns = ['Player', 'Year', 'Types']
    final_results2 = pd.merge(inns2, final_results, on=['Player', 'Types'], how='left')
    final_results3 = pd.merge(players_years, final_results2, on=['Player', 'Types'], how='left')
    final_results4 = pd.merge(final_results3, analysis_results, on=['Player', 'Types'], how='left')
    truevalues = truemetrics2(combined_df3)
    return final_results4.round(2)


# Load the data
@st.cache
def load_data(filename):
    data = pd.read_csv(filename, low_memory=False)
    return data


# The main app function
def main():
    st.title('Batting True Values by Bowling Type')

    data = pd.read_csv('IPLData5.csv', low_memory=False)
    # Set 'B' to 0 for deliveries that are wides
    data['B'] = 1

    # Set 'B' to 0 for deliveries that are wides
    # Assuming 'wides' column exists and is non-zero for wide balls
    data.loc[data['Notes'] == 'W', 'B'] = 0

    data['RC'] = data['Runs'] + data['Extras']

    # Extract the year from the 'start_date' column

    data['year'] = pd.to_datetime(data['StartDate'], format='mixed').dt.year

    # Remove any potential duplicate rows
    combined_data = data.drop_duplicates()

    years = data['year'].unique()

    # Remove any potential duplicate rows
    data = data.drop_duplicates()

    data['ball2'] = pd.to_numeric(data['Over'], errors='coerce')
    data['over'] = data['ball2'] // 1 + 1

    rpace = ['RF', 'RFM', 'RM', 'RMF']
    lpace = ['LF', 'LF/LM', 'LFM', 'LM', 'LMF']
    roff = ['OB', 'RM/OB', 'S']
    loff = ['SLA']
    rleg = ['LB']
    lleg = ['SLW']

    data['Types'] = 'L'
    data.loc[data['BowlType'].isin(rpace), 'Types'] = 'Right Arm Pace'
    data.loc[data['BowlType'].isin(lpace), 'Types'] = 'Left Arm Pace'
    data.loc[data['BowlType'].isin(roff), 'Types'] = 'Right Arm Finger Spin'
    data.loc[data['BowlType'].isin(loff), 'Types'] = 'Left Arm Finger Spin'
    data.loc[data['BowlType'].isin(rleg), 'Types'] = 'Right Arm Wrist Spin'
    data.loc[data['BowlType'].isin(lleg), 'Types'] = 'Left Arm Wrist Spin'

    types = ['Right Arm Pace', 'Left Arm Pace', 'Right Arm Finger Spin', 'Left Arm Finger Spin', 'Right Arm Wrist Spin',
             'Left Arm Wrist Spin', ]
    # Selectors for user input
    options = ['Overall Stats', 'Season By Season']
    # Create a select box
    choice = st.selectbox('Select your option:', options)
    choice2 = st.selectbox('Individual Player or Everyone:', ['Individual', 'Everyone'])
    start_year, end_year = st.slider('Select Years Range:', min_value=min(years), max_value=max(years),
                                     value=(min(years), max(years)))
    start_over, end_over = st.slider('Select Overs Range:', min_value=1, max_value=20, value=(1, 20))
    filtered_data = data[(data['over'] >= start_over) & (data['over'] <= end_over)]
    filtered_data2 = filtered_data[(filtered_data['year'] >= start_year) & (filtered_data['year'] <= end_year)]
    if choice2 == 'Individual':
        name = st.selectbox('Choose the Player From the list', data['Batter'].unique())
    x = filtered_data2
    # A button to trigger the analysis
    if st.button('Analyse'):
        # Call a hypothetical function to analyze data
        all_data = []

        # Analyze data and save results for each year
        for year in filtered_data2['year'].unique():
            results = analyze_data_for_year3(year, filtered_data2)
            all_data.append(results)

        combined_data = pd.concat(all_data, ignore_index=True)

        truevalues = combined_data.groupby(['Player', 'Types'])[
            ['I', 'Runs Scored', 'BF', 'Out', 'Expected Runs', 'Expected Outs']].sum().reset_index()
        final_results = truemetrics(truevalues)

        final_results = final_results.sort_values(by=['Runs Scored'], ascending=False)
        if choice == 'Overall Stats':
            # Display the results
            if choice2 == 'Individual':
                if name in final_results['Player'].unique():
                    final_results = final_results[final_results['Player'] == name]
                else:
                    st.subheader('Player not in this list')
            final_results = final_results.sort_values(by=['Runs Scored'], ascending=False)
            st.dataframe(final_results.round(2))
        elif choice == 'Season By Season':
            if choice2 == 'Individual':
                if name in combined_data['Player'].unique():
                    combined_data = combined_data[combined_data['Player'] == name]
                else:
                    st.subheader('Player not in this list')
            combined_data = combined_data.sort_values(by=['Runs Scored'], ascending=False)
            st.dataframe(combined_data)


# Run the main function
if __name__ == '__main__':
    main()