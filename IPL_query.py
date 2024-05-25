import pandas as pd
import numpy as np
import streamlit as st

per_del = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
match = pd.read_csv('IPL_Matches_2008_2022.csv')

ipl_teams = {'Rajasthan Royals':'RR','Royal Challengers Bangalore':'RCB','Sunrisers Hyderabad':'SRH',
            'Delhi Capitals':'DCs','Chennai Super Kings':'CSK','Gujarat Titans':'GT','Lucknow Super Giants':'LSG',
            'Kolkata Knight Riders':'KKR','Punjab Kings':'PBKS','Mumbai Indians':'MI','Rising Pune Supergiant':'RPS',
            'Gujarat Lions':'GL','Pune Warriors India':'PWI','Deccan Chargers':'DC','Kochi Tuskers Kerala':'KTK'}

if "seasonn" not in st.session_state:
    st.session_state['seasonn'] = 2022

def Datacleaning():
    match['City'] = match['City'].fillna('Dubai')
    match['City'] = match['City'].replace('Bengaluru', 'Bangalore')
    match['Date']=match['Date'].astype('datetime64[ns]').dt.date
    per_del['BattingTeam'] = per_del['BattingTeam'].replace(
        ['Rising Pune Supergiants', 'Delhi Daredevils', 'Kings XI Punjab','Pune Warriors'],
        ['Rising Pune Supergiant', 'Delhi Capitals', 'Punjab Kings','Pune Warriors India'])
    match['Team1'] = match['Team1'].replace(['Rising Pune Supergiants', 'Delhi Daredevils', 'Kings XI Punjab','Pune Warriors'],
                                            ['Rising Pune Supergiant', 'Delhi Capitals', 'Punjab Kings','Pune Warriors India'])
    match['Team2'] = match['Team2'].replace(['Rising Pune Supergiants', 'Delhi Daredevils', 'Kings XI Punjab','Pune Warriors'],
                                            ['Rising Pune Supergiant', 'Delhi Capitals', 'Punjab Kings','Pune Warriors India'])
    match['WinningTeam'] = match['WinningTeam'].replace(['Rising Pune Supergiants', 'Delhi Daredevils', 'Kings XI Punjab','Pune Warriors'],
                                                        ['Rising Pune Supergiant', 'Delhi Capitals', 'Punjab Kings','Pune Warriors India'])
    match['TossWinner'] = match['TossWinner'].replace(['Rising Pune Supergiants', 'Delhi Daredevils', 'Kings XI Punjab','Pune Warriors'],
                                                      ['Rising Pune Supergiant', 'Delhi Capitals', 'Punjab Kings','Pune Warriors India'])

    match['Season'] = match['Season'].replace(['2007/08', '2009/10', '2020/21'], ['2008', '2010', '2020'])
    match['Season'] = match['Season'].astype(int)
    match['Venue'] = match['Venue'].replace(
        ['Narendra Modi Stadium, Ahmedabad', 'Eden Gardens, Kolkata', 'Wankhede Stadium, Mumbai'
            , 'Brabourne Stadium, Mumbai', 'Dr DY Patil Sports Academy, Mumbai',
         'Maharashtra Cricket Association Stadium, Pune',
         'Zayed Cricket Stadium, Abu Dhabi', 'Arun Jaitley Stadium, Delhi', 'MA Chidambaram Stadium, Chepauk, Chennai'
            , 'Rajiv Gandhi International Stadium, Uppal', 'M Chinnaswamy Stadium', 'Feroz Shah Kotla',
         'Punjab Cricket Association IS Bindra Stadium, Mohali'
            , 'MA Chidambaram Stadium, Chepauk', 'Sardar Patel Stadium, Motera', 'Subrata Roy Sahara Stadium',
         'Punjab Cricket Association Stadium, Mohali','Dubai International Cricket Stadium','Punjab Cricket Association IS Bindra Stadium'
         ,'Maharashtra Cricket Association Stadium','Rajiv Gandhi International Stadium','Himachal Pradesh Cricket Association Stadium',
         'Dr DY Patil Sports Academy'],
        ['Narendra Modi Stadium', 'Eden Gardens', 'Wankhede Stadium', 'Brabourne Stadium'
            , 'Dr DY Patil Sports Academy', 'MCA Stadium'
            , 'Sheikh Zayed Stadium', 'Arun Jaitley Stadium', 'MA Chidambaram Stadium',
         'R.Gandhi International Stadium'
            , 'M.Chinnaswamy Stadium', 'Arun Jaitley Stadium', 'IS Bindra Stadium'
            , 'MA Chidambaram Stadium', 'Narendra Modi Stadium', 'MCA Stadium',
         'IS Bindra Stadium','Dubai International Stadium','IS Bindra Stadium','MCA Stadium','R.Gandhi International Stadium'
         ,'Dharamshala Stadium','DY Patil Stadium'])
    return per_del,match

per_del,match = Datacleaning()

most_runs = per_del.groupby('batter')['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
batsman_most_run = most_runs.iloc[0]['batter']
most_run = int(most_runs.iloc[0]['batsman_run'])

HS = per_del.groupby(['ID','innings','batter'])['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)[['batter','batsman_run']]
batsman_HS = HS.iloc[0]['batter']
HS_run = int(HS.iloc[0]['batsman_run'])

most_4s = per_del[per_del['batsman_run']==4].groupby(['batter'])['ID'].count().reset_index().rename(columns={'ID':'4s'}).sort_values(by='4s',ascending=False)
batsman_4 = most_4s.iloc[0]['batter']
batsman_4count = int(most_4s.iloc[0]['4s'])

most_6s = per_del[per_del['batsman_run']==6].groupby(['batter'])['ID'].count().reset_index().rename(columns={'ID':'6s'}).sort_values(by='6s',ascending=False)
batsman_6 = most_6s.iloc[0]['batter']
batsman_6count = most_6s.iloc[0]['6s']

batsman_rec = per_del.groupby('batter').agg({
    'ballnumber':'count',
    'batsman_run':'sum'
}).reset_index()
most_faced_batsman = batsman_rec[batsman_rec['ballnumber']>=1000]
def StrikeRate(x):
    return (x['batsman_run']/x['ballnumber'])*100
most_faced_batsman['strike_rate'] = most_faced_batsman.apply(StrikeRate,axis=1)
top_strikers = most_faced_batsman.sort_values(by='strike_rate',ascending=False)
top_striker = top_strikers.iloc[0]['batter']
top_strike_rate = round(top_strikers.iloc[0]['strike_rate'],2)

match_per_city = match['City'].value_counts().reset_index()
match_per_city['City'] = np.where(match_per_city['count']>10,match_per_city.City,'others')
updated_match_per_city = match_per_city.groupby('City')['count'].sum().reset_index()


per_del_venue = per_del.merge(match[['ID', 'Venue']], on='ID', how='left')
def boundary_count(x):
    boundary_in_stadium = per_del_venue[per_del_venue['Venue'] == x['Stadium']]
    four_in_stadium = boundary_in_stadium[boundary_in_stadium['batsman_run'] == 4]
    six_in_stadium = boundary_in_stadium[boundary_in_stadium['batsman_run'] == 6]
    return pd.Series([four_in_stadium.shape[0], six_in_stadium.shape[0]])

boundary_per_stadium = pd.DataFrame(match.Venue.unique(), columns=['Stadium'])
boundary_per_stadium[['4s', '6s']] = boundary_per_stadium.apply(boundary_count, axis=1)
boundary_per_stadium['Stadium'] = np.where((boundary_per_stadium['4s'] + boundary_per_stadium['6s']) > 900,
                                           boundary_per_stadium.Stadium, 'others')
boundary_per_stadium = boundary_per_stadium.groupby('Stadium').agg({
    '4s': 'sum',
    '6s': 'sum'
}).reset_index()

per_del_venue = per_del.merge(match[['ID','Venue']],on='ID',how='left')
per_match_score = per_del_venue.groupby(['Venue','ID','innings'])['total_run'].sum().reset_index()
avgRun_per_stadium = pd.DataFrame(per_match_score.groupby('Venue')['total_run'].mean().sort_values(ascending=False).reset_index()).rename(columns={'total_run':'avg_run'})
avgRun_1_inning = per_match_score[per_match_score['innings']==1].groupby('Venue')['total_run'].mean().reset_index().rename(columns={'total_run':'avg_1st_inning_score'})
avgRun_2_inning = per_match_score[per_match_score['innings']==2].groupby('Venue')['total_run'].mean().reset_index().rename(columns={'total_run':'avg_2nd_inning_score'})
runs_per_stadium = avgRun_per_stadium.merge(avgRun_1_inning,on='Venue',how='left').merge(avgRun_2_inning,on='Venue',how='left')[:15]

Most_dot_ball = per_del[(per_del['total_run']==0)|(per_del['extra_type'].isin(['legbyes','byes']))].groupby('bowler')['ID'].count().reset_index().rename(columns={'ID':'dot_ball'}).sort_values(by='dot_ball',ascending=False)
bowler_most_dot_ball = Most_dot_ball.iloc[0]['bowler']
most_dot_ball_count = int(Most_dot_ball.iloc[0]['dot_ball'])

bowlers_wicket = per_del[
    (per_del['kind'].isin(['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket', 'nan'])) | per_del[
        'kind'].isnull()]
@st.cache_data
def myfunc(x):
    bowl = bowlers_wicket[(bowlers_wicket['ID'] == x['ID']) & (bowlers_wicket['innings'] == x['innings'])]
    bowl1 = bowl[bowl['bowler'] == x['bowler']]
    bowl2 = bowl1[bowl1['kind'].notnull()]
    return bowl2.shape[0]
bowling_fig = pd.DataFrame(
    bowlers_wicket.groupby(['ID', 'innings', 'bowler'])['batsman_run'].sum().reset_index().rename(
        columns={'batsman_run': 'runs'}))
bowling_fig['wickets'] = bowling_fig.apply(myfunc, axis=1)
bowling_fig = bowling_fig.sort_values(by=['wickets', 'runs'], ascending=[False, True])
best_fig_bowler = bowling_fig.iloc[0]['bowler']
best_figure = str(bowling_fig['wickets'].iloc[0])+'/'+str(bowling_fig['runs'].iloc[0])

@st.cache_data
def cal_economy(x):
    bowler_data = per_del[per_del['bowler']==x['bowler']]
    bowler_over = bowler_data.drop_duplicates(subset=['ID','innings','overs'],keep='first').shape[0]
    total_run = bowler_data['total_run'].sum()
    return total_run/bowler_over
bowlers_wicket = per_del[per_del['kind'].isin(['caught','caught and bowled','bowled', 'stumped','lbw', 'hit wicket'])]
bowler_avg = bowlers_wicket.bowler.value_counts().reset_index().rename(columns={'count':'wickets'})
bowler_avg['runs'] = bowler_avg.apply(lambda x: per_del[per_del['bowler']==x['bowler']]['total_run'].sum(),axis=1)
bowler_avg['avg'] = bowler_avg.apply(lambda x: x['runs']/x['wickets'],axis=1)
bowler_avg['economy_rate'] = bowler_avg.apply(cal_economy,axis=1)
bowler_most_wickets = bowler_avg.sort_values(by='wickets',ascending=False)[:1].iloc[0]['bowler']
bowler_Mwicket_count = bowler_avg.sort_values(by='wickets',ascending=False)[:1].iloc[0]['wickets']
bowler_best_average = bowler_avg[bowler_avg['wickets']>=25].sort_values(by='avg',ascending=True)[:1].iloc[0]['bowler']
best_average = bowler_avg[bowler_avg['wickets']>=25].sort_values(by='avg',ascending=True)[:1].iloc[0]['avg']
bowler_best_economy = bowler_avg[bowler_avg['wickets']>=25].sort_values(by='economy_rate',ascending=True)[:1].iloc[0]['bowler']
best_economy = bowler_avg[bowler_avg['wickets']>=25].sort_values(by='economy_rate',ascending=True)[:1].iloc[0]['economy_rate']

stumping = per_del[per_del['kind']=='stumped']['fielders_involved'].value_counts(ascending=False).reset_index().rename(columns={'count':'stumping'})
caught = per_del[per_del['kind']=='caught']['fielders_involved'].value_counts(ascending=False).reset_index().rename(columns={'count':'catches'})
run_out = per_del[per_del['kind']=='run out']['fielders_involved'].value_counts(ascending=False).reset_index().rename(columns={'count':'run_out'})
fielders = caught.merge(run_out,on='fielders_involved',how='left').merge(stumping,on='fielders_involved',how='left')
fielders = fielders.fillna(0)
fielders['total_dismissals'] = fielders.apply(lambda x: x['catches']+x['run_out']+x['stumping'],axis=1)
fielders = fielders.sort_values(by='total_dismissals',ascending=False)

trophy = match[match['MatchNumber']=='Final']['WinningTeam'].value_counts().reset_index()
labels = trophy['WinningTeam']
def funcc(x):
    return match[(match['MatchNumber']=='Final')&(match['WinningTeam']==x['WinningTeam'])]['Season'].to_list()
trophy['winning_season'] = trophy.apply(funcc,axis=1)
team_winning_season = trophy['winning_season']

@st.cache_data
def team_stats(x):
    team = match[(match['Team1']==x['Team'])|(match['Team2']==x['Team'])]
    wins = team[team['WinningTeam']==x['Team']].shape[0]
    total_match = team.shape[0]
    unknown = team[team['WinningTeam'].isnull()].shape[0]
    win_percentage = np.round((wins/total_match)*100,2)
    return pd.Series([total_match,wins,unknown,win_percentage])
team_record = pd.DataFrame(match['Team1'].unique(),columns=['Team'])
team_record[['matches','wins','unknown','win%']] = team_record.apply(team_stats,axis=1)
team_record = team_record.sort_values(by='win%',ascending=False)
team_record = team_record.reset_index().drop('index',axis=1)

match_without_tie = match[match['Margin'].notnull()]
def fun(x):
    temp = match_without_tie[match_without_tie['TossDecision']==x['TossDecision']]
    win = temp[temp['TossWinner'] == temp['WinningTeam']]
    loss = temp[temp['TossWinner'] != temp['WinningTeam']]
    return pd.Series([win.shape[0],loss.shape[0]])

match_toss_result = pd.DataFrame(match_without_tie.TossDecision.unique(),columns=['TossDecision'])
match_toss_result[['win','loss']] = match_toss_result.apply(fun,axis=1)
toss_result = match_toss_result.melt(id_vars='TossDecision')

Toss_decision_per_season = match.groupby(['Season','TossDecision'])['ID'].count().unstack().reset_index().rename(columns={'bat':'Bat 1st','field':'Bat 2nd'})
Toss_decision_per_season = Toss_decision_per_season.melt(id_vars='Season',value_name='count')

Most_run_per_season = per_del.merge(match[['ID','Season']],on='ID',how='left').groupby(['Season','batter'])['batsman_run'].sum().sort_values(ascending=False).reset_index().drop_duplicates(subset=['Season'],keep='first')
Most_run_per_season = Most_run_per_season.rename(columns={'batter':'Most_run_batter','batsman_run':'total_runs'})
batter_list = Most_run_per_season.sort_values(by='Season')['Most_run_batter'].tolist()

wickets_in_season = per_del[per_del['kind'].isin(['caught','caught and bowled','bowled', 'stumped','lbw', 'hit wicket'])].merge(match[['ID','Season']],on='ID',how='left')
wickets_per_season = wickets_in_season.groupby(['Season','bowler'])['ID'].count().sort_values(ascending=False).reset_index().drop_duplicates(subset=['Season'],keep='first').rename(columns={'ID':'wickets'})
wickets_per_season = wickets_per_season.sort_values(by='Season')

HS_per_season = per_del.merge(match[['ID','Season']],on='ID',how='left').groupby(['Season','ID','innings','batter'])['batsman_run'].sum().sort_values(ascending=False).reset_index().drop_duplicates(subset=['Season'],keep='first')[['Season','batter','batsman_run']]
HS_per_season = HS_per_season.rename(columns={'batter':'batsman','batsman_run':'High_Score'})
HS_per_season = HS_per_season.sort_values(by='Season')

individual_score = per_del.merge(match[['ID','Season']],on='ID',how='left').groupby(['Season','ID','innings','batter'])['batsman_run'].sum().reset_index().rename(columns={'batsman_run':'runs'})
fifties_per_season = individual_score[(individual_score['runs']>=50)&(individual_score['runs']<100)].groupby('Season')['ID'].count().reset_index().rename(columns={'ID':'Half_century'})
century_per_season = individual_score[individual_score['runs']>=100].groupby('Season')['ID'].count().reset_index().rename(columns={'ID':'Centuries'})
record_per_season = century_per_season.merge(fifties_per_season,on='Season',how='left').melt(id_vars='Season',var_name='Record',value_name='count')


def partner_name(x):
    batter_pair = []
    batter_pair.append(x['batter'])
    batter_pair.append(x['non-striker'])
    batter_pair.sort()
    return batter_pair[0] + '-' + batter_pair[1]
partnership_run = per_del.merge(match[['ID', 'Season']], on='ID', how='left').groupby(
    ['Season', 'ID', 'innings', 'batter', 'non-striker'])['total_run'].sum().reset_index()
partnership_run['batsman_pair'] = partnership_run.apply(partner_name, axis=1)

highest_partnership_per_season = partnership_run[['Season', 'ID', 'innings', 'batsman_pair', 'total_run']].groupby(
    ['Season', 'ID', 'innings', 'batsman_pair'])['total_run'].sum().sort_values(
    ascending=False).reset_index().drop_duplicates(subset=['Season'], keep='first')[
    ['Season', 'batsman_pair', 'total_run']]
highest_partnership_per_season = highest_partnership_per_season.rename(columns={'total_run': 'runs'})
highest_partnership_per_season = highest_partnership_per_season.sort_values(by='Season')


seasons = match['Season'].unique().tolist()
def new_season():
    global seasonal_data,teams_in_season
    season = match[match['Season']==st.session_state['seasonn']]
    seasonal_data = per_del.merge(match[['ID', 'Season', 'Team1', 'Team2', 'Date', 'method', 'Margin']], on='ID',how='left')
    seasonal_data = seasonal_data[seasonal_data['Season'] == st.session_state['seasonn']]
    def bowling_team(x):
        if x['BattingTeam'] == x['Team1']:
            return x['Team2']
        else:
            return x['Team1']

    seasonal_data['BowlingTeam'] = seasonal_data.apply(bowling_team, axis=1)
    teams_in_season = pd.DataFrame(seasonal_data['BowlingTeam'].unique(), columns=['Teams'])
    return season
def winner():
    data = new_season()
    return data[data['MatchNumber'] == 'Final'].iloc[0]['WinningTeam']

def runner_up():
    data = new_season()
    dete = data[data['MatchNumber'] == 'Final']
    win_team = dete.iloc[0]['WinningTeam']
    if dete.iloc[0]['Team1']==win_team:
        return dete.iloc[0]['Team2']
    else :
        return dete.iloc[0]['Team1']

def teams_win_in_season():
    seasonal = new_season()
    team_stats_in_season = pd.DataFrame(seasonal['Team1'].unique(),columns=['Teams'])
    def stats(x):
        total_match = seasonal[(seasonal['Team1']==x['Teams'])|(seasonal['Team2']==x['Teams'])].shape[0]
        victory = seasonal[seasonal['WinningTeam']==x['Teams']].shape[0]
        return pd.Series([victory,total_match-victory])

    team_stats_in_season[['win','loss']] = team_stats_in_season.apply(stats,axis=1)
    team_stats_in_season['Teams'] = team_stats_in_season['Teams'].apply(lambda x: ipl_teams[x])
    team_stats_in_season = team_stats_in_season.melt(id_vars='Teams',var_name='Result',value_name='count')
    return team_stats_in_season

def loading_field_stats():
    def fielding_performance(x):
        data = seasonal_data
        catch = data[(data['kind'].isin(['caught', 'caught and bowled'])) & (
                    data['BowlingTeam'] == x['Teams'])].shape[0]
        run_out = data[(data['kind'] == 'run out') & (data['BowlingTeam'] == x['Teams'])].shape[0]
        stumping = data[(data['kind'] == 'stumped') & (data['BowlingTeam'] == x['Teams'])].shape[0]
        return pd.Series([catch, run_out, stumping])

    fielding_in_season = teams_in_season.copy()
    fielding_in_season[['catches', 'run outs', 'stumping']] = fielding_in_season.apply(fielding_performance, axis=1)
    fielding_in_season['total'] = fielding_in_season['catches'] + fielding_in_season['run outs'] + fielding_in_season[
        'stumping']
    fielding_in_season = fielding_in_season.sort_values(by='total', ascending=False).drop('total', axis=1)
    fielding_in_season['Teams'] = fielding_in_season['Teams'].apply(lambda x: ipl_teams[x])
    fielding_in_season = fielding_in_season.melt(id_vars='Teams', var_name='kind', value_name='dismissals')
    return fielding_in_season

def loading_run_rate():
    def run_rate_per_team(x):
        powerplay = np.round(seasonal_data[(seasonal_data['overs'] < 6) & (seasonal_data['BattingTeam'] == x['Teams'])]
                             .groupby(['BattingTeam', 'ID'])['total_run'].sum().reset_index()['total_run'].mean() / 6, 2)
        middle = np.round(seasonal_data[(seasonal_data['overs'] > 5) & (seasonal_data['overs'] < 16) & (seasonal_data['BattingTeam'] == x['Teams'])]
                          .groupby(['BattingTeam', 'ID'])['total_run'].sum().reset_index()['total_run'].mean() / 10, 2)
        death = np.round(seasonal_data[(seasonal_data['overs'] > 15) & (seasonal_data['BattingTeam'] == x['Teams'])]
                         .groupby(['BattingTeam', 'ID'])['total_run'].sum().reset_index()['total_run'].mean() / 4, 2)
        return pd.Series([powerplay, middle, death])

    runRate_in_season = teams_in_season.copy()
    runRate_in_season[['Powerplay_overs', 'Middle_overs', 'Death_overs']] = runRate_in_season.apply(run_rate_per_team,                                                                                           axis=1)
    runRate_in_season['Teams'] = runRate_in_season['Teams'].apply(lambda x: ipl_teams[x])
    runRate_in_season = runRate_in_season.melt(id_vars='Teams', var_name='overs', value_name='run_rate')
    return runRate_in_season

def loading_strike_rate():
    def strike_rate_per_team(x):
        pp = seasonal_data[(seasonal_data['BattingTeam'] == x['Teams']) & (seasonal_data['overs'] < 6) & ((seasonal_data['extra_type'].isnull()) | (seasonal_data['extra_type'].isin(['legbyes', 'byes', 'noballs'])))]['batsman_run']
        powerplay = np.round((pp.sum() / pp.count()) * 100, 2)
        mid = seasonal_data[(seasonal_data['BattingTeam'] == x['Teams']) & (seasonal_data['overs'] > 5) & (seasonal_data['overs'] < 16) & ((seasonal_data['extra_type'].isnull()) | (seasonal_data['extra_type'].isin(['legbyes', 'byes', 'noballs'])))]['batsman_run']
        middle = np.round((mid.sum() / mid.count()) * 100, 2)
        dead = seasonal_data[(seasonal_data['BattingTeam'] == x['Teams']) & (seasonal_data['overs'] > 15) & ((seasonal_data['extra_type'].isnull()) | (seasonal_data['extra_type'].isin(['legbyes', 'byes', 'noballs'])))]['batsman_run']
        death = np.round((dead.sum() / dead.count()) * 100, 2)
        return pd.Series([powerplay, middle, death])

    strikeRate_in_season = teams_in_season.copy()
    strikeRate_in_season[['Powerplay_overs', 'Middle_overs', 'Death_overs']] = strikeRate_in_season.apply(
        strike_rate_per_team, axis=1)
    strikeRate_in_season['Teams'] = strikeRate_in_season['Teams'].apply(lambda x: ipl_teams[x])
    strikeRate_in_season = strikeRate_in_season.melt(id_vars='Teams', var_name='over', value_name='strike_rate')
    return strikeRate_in_season

def seasonal_orange():
    seasonal_orange_cap = seasonal_data.groupby('batter')['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
    seasonal_orange_cap_batter = seasonal_orange_cap.iloc[0]['batter']
    seasonal_orange_cap_run = seasonal_orange_cap.iloc[0]['batsman_run']
    return seasonal_orange_cap_batter,seasonal_orange_cap_run

def seasonal_HS():
    seasonal_HScorer = seasonal_data.groupby(['ID', 'innings', 'batter'])['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)[['batter', 'batsman_run']]
    return seasonal_HScorer.iloc[0]['batter'],seasonal_HScorer.iloc[0]['batsman_run']

def seasonal_boundary():
    seasonal_4s = seasonal_data[seasonal_data['batsman_run'] == 4].groupby(['batter'])['ID'].count().reset_index().rename(columns={'ID': '4s'}).sort_values(by='4s', ascending=False)
    return seasonal_4s.iloc[0]['batter'],seasonal_4s.iloc[0]['4s']

def seasonal_sixes():
    seasonal_6s = seasonal_data[seasonal_data['batsman_run'] == 6].groupby(['batter'])['ID'].count().reset_index().rename(columns={'ID': '6s'}).sort_values(by='6s', ascending=False)
    return seasonal_6s.iloc[0]['batter'],seasonal_6s.iloc[0]['6s']

def seasonal_SR():
    batsman_rec = seasonal_data.groupby('batter').agg({
        'ballnumber': 'count',
        'batsman_run': 'sum'
    }).reset_index()
    most_faced_batsman = batsman_rec[batsman_rec['ballnumber'] >= 300]

    def StrikeRate(x):
        return round((x['batsman_run'] / x['ballnumber']) * 100,2)

    most_faced_batsman['strike_rate'] = most_faced_batsman.apply(StrikeRate, axis=1)
    SR = most_faced_batsman.sort_values(by='strike_rate', ascending=False)
    return SR.iloc[0]['batter'],SR.iloc[0]['strike_rate']

def seasonal_dotBalls():
    dot = seasonal_data[(seasonal_data['total_run'] == 0) | (per_del['extra_type'].isin(['legbyes', 'byes']))].groupby('bowler')['ID'].count().sort_values(ascending=False).reset_index()
    return dot.iloc[0]['bowler'],dot.iloc[0]['ID']

@st.cache_data
def seasonal_best_fig():
    bowlers_wicket = seasonal_data[
        (seasonal_data['kind'].isin(['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket', 'nan'])) |
        per_del['kind'].isnull()]
    @st.cache_data
    def myfunc(x):
        bowl = bowlers_wicket[(bowlers_wicket['ID'] == x['ID']) & (bowlers_wicket['innings'] == x['innings'])]
        bowl1 = bowl[bowl['bowler'] == x['bowler']]
        bowl2 = bowl1[bowl1['kind'].notnull()]
        return bowl2.shape[0]

    bowling_fig = pd.DataFrame(
        bowlers_wicket.groupby(['ID', 'innings', 'bowler'])['batsman_run'].sum().reset_index().rename(
            columns={'batsman_run': 'runs'}))
    bowling_fig['wicktes'] = bowling_fig.apply(myfunc, axis=1)
    bowling_fig = bowling_fig.sort_values(by=['wicktes', 'runs'], ascending=[False, True])
    return bowling_fig.iloc[0]['bowler'],str(bowling_fig.iloc[0]['wicktes'])+'/'+str(bowling_fig.iloc[0]['runs'])

@st.cache_data
def seasonal_bowling_stats():
    bowlers_wicket = seasonal_data[seasonal_data['kind'].isin(['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket'])]
    bowler_avgg = bowlers_wicket.bowler.value_counts().reset_index().rename(columns={'count': 'wickets'})
    bowler_avgg['runs'] = bowler_avgg.apply(lambda x: per_del[per_del['bowler'] == x['bowler']]['total_run'].sum(),axis=1)
    bowler_avgg['avg'] = bowler_avgg.apply(lambda x: x['runs'] / x['wickets'], axis=1)
    @st.cache_data
    def cal_economyy(x):
        bowler_data = seasonal_data[seasonal_data['bowler'] == x['bowler']]
        bowler_over = bowler_data.drop_duplicates(subset=['ID', 'innings', 'overs'], keep='first').shape[0]
        total_run = bowler_data['total_run'].sum()
        return total_run / bowler_over

    bowler_avgg['economy_rate'] = bowler_avgg.apply(cal_economyy, axis=1)

    wickets = bowler_avgg[bowler_avgg['wickets'] >= 0].sort_values(by='wickets', ascending=False)
    purple_cap_bowler = wickets.iloc[0]['bowler']
    purple_cap_wickets = wickets.iloc[0]['wickets']

    wickets = bowler_avgg[bowler_avgg['wickets'] >= 1].sort_values(by='avg', ascending=True)
    avg_bowler = wickets.iloc[0]['bowler']
    bowler_average = round(wickets.iloc[0]['avg'],2)

    wickets = bowler_avgg[bowler_avgg['wickets'] >= 1].sort_values(by='economy_rate', ascending=True)
    economy_bowler = wickets.iloc[0]['bowler']
    bowler_economy = round(wickets.iloc[0]['economy_rate'])
    return purple_cap_bowler,purple_cap_wickets,avg_bowler,bowler_average,economy_bowler,bowler_economy

def seasonal_expensive_inn():
    expensive_inn = seasonal_data.groupby(['ID', 'innings', 'bowler'])['total_run'].sum().sort_values(ascending=False).reset_index()
    return expensive_inn.iloc[0]['bowler'],expensive_inn.iloc[0]['total_run']