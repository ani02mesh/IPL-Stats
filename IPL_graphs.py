import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import IPL_query

def strikeRate_avg_gauge():
    trace1 = go.Indicator(mode="gauge+number", value=IPL_query.top_strike_rate, domain={'row': 1, 'column': 1}, title={'text': "{}".format(IPL_query.top_striker)})
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{'type': 'indicator'}]],
        subplot_titles=("Highest Strike-rate")
    )
    fig.add_trace(trace1, row=1, col=1)
    fig.update_layout(
        height=180,
        width=250,
        margin=dict(l=0, r=0, t=100, b=0),
    )
    fig.layout.annotations[0].update(x=0.140)
    return fig

def Matches_hosted_by_city():
    fig,ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax = sns.barplot(x=IPL_query.updated_match_per_city['count'], y=IPL_query.updated_match_per_city['City'], palette='rainbow')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors='white',labelsize=12)
    ax.set(xlim=(-5, 180))
    plt.title('\nTop 14 cities with most \n matches hosted\n', size=20, c='grey')
    plt.xlabel('\nNo. of Matches', c='grey', fontsize=15)
    plt.ylabel('Cities\n', c='grey', fontsize=15)
    ax.grid(axis='x', linewidth=1, alpha=0.2, color='#b2d6c7')
    for i in ['top', 'right','left','bottom']:
        ax.spines[i].set_visible(False)
        for i in ax.patches:
            ax.text(i.get_width()+3, i.get_y()+0.5,
                    '{:,d}'.format(int(i.get_width())), fontsize=13,
                    weight='bold', color='white')
    return fig

def boundary_count_per_stadium():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor("#0E1117")
    ax.bar(IPL_query.boundary_per_stadium['Stadium'], IPL_query.boundary_per_stadium['6s'], color='#8458B3', label='no. of 6s')
    ax.bar(IPL_query.boundary_per_stadium['Stadium'],IPL_query.boundary_per_stadium['4s'], bottom=IPL_query.boundary_per_stadium['6s'],
           color='#d0bdf4', label='no. of 4s')
    ax.tick_params(axis='x', rotation=90, colors='white',labelsize=16)
    ax.tick_params(axis='y', colors='white',labelsize=15)
    plt.title('\nTotal Boundaries conceded at Stadium\n', size=20, c='grey')
    plt.xlabel('\nStadiums', c='grey', fontsize=17)
    plt.ylabel('Boundary Count\n', c='grey', fontsize=17)
    ax.grid(axis='y', linewidth=1, alpha=0.2, color='#b2d6c7')
    ax.set_yticks(np.arange(-50, 5600, 400))
    for i in ['top', 'right']:
        ax.spines[i].set_visible(False)

    ax.spines['bottom'].set_bounds(0, 14)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    plt.legend(loc='best', prop={'size': 16},fontsize='x-large')
    return fig

def avg_run_stadium():
    fig, ax = plt.subplots(figsize=(12, 10), dpi=80)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor("#0E1117")
    ax = sns.barplot(x=IPL_query.runs_per_stadium['Venue'], y=IPL_query.runs_per_stadium['avg_run'], palette='winter_r', label='avg Run')
    ax1 = ax.twinx()
    ax1 = sns.lineplot(x=IPL_query.runs_per_stadium['Venue'], y=IPL_query.runs_per_stadium['avg_1st_inning_score'], marker='o',
                       color='white', alpha=1, label='Avg 1st Inning Score')
    ax2 = ax.twinx()
    ax2 = sns.lineplot(x=IPL_query.runs_per_stadium['Venue'], y=IPL_query.runs_per_stadium['avg_2nd_inning_score'], marker='o',
                       color='black', alpha=1, label='Avg 2nd Inning Score')
    ax.set(ylim=(140, 176))
    ax.set_yticks(np.arange(140, 176, 5))
    ax1.set(ylim=(140, 220))
    ax2.set(ylim=(138, 260))

    plt.title('\n Batting Friendly Venues \n', size=18, color='grey')
    ax.set_xlabel('\n Stadium', size=17, color='grey')
    ax.set_ylabel('\n Avg Runs \n', size=17, color='grey')

    for i in ['top', 'right']:
        ax.spines[i].set_visible(False)
        ax1.spines[i].set_visible(False)
        ax2.spines[i].set_visible(False)

    ax1.axes.yaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)

    ax.spines['bottom'].set_bounds(0, 14)
    ax1.spines['bottom'].set_bounds(0, 14)
    ax2.spines['bottom'].set_bounds(0, 14)
    ax.tick_params(axis='x', rotation=90, colors='white', labelsize=16)
    ax.tick_params(axis='y', colors='white', labelsize=15)
    ax.legend(loc='upper left', facecolor='#848884', fontsize='large')
    ax1.legend(loc='upper right', facecolor='#848884', fontsize='large')
    ax2.legend(loc='upper center', facecolor='#848884', fontsize='large')
    bar_heights = [bar.get_height() for bar in ax.containers[0]]
    line_data1 = ax1.lines[0].get_ydata()
    line_data2 = ax2.lines[0].get_ydata()
    for i, (bar, line_y1, line_y2) in enumerate(zip(bar_heights, line_data1, line_data2)):
        # Print the values over each individual bar
        ax.text(i, bar, f"{bar:.2f}", ha='center', va='bottom', fontsize=11, color='gray')
        ax1.text(i, line_y1 + 3, f"{line_y1:.2f}", ha='center', va='bottom', fontsize=10, color='white')
        ax2.text(i, line_y2 - 6, f"{line_y2:.2f}", ha='center', va='bottom', fontsize=10, color='black')

    return fig

def bowling_avg():
    trace1 = go.Indicator(mode="gauge+number", value=IPL_query.best_average, domain={'row': 1, 'column': 1}, title={'text': "{}".format(IPL_query.bowler_best_average)})
    trace2 = go.Indicator(mode="gauge+number", value=IPL_query.best_economy, domain={'row': 1, 'column': 2}, title={'text': "{}".format(IPL_query.bowler_best_economy)})
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Best Average-least 25 wickets", "Best Economy-least 25 wickets")
    )
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    fig.update_layout(
        height=280,
        width=550,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    fig.layout.annotations[0].update(x=0.190)
    fig.layout.annotations[1].update(x=0.725)
    return fig

def field_pie():
    dismissals = IPL_query.per_del[IPL_query.per_del['kind'].isin(['caught', 'caught and bowled', 'run out', 'bowled', 'stumped',
                                               'lbw', 'hit wicket', 'retired hurt', 'retired out',
                                               'obstructing the field'])].kind.value_counts().reset_index()

    fig = go.Figure(data=[go.Pie(labels=dismissals['kind'], values=dismissals['count'], hole=.4)])
    fig.update_layout(
        height=180,
        width=380,
        margin=dict(l=5, r=0, t=20, b=0),
        title={
            'text':'Dismissals Type'
        },
        font=dict(size=12, color='gray')
    )
    return fig

def trophy_count():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=80)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor("#0E1117")
    ax = sns.barplot(x=IPL_query.trophy['WinningTeam'], y=IPL_query.trophy['count'], palette='cool_r')
    ax.set(ylim=(0, 5))
    ax.axes.xaxis.set_visible(False)
    ax.tick_params(axis='y', colors='white', labelsize=15)

    plt.title('\n IPL Teams and Trophy Counts \n', size=18, color='grey')
    ax.set_ylabel('\n No. of IPL Trophy \n', size=17, color='grey')

    for i in ['top', 'right', 'bottom']:
        ax.spines[i].set_visible(False)

    label_count = 0
    for i in ax.patches:
        if i.get_height() <= 2:
            ax.text(i.get_x() + 0.23, i.get_height() + 0.1, IPL_query.labels[label_count], fontsize=18, color='gray',
                    rotation='vertical')
            ax.text(i.get_x() + 0.5, i.get_height() + 0.1, str(IPL_query.team_winning_season[label_count]), fontsize=18,
                    color='white', rotation='vertical')
            label_count += 1
        else:
            ax.text(i.get_x() + 0.23, i.get_height() - 3.8, IPL_query.labels[label_count], fontsize=18, color='black',
                    rotation='vertical')
            ax.text(i.get_x() + 0.5, i.get_height() - 3.8, str(IPL_query.team_winning_season[label_count]), fontsize=18,
                    color='white', rotation='vertical')
            label_count += 1

    return fig

def toss_result():
    fig = px.sunburst(IPL_query.toss_result, path=['TossDecision', 'variable'], values='value')
    fig.update_layout(title={
        'text':'Toss Result'},
        font=dict(size=12, color='gray'),
        margin=dict(l=0, r=80, t=25, b=0),
    )
    return fig

def toss_per_season():
    fig = px.bar(IPL_query.Toss_decision_per_season, x="Season", y="count", color='TossDecision',color_discrete_map={
                'Bat 1st' : '#EF553B',
                'Bat 2nd' : '#636EFA'
            },text_auto='.2s')
    fig.update_traces(textfont_size=12, textposition='inside')
    fig.update_yaxes(ticks='outside', tickcolor='white', tickfont=dict(family='Rockwell', color='white', size=14),
                     showline=True, showgrid=False, gridcolor='gray', title_text='Toss Decision Count',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(tick0=2008, dtick=1, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=12), showline=True, showgrid=False,
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', uniformtext_minsize=12,
                      uniformtext_mode='hide', margin=dict(l=50, r=20, t=50, b=50),
                      title={
                          'text': 'Toss Decision Per Season'},
                      font=dict(size=12, color='gray'),
                      )
    return fig

def orange_cap_per_season():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor("#0E1117")
    ax.set(ylim=(300, 1050))
    ax = sns.barplot(x=IPL_query.Most_run_per_season['Season'], y=IPL_query.Most_run_per_season['total_runs'],
                     palette='crest')
    ax.tick_params(axis='y', colors='white', labelsize=12)
    ax.tick_params(axis='x', colors='white', labelsize=8)

    ax.set_ylabel('\n Most Runs per Season \n', size=17, color='grey')
    ax.set_xlabel('\n Season \n', size=17, color='grey')

    for i in ['top', 'right', 'bottom']:
        ax.spines[i].set_visible(False)
    ax.spines['left'].set_color('white')

    labelcount = 0
    for i in ax.patches:
        ax.text(i.get_x() + 0.25, i.get_height() - 250, IPL_query.batter_list[labelcount], color='white',
                rotation='vertical',
                fontsize=14)
        ax.text(i.get_x() + 0.1, i.get_height() + 7, int(i.get_height()), color='white', fontsize=12)

        labelcount += 1
    return fig

def purple_cap_per_season():
    fig = px.line(IPL_query.wickets_per_season, x="Season", y="wickets", markers=True, hover_data=['bowler'])
    fig.update_traces(line_color='purple')
    fig.update_yaxes(tick0=18, dtick=2, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=True, showgrid=False,
                     gridcolor='gray', title_text='no. of Wickets',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(tick0=2008, dtick=1, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=True, showgrid=False,
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                      margin=dict(l=50, r=20, t=50, b=50),
                      title={
                          'text': 'Purple Cap Winner per Season'},
                      font=dict(size=18, color='gray'),
                      )
    return fig

def HS_per_season():
    fig = px.line(IPL_query.HS_per_season, x="Season", y="High_Score", text='batsman', markers=True)
    fig.update_traces(line_color='#79FB06', textposition="top center", textfont=dict(
        size=10,
        color="gray"
    ))
    fig.update_yaxes(dtick=8, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=True, showgrid=False,
                     gridcolor='gray', title_text='High Score',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(tick0=2008, dtick=1, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=True, showgrid=False,
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                      margin=dict(l=50, r=20, t=50, b=50),
                      title={
                          'text': 'Highest Score Per Season'},
                      font=dict(size=18, color='gray'),
                      )
    return fig

def records_per_season():
    discrete_sequence = ['#ffa600', '#ef5675']
    fig = px.bar(IPL_query.record_per_season, x="Season", y="count", color='Record', text_auto='.3s',
                 color_discrete_sequence=discrete_sequence)
    fig.update_traces(textfont=dict(size=10, color='white'))
    fig.update_yaxes(ticks='outside', tickcolor='white', tickfont=dict(family='Rockwell', color='white', size=14),
                     showline=True, showgrid=False, gridcolor='gray', title_text='no. of Wickets',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(tick0=2008, dtick=1, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=True, showgrid=False,
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', uniformtext_minsize=12,
                      uniformtext_mode='hide', margin=dict(l=50, r=20, t=50, b=50),
                      title={
                          'text': 'Century/half-Century per Season'},
                      font=dict(size=12, color='gray'),
                      )
    return fig

def partnership_per_season():
    fig = px.line(IPL_query.highest_partnership_per_season, x="Season", y="runs", markers=True, hover_data=['batsman_pair'])
    fig.update_traces(line_color='#00FAD5')
    fig.update_yaxes(dtick=8, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=True, showgrid=False,
                     gridcolor='gray', title_text='Partnership Runs',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(tick0=2008, dtick=1, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=True, showgrid=False,
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                      margin=dict(l=50, r=20, t=50, b=50),
                      title={
                          'text': 'Highest Partnership Per Season'},
                      font=dict(size=18, color='gray'),
                      )
    return fig

def team_stats_in_season():
    team_data = IPL_query.teams_win_in_season()
    discrete_sequence = ['#316879', '#7fe7dc']
    fig = px.bar(team_data, x="Teams", y="count",
                 color='Result', barmode='group', text_auto='.2d', color_discrete_sequence=discrete_sequence)
    fig.update_traces(textfont_size=12, textposition='inside')
    fig.update_yaxes(ticks='outside', tickcolor='white', tickfont=dict(family='Rockwell', color='white', size=14),
                     showline=True, showgrid=False, title_text='Matches Played',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(dtick=1, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=False, showgrid=False,
                     title_font=dict(size=18, family='Rockwell', color='white'))

    fig.update_layout(width=550, height=550, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', uniformtext_minsize=12,
                      uniformtext_mode='hide', margin=dict(l=80, r=20, t=50, b=50),
                      title={
                          'text': 'Win/Loss Count per Team'},
                      font=dict(size=12, color='gray'),
                      )
    return fig

def fielding_stats_in_season():
    discrete_sequence = ['#CADCFC', '#8AB6F9', '#00246B']
    data = IPL_query.loading_field_stats()
    fig = px.bar(data, x="Teams", y="dismissals",
                 color='kind', color_discrete_sequence=discrete_sequence)
    fig.update_traces(textfont_size=18, textposition='auto')
    fig.update_yaxes(ticks='outside', tickcolor='white', tickfont=dict(family='Rockwell', color='white', size=14),
                     showline=True, showgrid=False, title_text='No. of Dismissals',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(dtick=1, ticks='outside', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=False, showgrid=False,
                     title_font=dict(size=18, family='Rockwell', color='white'))

    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', uniformtext_minsize=14,
                      uniformtext_mode='hide', margin=dict(l=80, r=10, t=50, b=50),
                      title={
                          'text': 'Fielding Stats per Team'},
                      font=dict(size=12, color='gray'),
                      )
    return fig

def run_rate_in_season():
    discrete_sequence = ['#b6bcbf', '#6d6d6d', '#219ebc']
    data=IPL_query.loading_run_rate()
    fig = px.line(data, x="Teams", y="run_rate", color='overs', markers=True, text='run_rate',
                  color_discrete_sequence=discrete_sequence)
    fig.update_traces(textposition="top center", textfont=dict(size=10, color="gray"))
    fig.update_yaxes(tick0=7, dtick=1, ticks='outside', tickmode='linear', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=False, showgrid=True,
                     gridcolor='black', title_text='Run Rate',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(ticks='outside', tickcolor='white', tickfont=dict(family='Rockwell', color='white', size=14),
                     showline=True, showgrid=False, title_font=dict(size=18, family='Courier', color='white'))
    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                      margin=dict(l=50, r=0, t=40, b=50),
                      title={
                          'text': 'Run-Rate per Team '},
                      font=dict(size=14, color='gray'),
                      legend=dict(
                          # orientation="h",
                          bgcolor='#323232'))
    return fig

def strike_rate_in_season():
    data = IPL_query.loading_strike_rate()
    fig = px.line(data, x="Teams", y="strike_rate", color='over', markers=True, text='strike_rate')
    fig.update_traces(textposition="top center", textfont=dict(size=10, color="gray"))
    fig.update_yaxes(tick0=100, dtick=20, ticks='outside', tickmode='linear', tickcolor='white',
                     tickfont=dict(family='Rockwell', color='white', size=14), showline=False, showgrid=True,
                     gridcolor='grey', title_text='Strike Rate',
                     title_font=dict(size=18, family='Courier', color='white'))
    fig.update_xaxes(ticks='outside', tickcolor='white', tickfont=dict(family='Rockwell', color='white', size=14),
                     showline=True, showgrid=False, title_font=dict(size=18, family='Courier', color='white'))
    fig.update_layout(width=700, height=500, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                      margin=dict(l=50, r=20, t=50, b=50),
                      title={
                          'text': 'Individual Teams Strike Rate'},
                      font=dict(size=14, color='gray'),
                      )
    return fig