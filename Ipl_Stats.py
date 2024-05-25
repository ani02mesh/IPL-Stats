import streamlit as st
from streamlit_option_menu import option_menu
st.set_page_config(layout='wide')
#from st_pages import show_pages, hide_pages, Page
import pandas as pd
from streamlit_option_menu import option_menu
import IPL_query
import IPL_graphs
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
per_del,match = IPL_query.Datacleaning()

st.title ("IPL Stats -> 2008-2022🏏")

cols = st.columns((1,1),gap='small')
with cols[0]:
    show_dataframe1 = st.toggle('Show DataFrame1')
with cols[1]:
    show_dataframe2 = st.toggle('show Dataframe2')

dataholder1 = st.empty()
dataholder2 = st.empty()
if show_dataframe1:
    with dataholder1.container():
        st.text('IPL ball by ball record')
        st.dataframe(per_del,height=200)
if show_dataframe2:
    with dataholder2.container():
        st.text('IPL Matches')
        st.dataframe(match,height=200)


cols = st.columns((6,5),gap='medium')
with cols[0]:
    st.header("Batsman Stats")
    two_col = st.columns((1,1),gap='small')
    with two_col[0]:
        st.metric("Most Runs", IPL_query.most_run,IPL_query.batsman_most_run)
    with two_col[1]:
        st.metric("Highest Score", IPL_query.HS_run,IPL_query.batsman_HS)
    two_col = st.columns((1,1),gap='small')
    with two_col[0]:
        st.metric("Most 4s", IPL_query.batsman_4count, IPL_query.batsman_4)
    with two_col[1]:
        st.metric("Most 6s", IPL_query.batsman_6count,IPL_query.batsman_6)

    st.metric("Best Strike-Rate", IPL_query.top_strike_rate, IPL_query.top_striker)

with cols[1]:
    option = st.selectbox(
        '',
        ('Matches Hosted by City', 'Stadium with most boundary','Average runs in stadium'),
        label_visibility='collapsed'
    )
    holder = st.empty()
    if option =='Matches Hosted by City':
        grph = IPL_graphs.Matches_hosted_by_city()
        holder.pyplot(grph)
    elif option == 'Stadium with most boundary':
        grph = IPL_graphs.boundary_count_per_stadium()
        holder.pyplot(grph)
    elif option == 'Average runs in stadium':
        grph = IPL_graphs.avg_run_stadium()
        holder.pyplot(grph)

cols = st.columns((6,5),gap='medium')
with cols[0]:
    st.header("Bowler Stats")
    two_col = st.columns((1,1),gap='small')
    with two_col[0]:
        st.metric("Most Wickets", IPL_query.bowler_Mwicket_count,IPL_query.bowler_most_wickets)
    with two_col[1]:
        st.metric("Most Dot Balls", IPL_query.most_dot_ball_count,IPL_query.bowler_most_dot_ball)
    st.metric("Best Bowling Figure", IPL_query.best_figure,IPL_query.best_fig_bowler)

    strikeRate_avg = IPL_graphs.bowling_avg()
    st.plotly_chart(strikeRate_avg)

with cols[1]:
    st.header("Fielders Stats")
    st.dataframe(IPL_query.fielders, height=200)
    fielders_rec = IPL_graphs.field_pie()
    st.plotly_chart(fielders_rec)

cols = st.columns((6,5),gap='medium')
with cols[0]:
    grph = IPL_graphs.trophy_count()
    st.pyplot(grph)
with cols[1]:
    st.subheader('Win % of each Team')
    st.dataframe(IPL_query.team_record, height=350)

cols = st.columns((5,6),gap='medium')
with cols[0]:
    grph = IPL_graphs.toss_result()
    st.plotly_chart(grph)

with cols[1]:
    option1 = st.selectbox(
        '',
        ('Toss Decision', 'Orange Cap','Purple Cap','Highest Individual Score','Century/Half-Century count',
         'Highest Partnership'),
        label_visibility='collapsed'
    )
    holder1 = st.empty()
    if option1 == 'Toss Decision':
        grph = IPL_graphs.toss_per_season()
        holder1.plotly_chart(grph)
    elif option1 == 'Orange Cap':
        grph = IPL_graphs.orange_cap_per_season()
        holder1.pyplot(grph)
    elif option1 == 'Purple Cap':
        grph = IPL_graphs.purple_cap_per_season()
        holder1.plotly_chart(grph)
    elif option1 == 'Highest Individual Score':
        grph = IPL_graphs.HS_per_season()
        holder1.plotly_chart(grph)
    elif option1 == 'Century/Half-Century count':
        grph = IPL_graphs.records_per_season()
        holder1.plotly_chart(grph)
    elif option1 == 'Highest Partnership':
        grph = IPL_graphs.partnership_per_season()
        holder1.plotly_chart(grph)

cols = st.columns((1,2,2),gap='small')
with cols[0]:
    season = st.selectbox(
        '',IPL_query.seasons,
        label_visibility='collapsed',
        on_change=IPL_query.new_season,
        key='seasonn'
    )
with cols[1]:
    st.subheader('Winner :  ' + IPL_query.winner()+' 🏆')
with cols[2]:
    st.subheader('Runner up :  ' + IPL_query.runner_up()+' 🥈')
cols = st.columns((1,2.2,2.2),gap='medium')

cols = st.columns((3,4),gap='large')
with cols[0]:
    grph = IPL_graphs.team_stats_in_season()
    st.plotly_chart(grph)
with cols[1]:
    selection = st.selectbox(
        '', ('Fielding Performance','run-rate per team','StrikeRate per team'),
        label_visibility='collapsed',
    )
    holder2 = st.empty()
    if selection == 'Fielding Performance':
        grph = IPL_graphs.fielding_stats_in_season()
        holder2.plotly_chart(grph)
    elif selection == 'run-rate per team':
        grph = IPL_graphs.run_rate_in_season()
        holder2.plotly_chart(grph)
    elif selection == 'StrikeRate per team':
        grph = IPL_graphs.strike_rate_in_season()
        holder2.plotly_chart(grph)

cols = st.columns((1,1,1,1),gap='medium')
with cols[0]:
    batsman,mostRuns = IPL_query.seasonal_orange()
    st.metric("Orange Cap Winner", mostRuns, batsman)
with cols[1]:
    batsman,HS_run = IPL_query.seasonal_HS()
    st.metric("Highest Score", HS_run, batsman)
with cols[2]:
    batsman,fours = IPL_query.seasonal_boundary()
    st.metric("Batsman with Most 4s", fours, batsman)
with cols[3]:
    batsman,sixes = IPL_query.seasonal_sixes()
    st.metric("Batsman with Most 6s", sixes, batsman)

cols = st.columns((1,1,1,1),gap='medium')
with cols[0]:
    batsman,SR = IPL_query.seasonal_SR()
    st.metric("Highest Strike-Rate", SR, batsman)
with cols[1]:
    bowler,wickets,avg_bowler,bowler_avg,economy_bowler,bowler_economy = IPL_query.seasonal_bowling_stats()
    st.metric("Purple Cap Winner", wickets, bowler)
with cols[2]:
    bowler,dot = IPL_query.seasonal_dotBalls()
    st.metric("Most Dot Balls", dot, bowler)
with cols[3]:
    bowler,fig = IPL_query.seasonal_best_fig()
    st.metric("Best Bowling figure", fig, bowler)

cols = st.columns((1,1,1),gap='medium')
with cols[0]:
    st.metric("Best Bowling Average", bowler_avg, avg_bowler)
with cols[1]:
    st.metric("Best Bowling Economy", bowler_economy, economy_bowler)
with cols[2]:
    bowler,runs = IPL_query.seasonal_expensive_inn()
    st.metric("Most runs in a inning ", runs, bowler)