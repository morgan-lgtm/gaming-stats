import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import boto3
import os  # Import os for environment variables

# Set page configuration
st.set_page_config(
    page_title="NHL Gaming Stats Dashboard V3",
    page_icon="🏒",
    layout="wide"
)

@st.cache_data
def load_data():
    # Determine the environment
    is_aws = os.getenv('IS_AWS', 'false').lower() == 'true'

    if is_aws:
        # Load data from S3
        s3 = boto3.client('s3')
        bucket_name = os.getenv('S3_BUCKET')  # S3 bucket name from env variable
        file_name = os.getenv('CSV_KEY')      # CSV file key from env variable
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=file_name)
            df = pd.read_csv(obj['Body'])
        except Exception as e:
            st.error(f"Error loading data from S3: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    else:
        # Load data locally
        try:
            df = pd.read_csv('data/nhl_stats.csv')
        except FileNotFoundError:
            st.error("Local CSV file not found. Please ensure 'data/nhl_stats.csv' exists.")
            return pd.DataFrame()  # Return empty DataFrame if file not found

    # Data processing
    df['Date'] = pd.to_datetime(df['Date'])
    df['SequentialIndex'] = range(len(df))
    df = df.sort_values(['Date', 'SequentialIndex']).reset_index(drop=True)
    df['Goal Differential'] = df['Us Goals'] - df['Opponent Goals']
    df['GameCount'] = df.groupby('Date').cumcount()
    df['DateTime'] = df['Date'] + pd.to_timedelta(df['GameCount'] * 5, unit='m')

    # Convert 'First Game of Night' and 'Last Game of Night' to integer if they are not
    df['First Game of Night'] = df['First Game of Night'].astype(int)
    df['Last Game of Night'] = df['Last Game of Night'].astype(int)

    # Handle Missing Values in new columns
    new_columns = [
        'Opponent Total Shots',
        'Us Total Shots',
        'Opponent Hits',
        'Us Hits',
        'Opponent TOA',
        'Us TOA',
        'Opponent Passing Rate',
        'Us Passing Rate',
        'Opponent Faceoffs Won',
        'Us Faceoffs Won',
        'Opponent Penalty Minutes',
        'Us Penalty Minutes',
        'Opponent Power Play Minutes',
        'Us Power Play Minutes',
        'Opponent Shorthanded Goals',
        'Us Shorthanded Goals'
    ]

    # Function to parse 'mm:ss' format to minutes as float
    def parse_time_to_minutes(time_str):
        try:
            if pd.isnull(time_str):
                return np.nan
            time_str = str(time_str)
            minutes, seconds = map(int, time_str.split('.'))
            return minutes + seconds / 60
        except:
            return np.nan

    # Columns that are in 'mm:ss' format
    time_columns = [
        'Opponent TOA',
        'Us TOA',
        'Opponent Power Play Minutes',
        'Us Power Play Minutes'
    ]

    # Replace 'Missing' with NaN and parse columns
    for col in new_columns:
        df[col] = df[col].replace('Missing', np.nan)
        if col in time_columns:
            df[col] = df[col].apply(parse_time_to_minutes)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle Missing Values in essential columns
    df_cleaned = df.dropna(subset=['Goal Differential', 'Us Goals', 'Opponent Goals', 'Win / Loss'])

    return df_cleaned

# Load the data
df = load_data()

# If DataFrame is empty, stop execution
if df.empty:
    st.stop()

# Main Dashboard with Tabs
st.title("🏒 NHL 25 Gaming Stats")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images6.alphacoders.com/982/thumb-1920-982416.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs([
    "Dashboard",
    "Team Analysis",
    "Player Analysis",
    "Opponent Analysis",
    "Game Breakdown",
    "AI Insights",
    "Predictions"
])

# Dashboard Tab
with tabs[0]:
    st.header("Overview")

    # Wins-Losses-Forfeits (Quits) Distribution
    df['Result Category'] = df['Win / Loss'].map({
        'Win': 'Win',
        'Loss': 'Loss',
        'Quit': 'Forfeit'  # Treat quits as forfeits
    })

    # Display Win-Loss-Forfeit Record
    wins = (df['Result Category'] == 'Win').sum()
    losses = (df['Result Category'] == 'Loss').sum()
    forfeits = (df['Result Category'] == 'Forfeit').sum()

    st.subheader(f"Record: {wins}-{losses}-{forfeits}")

    # Calculate MVP Based on Last 5 Games
    last_5_games = df.tail(5)
    players = ['Nolan', 'Andrew', 'Morgan']

    # Calculate total goals for each player over the last 5 games
    last_5_goals = {player: last_5_games[f'{player} Goal'].sum() for player in players}
    mvp_player = max(last_5_goals, key=last_5_goals.get)
    mvp_goals = last_5_goals[mvp_player]

    # Enhance MVP section with visual and layout improvements, including the team logo
    st.markdown("""
        <div style="background-color:#f9f9f9;padding:5px 10px;border-radius:10px;margin-bottom:10px;text-align:center;">
            <img src="https://1000logos.net/wp-content/uploads/2018/06/Nashville-Predators-Logo.png" alt="Team Logo" style="width:50px;height:auto;margin-bottom:5px;">
            <h2 style="color:#041E42;font-size:20px;margin:5px 0;">🏅 Current MVP (Last 5 Games)</h2>
            <h1 style="color:#228B22;font-size:28px;margin:5px 0;">{}</h1>
            <p style="color:#041E42;font-size:16px;margin:5px 0;">Scored <strong>{}</strong> goals in the last 5 games!</p>
        </div>
        """.format(mvp_player, mvp_goals),
        unsafe_allow_html=True
    )

    # Rest of the Overview metrics
    cols = st.columns(4)  # Reduced to 4 columns to accommodate the pie chart

    # Calculate Current Streak (Consecutive Wins or Losses)
    def calculate_streak(df):
        streak = 0
        streak_type = None

        for diff in reversed(df[df['Win / Loss']!='Quit']['Goal Differential']):
            if diff > 0:
                if streak_type is None or streak_type == 'win':
                    streak_type = 'win'
                    streak += 1
                else:
                    break
            elif diff < 0:
                if streak_type is None or streak_type == 'loss':
                    streak_type = 'loss'
                    streak += 1
                else:
                    break
            else:
                break
        return streak, streak_type

    streak, streak_type = calculate_streak(df)
    streak_label = f"{streak} {'wins' if streak_type == 'win' else 'losses' if streak_type == 'loss' else 'games'}"

    metrics = [
        ("Total Games", len(df)),
        ("Avg Goal Differential", f"{df['Goal Differential'].mean():.2f}"),
        ("Current Streak", streak_label),
        ("Win Rate", f"{(df['Goal Differential'] > 0).mean():.2%}")
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    # Pie Chart for Wins, Losses, and Forfeits
    performance_counts = df['Result Category'].value_counts().reset_index()
    performance_counts.columns = ['Result', 'Count']

    fig_pie = px.pie(
        performance_counts,
        names='Result',
        values='Count',
        title='Wins, Losses, and Forfeits Distribution',
        color='Result',
        color_discrete_map={
            'Win': 'green',
            'Loss': 'red',
            'Forfeit': 'gray'
        },
        hole=0.4  # Creates a donut chart; set to 0 for a full pie chart
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Goal Differential Chart Using SequentialIndex
    plot_df = df.tail(10)  # Last 10 games
    plot_df['Color'] = plot_df['Goal Differential'].apply(
        lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Forfeit')
    )

    fig_goal_diff = px.line(
        plot_df,
        x='SequentialIndex',
        y='Goal Differential',
        title="Last 10 Games Goal Differential",
        markers=True,
        hover_data={
            'SequentialIndex': False,
            'DateTime': True,
            'Goal Differential': True
        },
        color='Color',
        color_discrete_map={'Win': 'blue', 'Loss': 'red', 'Forfeit': 'gray'}
    )

    fig_goal_diff.update_xaxes(
        title="Date and Time",
        tickmode='array',
        tickvals=plot_df['SequentialIndex'],
        ticktext=plot_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M'),
        tickangle=45
    )
    fig_goal_diff.update_layout(
        yaxis_title="Goal Differential",
        hovermode='x unified'
    )
    st.plotly_chart(fig_goal_diff, use_container_width=True)

    # Pie Chart for Faceoff Win Percentage
    st.subheader("Faceoff Win Percentage")
    total_faceoffs = df['Us Faceoffs Won'].sum() + df['Opponent Faceoffs Won'].sum()
    if total_faceoffs > 0:
        us_faceoffs = df['Us Faceoffs Won'].sum()
        opponent_faceoffs = df['Opponent Faceoffs Won'].sum()
        faceoff_data = pd.DataFrame({
            'Team': ['Us', 'Opponent'],
            'Faceoffs Won': [us_faceoffs, opponent_faceoffs]
        })
        fig_faceoff_pie = px.pie(
            faceoff_data,
            names='Team',
            values='Faceoffs Won',
            title='Faceoff Win Distribution',
            color='Team',
            color_discrete_map={'Us': 'blue', 'Opponent': 'red'},
            hole=0.4  # Donut chart
        )
        st.plotly_chart(fig_faceoff_pie, use_container_width=True)
    else:
        st.info("No faceoff data available.")

# Team Analysis Tab (Tab 2)
with tabs[1]:
    st.header("Team Performance Analysis")

    # Existing Team Stats
    team_stats = {
        'Win Rate': (df['Goal Differential'] > 0).mean(),
        'Avg Goals Scored': df['Us Goals'].mean(),
        'Avg Goals Conceded': df['Opponent Goals'].mean(),
        'Goal Differential': df['Goal Differential'].mean(),
        'Clean Sheets': (df['Opponent Goals'] == 0).mean(),
        'Comeback Rate': (
            (df['Opponent Goals'] > df['Us Goals']) & (df['Goal Differential'] > 0)
        ).mean()
    }

    # New Stats Incorporating the New Data Points
    additional_stats = {
        'Avg Us Total Shots': df['Us Total Shots'].mean(),
        'Avg Opponent Total Shots': df['Opponent Total Shots'].mean(),
        'Avg Us Hits': df['Us Hits'].mean(),
        'Avg Opponent Hits': df['Opponent Hits'].mean(),
        'Avg Us TOA (min)': df['Us TOA'].mean(),
        'Avg Opponent TOA (min)': df['Opponent TOA'].mean(),
        'Avg Us Passing Rate (%)': df['Us Passing Rate'].mean(),
        'Avg Opponent Passing Rate (%)': df['Opponent Passing Rate'].mean(),
        'Avg Us Faceoffs Won': df['Us Faceoffs Won'].mean(),
        'Avg Opponent Faceoffs Won': df['Opponent Faceoffs Won'].mean(),
        'Avg Us Penalty Minutes': df['Us Penalty Minutes'].mean(),
        'Avg Opponent Penalty Minutes': df['Opponent Penalty Minutes'].mean(),
        'Avg Us Power Play Minutes (min)': df['Us Power Play Minutes'].mean(),
        'Avg Opponent Power Play Minutes (min)': df['Opponent Power Play Minutes'].mean(),
        'Avg Us Shorthanded Goals': df['Us Shorthanded Goals'].mean(),
        'Avg Opponent Shorthanded Goals': df['Opponent Shorthanded Goals'].mean()
    }

    # Combine the stats
    all_team_stats = {**team_stats, **additional_stats}

    # Display the stats in a DataFrame
    team_stats_df = pd.DataFrame.from_dict(all_team_stats, orient='index', columns=['Value'])
    team_stats_df.reset_index(inplace=True)
    team_stats_df.rename(columns={'index': 'Metric'}, inplace=True)

    # Display the metrics in a table
    st.table(team_stats_df.style.format({'Value': '{:.2f}'}))

    # Visualization of New Stats
    st.subheader("Additional Team Performance Metrics")

    # Create subplots for visualizing stats
    fig = make_subplots(rows=4, cols=2, subplot_titles=[
        'Total Shots', 'Hits', 'Time on Attack (min)', 'Passing Rate (%)',
        'Faceoffs Won', 'Penalty Minutes', 'Power Play Minutes (min)', 'Shorthanded Goals'
    ])

    # Total Shots
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us Total Shots'].mean(), df['Opponent Total Shots'].mean()],
            name='Total Shots',
            marker_color=['blue', 'red']
        ),
        row=1, col=1
    )

    # Hits
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us Hits'].mean(), df['Opponent Hits'].mean()],
            name='Hits',
            marker_color=['blue', 'red']
        ),
        row=1, col=2
    )

    # Time on Attack
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us TOA'].mean(), df['Opponent TOA'].mean()],
            name='Time on Attack (min)',
            marker_color=['blue', 'red']
        ),
        row=2, col=1
    )

    # Passing Rate
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us Passing Rate'].mean(), df['Opponent Passing Rate'].mean()],
            name='Passing Rate (%)',
            marker_color=['blue', 'red']
        ),
        row=2, col=2
    )

    # Faceoffs Won
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us Faceoffs Won'].mean(), df['Opponent Faceoffs Won'].mean()],
            name='Faceoffs Won',
            marker_color=['blue', 'red']
        ),
        row=3, col=1
    )

    # Penalty Minutes
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us Penalty Minutes'].mean(), df['Opponent Penalty Minutes'].mean()],
            name='Penalty Minutes',
            marker_color=['blue', 'red']
        ),
        row=3, col=2
    )

    # Power Play Minutes
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us Power Play Minutes'].mean(), df['Opponent Power Play Minutes'].mean()],
            name='Power Play Minutes (min)',
            marker_color=['blue', 'red']
        ),
        row=4, col=1
    )

    # Shorthanded Goals
    fig.add_trace(
        go.Bar(
            x=['Us', 'Opponent'],
            y=[df['Us Shorthanded Goals'].mean(), df['Opponent Shorthanded Goals'].mean()],
            name='Shorthanded Goals',
            marker_color=['blue', 'red']
        ),
        row=4, col=2
    )

    fig.update_layout(height=1200, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Compare the team's performance with opponents across various metrics.
    - Identify areas where the team excels or needs improvement.
    """)

    # Analyze Correlation between New Metrics and Winning
    st.subheader("Correlation between New Metrics and Goal Differential")

    correlation_metrics = [
        'Us Total Shots',
        'Us Hits',
        'Us TOA',
        'Us Passing Rate',
        'Us Faceoffs Won',
        'Us Penalty Minutes',
        'Us Power Play Minutes',
        'Us Shorthanded Goals'
    ]

    correlations = {}
    for metric in correlation_metrics:
        correlations[metric] = df['Goal Differential'].corr(df[metric])

    correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with Goal Differential'])
    correlation_df.reset_index(inplace=True)
    correlation_df.rename(columns={'index': 'Metric'}, inplace=True)

    fig_corr = px.bar(
        correlation_df,
        x='Metric',
        y='Correlation with Goal Differential',
        title='Correlation between Team Metrics and Goal Differential',
        color='Correlation with Goal Differential',
        color_continuous_scale='RdBu',
        range_color=[-1, 1]
    )
    fig_corr.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_corr, use_container_width=True)