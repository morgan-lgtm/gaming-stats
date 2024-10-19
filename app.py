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
    page_title="NHL Gaming Stats Dashboard",
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
    cols = st.columns(4)  # Adjusted to 4 columns

    # Calculate Current Streak (Consecutive Wins or Losses)
    def calculate_streak(df):
        streak = 0
        streak_type = None

        # Reverse the Series using .iloc[::-1]
        goal_diff_series = df[df['Win / Loss'] != 'Quit']['Goal Differential'].iloc[::-1]

        for diff in goal_diff_series:
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
    st.subheader("Game Results Distribution")
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

    # Add back the original plot for Goal Differential based on Game Order
    st.subheader("Goal Differential Based on Game Order")
    # Create a new column to categorize games
    def game_order(row):
        if row['First Game of Night'] == 1:
            return 'First Game'
        elif row['Last Game of Night'] == 1:
            return 'Last Game'
        else:
            return 'Middle Game'

    df['Game Order'] = df.apply(game_order, axis=1)

    # Calculate average goal differential for each game order category
    game_order_stats = df.groupby('Game Order')['Goal Differential'].mean().reset_index()

    # Plotting
    fig_game_order = px.bar(
        game_order_stats,
        x='Game Order',
        y='Goal Differential',
        title='Average Goal Differential by Game Order',
        color='Game Order',
        color_discrete_map={'First Game': 'green', 'Middle Game': 'blue', 'Last Game': 'orange'}
    )
    fig_game_order.update_layout(showlegend=False)
    st.plotly_chart(fig_game_order, use_container_width=True)

# Player Analysis Tab (Tab 3)
with tabs[2]:
    st.header("Player Performance Analysis")

    # Synergy Metrics Heatmap
    from itertools import product

    players = ['Nolan', 'Andrew', 'Morgan']
    synergy_matrix = pd.DataFrame(index=players, columns=players, dtype=float)

    for player1, player2 in product(players, repeat=2):
        # Condition where both players score
        condition = (df[f'{player1} Goal'] > 0) & (df[f'{player2} Goal'] > 0)
        if condition.any():
            avg_goal_diff = df.loc[condition, 'Goal Differential'].mean()
            synergy_matrix.loc[player1, player2] = avg_goal_diff
        else:
            synergy_matrix.loc[player1, player2] = np.nan  # Or set to 0

    st.subheader("Synergy Metrics Heatmap")
    fig_synergy = px.imshow(
        synergy_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Player Synergy Heatmap (Avg Goal Differential when both players score)"
    )
    st.plotly_chart(fig_synergy, use_container_width=True)

    # Individual Player Statistics
    st.subheader("Individual Player Statistics")

    player_stats = pd.DataFrame({
        'Player': players,
        'Total Goals': [df[f'{player} Goal'].sum() for player in players],
        'Average Goals per Game': [df[f'{player} Goal'].mean() for player in players],
        'Highest Goals in a Single Game': [df[f'{player} Goal'].max() for player in players],
        # Assuming equal contribution for hits and penalties
        #'Average Hits per Game': df['Us Hits'].mean() / len(players),
        #'Average Penalty Minutes per Game': df['Us Penalty Minutes'].mean() / len(players)
    })

    # Display individual player stats in a table
    st.table(player_stats.style.format({
        'Average Goals per Game': "{:.2f}",
        'Total Goals': "{:.0f}",
        'Highest Goals in a Single Game': "{:.0f}",
       # 'Average Hits per Game': "{:.2f}",
       # 'Average Penalty Minutes per Game': "{:.2f}"
    }))

    # Visualization of Individual Player Statistics
    st.subheader("Visual Representation of Player Statistics")

    # Total Goals Bar Chart
    fig_total_goals = px.bar(
        player_stats,
        x='Player',
        y='Total Goals',
        title='Total Goals per Player',
        text='Total Goals',
        color='Player',
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    fig_total_goals.update_traces(textposition='outside')
    fig_total_goals.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_total_goals, use_container_width=True)

    # # Average Hits per Game
    # fig_hits = px.bar(
    #     player_stats,
    #     x='Player',
    #     y='Average Hits per Game',
    #     title='Average Hits per Game per Player',
    #     text='Average Hits per Game',
    #     color='Player',
    #     color_discrete_sequence=px.colors.qualitative.Dark2
    # )
    # fig_hits.update_traces(textposition='outside')
    # fig_hits.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    # st.plotly_chart(fig_hits, use_container_width=True)

# Opponent Analysis Tab (Tab 4)
with tabs[3]:
    st.header("Opponent Analysis")
    avg_performance = df.groupby('# Opponent Players').agg({
        'Goal Differential': 'mean',
        'Us Goals': 'mean',
        'Opponent Goals': 'mean',
        'Opponent Hits': 'mean',
        'Opponent Penalty Minutes': 'mean'
    }).reset_index()

    # Ensure '# Opponent Players' is integer
    avg_performance['# Opponent Players'] = avg_performance['# Opponent Players'].astype(int)

    if not avg_performance.empty:
        fig = make_subplots(rows=2, cols=1, subplot_titles=(
            "Goal Differential and Goals Scored/Conceded",
            "Opponent Hits and Penalty Minutes"
        ), shared_xaxes=True)

        # Goal Differential and Goals
        fig.add_trace(
            go.Bar(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Goal Differential'],
                name="Avg Goal Differential",
                marker_color='indianred'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Us Goals'],
                name="Avg Goals Scored",
                mode="lines+markers",
                marker=dict(color='green')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Opponent Goals'],
                name="Avg Goals Conceded",
                mode="lines+markers",
                marker=dict(color='blue')
            ),
            row=1, col=1
        )

        # Opponent Hits and Penalty Minutes
        fig.add_trace(
            go.Bar(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Opponent Hits'],
                name="Avg Opponent Hits",
                marker_color='blue'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Opponent Penalty Minutes'],
                name="Avg Opponent Penalty Minutes",
                marker_color='orange'
            ),
            row=2, col=1
        )

        fig.update_layout(title="Opponent Performance Metrics", height=800)
        fig.update_xaxes(title_text="Number of Opponent Players", tickmode='linear', tick0=1, dtick=1)
        fig.update_yaxes(title_text="Values")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for opponent analysis.")

# Game Breakdown Tab (Tab 5)
with tabs[4]:
    st.header("Game-by-Game Breakdown")

    # Scatter Plot Using SequentialIndex with Enhanced Coloring
    scatter = px.scatter(
        df,
        x='SequentialIndex',
        y='Goal Differential',
        color=df['Goal Differential'].apply(
            lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Forfeit')
        ),
        color_discrete_map={'Win': 'blue', 'Loss': 'red', 'Forfeit': 'gray'},
        hover_data={
            'SequentialIndex': False,
            'DateTime': True,
            'Us Goals': True,
            'Opponent Goals': True,
            'Goal Differential': True,
            'Us Total Shots': True,
            'Opponent Total Shots': True,
            'Us Hits': True,
            'Opponent Hits': True,
            'Us TOA': True,
            'Opponent TOA': True,
            'Us Passing Rate': True,
            'Opponent Passing Rate': True,
            'Us Faceoffs Won': True,
            'Opponent Faceoffs Won': True,
            'Us Penalty Minutes': True,
            'Opponent Penalty Minutes': True,
            'Us Power Play Minutes': True,
            'Opponent Power Play Minutes': True,
            'Us Shorthanded Goals': True,
            'Opponent Shorthanded Goals': True
        },
        title="Goal Differential Over Time"
    )

    scatter.update_xaxes(
        title="Game Number",
        tickmode='array',
        tickvals=df['SequentialIndex'][::5],  # Show every 5th game for readability
        ticktext=df['DateTime'][::5].dt.strftime('%Y-%m-%d'),
        tickangle=45
    )
    scatter.update_layout(
        yaxis_title="Goal Differential",
        hovermode='closest'
    )
    st.plotly_chart(scatter, use_container_width=True)

    selected_game_idx = st.selectbox(
        "Select a game for detailed view",
        df.index,
        format_func=lambda x: f"Game {df.loc[x, 'SequentialIndex']} on {df.loc[x, 'Date'].strftime('%Y-%m-%d')}"
    )
    game_data = df.loc[selected_game_idx]
    st.subheader(f"Game Details: {game_data['Date'].strftime('%Y-%m-%d %H:%M')}")
    detail_cols = st.columns(3)
    details = [
        ("Us Goals", game_data['Us Goals']),
        ("Opponent Goals", game_data['Opponent Goals']),
        ("Goal Differential", game_data['Goal Differential'])
    ]
    for col, (label, value) in zip(detail_cols, details):
        col.metric(label, value)

    # Additional Game Stats
    st.subheader("Additional Game Statistics")
    additional_stats = {
        'Us Total Shots': game_data['Us Total Shots'],
        'Opponent Total Shots': game_data['Opponent Total Shots'],
        'Us Hits': game_data['Us Hits'],
        'Opponent Hits': game_data['Opponent Hits'],
        'Us TOA (min)': game_data['Us TOA'],
        'Opponent TOA (min)': game_data['Opponent TOA'],
        'Us Passing Rate (%)': game_data['Us Passing Rate'],
        'Opponent Passing Rate (%)': game_data['Opponent Passing Rate'],
        'Us Faceoffs Won': game_data['Us Faceoffs Won'],
        'Opponent Faceoffs Won': game_data['Opponent Faceoffs Won'],
        'Us Penalty Minutes': game_data['Us Penalty Minutes'],
        'Opponent Penalty Minutes': game_data['Opponent Penalty Minutes'],
        'Us Power Play Minutes (min)': game_data['Us Power Play Minutes'],
        'Opponent Power Play Minutes (min)': game_data['Opponent Power Play Minutes'],
        'Us Shorthanded Goals': game_data['Us Shorthanded Goals'],
        'Opponent Shorthanded Goals': game_data['Opponent Shorthanded Goals']
    }
    game_stats_df = pd.DataFrame.from_dict(additional_stats, orient='index', columns=['Value'])
    game_stats_df.reset_index(inplace=True)
    game_stats_df.rename(columns={'index': 'Metric'}, inplace=True)
    st.table(game_stats_df)

# AI Insights Tab
with tabs[5]:
    st.header("AI-Generated Insights")

    # Enhanced AI Insights incorporating new data points
    avg_goal_diff = df['Goal Differential'].mean()
    top_opponent_size = df.groupby('# Opponent Players')['Goal Differential'].mean().idxmax()
    player_impact = {
        player: df[df[f'{player} Goal'] > 0]['Goal Differential'].mean()
        for player in players
    }
    top_player = max(player_impact, key=player_impact.get)

    # Analyze performance based on new stats
    highest_passing_rate = df['Us Passing Rate'].mean()
    highest_hits = df['Us Hits'].mean()
    least_penalty_minutes = df['Us Penalty Minutes'].mean()

    # Faceoff Win Percentage
    total_faceoffs = df['Us Faceoffs Won'].sum() + df['Opponent Faceoffs Won'].sum()
    if total_faceoffs > 0:
        faceoff_win_percentage = df['Us Faceoffs Won'].sum() / total_faceoffs * 100
    else:
        faceoff_win_percentage = 0.0

    insights = [
        f"**Average Goal Differential**: {avg_goal_diff:.2f}. {'Focus on improving defensive strategies.' if avg_goal_diff < 0 else 'Keep up the good offensive work!'}",
        f"**Best Performance Against Teams with {top_opponent_size} Players**. Consider strategies that work well in these matchups.",
        f"**Top Performer**: {top_player} has the highest impact on goal differential when scoring. Creating more opportunities for them could improve overall performance.",
        f"**Passing Rate**: An average passing rate of {highest_passing_rate:.2f}%. {'Excellent puck movement is contributing to your success.' if highest_passing_rate > 70 else 'Improving passing accuracy could enhance your gameplay.'}",
        f"**Physical Play**: Averaging {highest_hits:.2f} hits per game. {'Aggressive play is paying off.' if highest_hits > df['Opponent Hits'].mean() else 'Consider increasing physical play to disrupt opponents.'}",
        f"**Discipline**: Averaging {least_penalty_minutes:.2f} penalty minutes per game. {'Good discipline is keeping you out of the box.' if least_penalty_minutes < df['Opponent Penalty Minutes'].mean() else 'Reducing penalties could prevent giving opponents power play opportunities.'}",
        f"**Faceoff Win Percentage**: {faceoff_win_percentage:.2f}%. {'Dominating faceoffs helps control the game.' if faceoff_win_percentage > 50 else 'Improving faceoff wins could provide more puck possession.'}"
    ]

    for insight in insights:
        st.info(insight)

# Predictions Tab (Tab 6)
with tabs[6]:
    st.header("Predictions and What-If Scenarios")

    # Ensure the DataFrame has these columns
    feature_cols = [
        'Us # Players',
        '# Opponent Players',
        'Nolan Goal',
        'Andrew Goal',
        'Morgan Goal',
        'First Game of Night',
        'Last Game of Night',
        'Us Total Shots',
        'Us Hits',
        'Us TOA',
        'Us Passing Rate',
        'Us Faceoffs Won',
        'Us Penalty Minutes',
        'Us Power Play Minutes',
        'Us Shorthanded Goals'
    ]

    # Ensure required columns are present
    missing_columns = [col for col in feature_cols + ['Goal Differential'] if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing from the data: {missing_columns}")
    else:
        # Handle missing values by dropping rows with missing values in the required columns
        data_for_model = df[feature_cols + ['Goal Differential']].dropna()
        if data_for_model.empty:
            st.error("Not enough data available for model training after dropping missing values.")
        else:
            # Prepare the data for predictions
            X = data_for_model[feature_cols]
            y = data_for_model['Goal Differential']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)

            st.subheader("Input Scenario")
            input_cols = st.columns(3)
            inputs = {
                "Us # Players": input_cols[0].number_input("Our Team Size", 1, 5, 3),
                "# Opponent Players": input_cols[0].number_input("Opponent Team Size", 1, 5, 3),
                "First Game of Night": input_cols[0].checkbox("Is this the First Game of the Night?", value=False),
                "Last Game of Night": input_cols[0].checkbox("Is this the Last Game of the Night?", value=False),
                "Nolan Goal": input_cols[1].number_input("Nolan's Goals", 0, 10, 1),
                "Andrew Goal": input_cols[1].number_input("Andrew's Goals", 0, 10, 1),
                "Morgan Goal": input_cols[1].number_input("Morgan's Goals", 0, 10, 1),
                "Us Total Shots": input_cols[2].number_input("Us Total Shots", 0, 100, 20),
                "Us Hits": input_cols[2].number_input("Us Hits", 0, 50, 10),
                "Us TOA": input_cols[2].number_input("Us Time on Attack (minutes)", 0.0, 60.0, 5.0),
                "Us Passing Rate": input_cols[2].number_input("Us Passing Rate (%)", 0.0, 100.0, 70.0),
                "Us Faceoffs Won": input_cols[2].number_input("Us Faceoffs Won", 0, 50, 15),
                "Us Penalty Minutes": input_cols[2].number_input("Us Penalty Minutes", 0, 20, 2),
                "Us Power Play Minutes": input_cols[2].number_input("Us Power Play Minutes (minutes)", 0.0, 20.0, 4.0),
                "Us Shorthanded Goals": input_cols[2].number_input("Us Shorthanded Goals", 0, 5, 0)
            }

            # Convert boolean inputs to integers (True -> 1, False -> 0)
            inputs["First Game of Night"] = int(inputs["First Game of Night"])
            inputs["Last Game of Night"] = int(inputs["Last Game of Night"])

            input_data = np.array([[inputs[col] for col in feature_cols]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            st.metric("Predicted Goal Differential", f"{prediction:.2f}")
            if prediction > 0:
                st.success(f"Expected to win by approximately {prediction:.2f} goals!")
            elif prediction < 0:
                st.error(f"Expected to lose by approximately {abs(prediction):.2f} goals.")
            else:
                st.info("This scenario predicts a closely matched game that could end in a tie.")

            # Feature Importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title="Feature Importance in Predicting Goal Differential",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)