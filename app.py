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

    # Handle Missing Values
    df_cleaned = df.dropna(subset=['Goal Differential', 'Us Goals', 'Opponent Goals', 'Win / Loss'])

    return df_cleaned

# Load the data
df = load_data()

# If DataFrame is empty, stop execution
if df.empty:
    st.stop()

# Main Dashboard with Tabs
st.title("🏒 NHL Gaming Analytics Dashboard")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://wallpapers.com/images/hd/n-h-l-logoon-dark-wood-texture-52oks3apn5xbkjsa.jpg");
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
    cols = st.columns(4)

    # Calculate Current Streak (Consecutive Wins or Losses)
    def calculate_streak(df):
        streak = 0
        streak_type = None

        for diff in reversed(df['Goal Differential']):
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

# Team Analysis Tab (Tab 2)
with tabs[1]:
    st.header("Team Performance Analysis")
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
    values = list(team_stats.values())
    categories = list(team_stats.keys())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=px.colors.sequential.Tealgrn,
        text=[f"{v:.2f}" for v in values],
        textposition='auto'
    ))
    fig.update_layout(title="Team Performance Metrics", xaxis_title="Metrics", yaxis_title="Values")
    st.plotly_chart(fig, use_container_width=True)

    # Analyze Performance in First and Last Games of the Night
    st.subheader("Performance in First and Last Games of the Night")

    # Calculate average goal differential in first games, last games, and other games
    first_game_avg = df[df['First Game of Night'] == 1]['Goal Differential'].mean()
    last_game_avg = df[df['Last Game of Night'] == 1]['Goal Differential'].mean()
    middle_games_avg = df[(df['First Game of Night'] == 0) & (df['Last Game of Night'] == 0)]['Goal Differential'].mean()

    game_type = ['First Game', 'Middle Game', 'Last Game']
    avg_goal_diff = [first_game_avg, middle_games_avg, last_game_avg]

    fig_game_type = px.bar(
        x=game_type,
        y=avg_goal_diff,
        title="Average Goal Differential by Game Position",
        labels={'x': 'Game Position', 'y': 'Average Goal Differential'},
        color=game_type,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_game_type, use_container_width=True)

    st.markdown("""
    **Insights:**
    - Analyze if there's a significant difference in performance during the first, middle, or last games of the night.
    """)

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
        'Highest Goals in a Single Game': [df[f'{player} Goal'].max() for player in players]
    })

    # Display individual player stats in a table
    st.table(player_stats.style.format({
        'Average Goals per Game': "{:.2f}",
        'Total Goals': "{:.0f}",
        'Highest Goals in a Single Game': "{:.0f}"
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

    # Average Goals per Game Bar Chart
    fig_avg_goals = px.bar(
        player_stats,
        x='Player',
        y='Average Goals per Game',
        title='Average Goals per Game per Player',
        text='Average Goals per Game',
        color='Player',
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    fig_avg_goals.update_traces(textposition='outside')
    fig_avg_goals.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_avg_goals, use_container_width=True)

    # Highest Goals in a Single Game Bar Chart
    fig_highest_goals = px.bar(
        player_stats,
        x='Player',
        y='Highest Goals in a Single Game',
        title='Highest Goals in a Single Game per Player',
        text='Highest Goals in a Single Game',
        color='Player',
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    fig_highest_goals.update_traces(textposition='outside')
    fig_highest_goals.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_highest_goals, use_container_width=True)

# Opponent Analysis Tab (Tab 4)
with tabs[3]:
    st.header("Opponent Analysis")
    avg_performance = df.groupby('# Opponent Players').agg({
        'Goal Differential': 'mean',
        'Us Goals': 'mean',
        'Opponent Goals': 'mean'
    }).reset_index()

    # Ensure '# Opponent Players' is integer
    avg_performance['# Opponent Players'] = avg_performance['# Opponent Players'].astype(int)

    if not avg_performance.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Goal Differential'],
                name="Avg Goal Differential",
                marker_color='indianred'
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Us Goals'],
                name="Avg Goals Scored",
                mode="lines+markers",
                marker=dict(color='green')
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                x=avg_performance['# Opponent Players'],
                y=avg_performance['Opponent Goals'],
                name="Avg Goals Conceded",
                mode="lines+markers",
                marker=dict(color='blue')
            ),
            secondary_y=True
        )
        fig.update_layout(title="Opponent Performance Metrics")
        fig.update_xaxes(title_text="Number of Opponent Players", tickmode='linear', tick0=1, dtick=1)
        fig.update_yaxes(title_text="Goal Differential", secondary_y=False)
        fig.update_yaxes(title_text="Goals", secondary_y=True)
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
            'Goal Differential': True
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

# AI Insights Tab
with tabs[5]:
    st.header("AI-Generated Insights")

    # Enhanced AI Insights
    avg_goal_diff = df['Goal Differential'].mean()
    top_opponent_size = df.groupby('# Opponent Players')['Goal Differential'].mean().idxmax()
    player_impact = {
        player: df[df[f'{player} Goal'] > 0]['Goal Differential'].mean()
        for player in players
    }
    top_player = max(player_impact, key=player_impact.get)

    # Analyze performance in first and last games
    first_game_performance = df[df['First Game of Night'] == 1]['Goal Differential'].mean()
    last_game_performance = df[df['Last Game of Night'] == 1]['Goal Differential'].mean()

    insights = [
        f"**Average Goal Differential**: {avg_goal_diff:.2f}. {'Focus on improving defensive strategies.' if avg_goal_diff < 0 else 'Keep up the good offensive work!'}",
        f"**Best Performance Against Teams with {top_opponent_size} Players**. Consider strategies that work well in these matchups.",
        f"**Top Performer**: {top_player} has the highest impact on goal differential when scoring. Creating more opportunities for them could improve overall performance.",
        f"**First Game Performance**: Average goal differential in first games is {first_game_performance:.2f}. {'A strong start sets the tone for the night!' if first_game_performance > 0 else 'Consider warming up before the first game to improve performance.'}",
        f"**Last Game Performance**: Average goal differential in last games is {last_game_performance:.2f}. {'Finishing strong!' if last_game_performance > 0 else 'Fatigue might be affecting the last game; consider strategies to maintain performance.'}"
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
        'Last Game of Night'
    ]

    # Ensure required columns are present
    missing_columns = [col for col in feature_cols + ['Goal Differential'] if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing from the data: {missing_columns}")
    else:
        # Check for missing values in required columns
        if df[feature_cols + ['Goal Differential']].isna().sum().sum() > 0:
            st.error("There are missing values in the dataset. Please check the data cleaning steps.")
        else:
            # Prepare the data for predictions
            X = df[feature_cols]
            y = df['Goal Differential']

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
                "Morgan Goal": input_cols[1].number_input("Morgan's Goals", 0, 10, 1)
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
            st.plotly_chart(fig, use_container_width=True)