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

    # Handle Missing Values
    df_cleaned = df.dropna(subset=['Goal Differential', 'Us Goals', 'Opponent Goals', 'Win / Loss'])

    return df_cleaned

# Load the data
df = load_data()

# If DataFrame is empty, stop execution
if df.empty:
    st.stop()

# Display DataFrame Columns for Verification
# st.sidebar.header("Data Information")
# st.sidebar.write("### DataFrame Columns:", df.columns.tolist())

# Sidebar for AI Insights
with st.sidebar:
    st.header("AI Insights")
    if st.button("Generate AI Insights"):
        insights = [
            f"**Average Goal Differential**: {df['Goal Differential'].mean():.2f}. {'Focus on improving defensive strategies.' if df['Goal Differential'].mean() < 0 else 'Keep up the good offensive work!'}",
            f"**Best Performance Against Teams with {df.groupby('# Opponent Players')['Goal Differential'].mean().idxmax()} Players**. Consider strategies that work well in these matchups.",
            f"**Top Performer**: {'Nolan' if df[df['Nolan Goal'] > 0]['Goal Differential'].mean() > max(df[df['Andrew Goal'] > 0]['Goal Differential'].mean(), df[df['Morgan Goal'] > 0]['Goal Differential'].mean()) else 'Andrew' if df[df['Andrew Goal'] > 0]['Goal Differential'].mean() > df[df['Morgan Goal'] > 0]['Goal Differential'].mean() else 'Morgan'} has the highest impact on goal differential when scoring. Creating more opportunities for them could improve overall performance.",
            "**Close Games Analysis**: Analyze your performance in close games (goal differential between -1 and 1) to identify areas for improvement in tight situations."
        ]
        for insight in insights:
            st.info(insight)

# Main Dashboard with Tabs
st.title("🏒 NHL Gaming Analytics Dashboard")

tabs = st.tabs(["Dashboard", "Team Analysis", "Player Analysis", "Opponent Analysis", "Game Breakdown", "Predictions"])

# Dashboard Tab
with tabs[0]:
    st.header("🏒 Overview")
    
    # Calculate MVP Based on Last 5 Games
    last_5_games = df.tail(5)
    players = ['Nolan', 'Andrew', 'Morgan']
    
    # Calculate total goals for each player over the last 5 games
    last_5_goals = {player: last_5_games[f'{player} Goal'].sum() for player in players}
    mvp_player = max(last_5_goals, key=last_5_goals.get)
    mvp_goals = last_5_goals[mvp_player]

    # Enhance MVP section with visual and layout improvements
    st.markdown("""
        <div style="background-color:#f9f9f9;padding:20px;border-radius:10px;margin-bottom:20px;text-align:center;">
            <img src="https://1000logos.net/wp-content/uploads/2018/06/Nashville-Predators-Logo.png" alt="Team Logo" style="width:100px;height:auto;margin-bottom:10px;">
            <h2 style="color:#041E42;">🏅 Current MVP (Last 5 Games)</h2>
            <h1 style="color:#228B22;font-size:48px;">{}</h1>
            <p style="font-size:24px;">Scored <strong>{}</strong> goals in the last 5 games!</p>
        </div>
        """.format(mvp_player, mvp_goals),
        unsafe_allow_html=True
    )

    # Rest of the Overview metrics
    cols = st.columns(4)
    
    # Calculate Current Streak
    streak = 0
    for diff in reversed(df['Goal Differential']):
        if diff > 0:
            streak += 1
        elif diff < 0:
            streak -= 1
            break
        else:
            break
    streak_label = f"{abs(streak)} game {'positive' if streak > 0 else 'negative'}"
    
    metrics = [
        ("Total Games", len(df)),
        ("Avg Goal Differential", f"{df['Goal Differential'].mean():.2f}"),
        ("Current Streak", streak_label),
        ("Win Rate", f"{(df['Goal Differential'] > 0).mean():.2%}")
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    
    # Pie Chart for Wins, Losses, and Quits
    st.subheader("Performance Distribution")
    performance_counts = df['Win / Loss'].value_counts().reset_index()
    performance_counts.columns = ['Result', 'Count']
    
    fig_pie = px.pie(
        performance_counts,
        names='Result',
        values='Count',
        title='Wins, Losses, and Quits Distribution',
        color='Result',
        color_discrete_map={
            'Win': 'green',
            'Loss': 'red',
            'Quit': 'gray'  # Adjust based on actual category names
        },
        hole=0.4  # Creates a donut chart; set to 0 for a full pie chart
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Goal Differential Chart Using SequentialIndex
    plot_df = df.tail(10)  # Last 10 games
    plot_df['Color'] = plot_df['Goal Differential'].apply(lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Tie'))
    
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
        color_discrete_map={'Win': 'blue', 'Loss': 'red', 'Tie': 'gray'}
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

# Team Analysis Tab
with tabs[1]:
    st.header("Team Performance Analysis")
    team_stats = {
        'Win Rate': (df['Goal Differential'] > 0).mean(),
        'Avg Goals Scored': df['Us Goals'].mean(),
        'Avg Goals Conceded': -(df['Opponent Goals'].mean()),
        'Goal Differential': df['Goal Differential'].mean(),
        'Clean Sheets': (df['Opponent Goals'] == 0).mean(),
        'Comeback Rate': ((df['Opponent Goals'] > df['Us Goals']) & (df['Goal Differential'] > 0)).mean()
    }
    values = list(team_stats.values())
    # Color Coding: Red for negative values, Blue otherwise
    colors = ['red' if v < 0 else 'blue' for v in values]
    
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=list(team_stats.keys()),
            fill='toself',
            marker=dict(color=colors, size=10)
        )
    )
    fig.update_layout(
        title="Team Performance Metrics",
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False  # Hide legend since colors indicate positive/negative
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanatory Text for Color Coding
    st.markdown("""
    **Color Coding:**
    - 🟦 **Blue**: Positive Values
    - 🟥 **Red**: Negative Values
    """)

# Player Analysis Tab
with tabs[2]:
    st.header("Player Performance Analysis")
    
    # Existing Synergy Metrics
    player_combinations = [
        ('Nolan', 'Andrew'),
        ('Nolan', 'Morgan'),
        ('Andrew', 'Morgan'),
        ('Nolan', 'Andrew', 'Morgan')
    ]
    synergy_data = []
    for combo in player_combinations:
        goal_cols = [f"{player} Goal" for player in combo]
        condition = df[goal_cols].sum(axis=1) > 0
        synergy_data.append({
            'Combination': ' & '.join(combo),
            'Avg Goal Diff': df[condition]['Goal Differential'].mean(),
            'Avg Goals Scored': df[condition]['Us Goals'].mean(),
            'Win Rate': (df[condition]['Goal Differential'] > 0).mean()
        })
    synergy_df = pd.DataFrame(synergy_data).set_index('Combination')
    
    st.subheader("Synergy Metrics")
    fig_synergy = px.imshow(
        synergy_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Synergy Metrics"
    )
    st.plotly_chart(fig_synergy, use_container_width=True)
    
    # New Section: Individual Player Statistics
    st.subheader("Individual Player Statistics")
    
    players = ['Nolan', 'Andrew', 'Morgan']
    player_stats = pd.DataFrame({
        'Player': players,
        'Total Goals': [df[f'{player} Goal'].sum() for player in players],
        'Average Goals per Game': [df[f'{player} Goal'].mean() for player in players],
        'Highest Goals in a Single Game': [df[f'{player} Goal'].max() for player in players]
    })
    
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
        color_discrete_sequence=px.colors.qualitative.Pastel
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
        color_discrete_sequence=px.colors.qualitative.Pastel
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
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_highest_goals.update_traces(textposition='outside')
    fig_highest_goals.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_highest_goals, use_container_width=True)

# Opponent Analysis Tab
with tabs[3]:
    st.header("Opponent Analysis")
    avg_performance = df.groupby('# Opponent Players').agg({
        'Goal Differential': 'mean',
        'Us Goals': 'mean',
        'Opponent Goals': 'mean'
    }).reset_index()
    
    # Ensure '# Opponent Players' is integer
    avg_performance['# Opponent Players'] = avg_performance['# Opponent Players'].astype(int)
    
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
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.update_yaxes(title_text="Goal Differential", secondary_y=False)
    fig.update_yaxes(title_text="Goals", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# Game Breakdown Tab
with tabs[4]:
    st.header("Game-by-Game Breakdown")
    
    # Scatter Plot Using SequentialIndex with Enhanced Coloring
    scatter = px.scatter(
        df,
        x='SequentialIndex',
        y='Goal Differential',
        color=df['Goal Differential'].apply(lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Tie')),
        color_discrete_map={'Win': 'blue', 'Loss': 'red', 'Tie': 'gray'},
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
    
    selected_game_idx = st.selectbox("Select a game for detailed view", df.index)
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

# Predictions Tab
with tabs[5]:
    st.header("Predictions and What-If Scenarios")
    # Update feature columns to match the input keys
    feature_cols = ['Us # Players', '# Opponent Players', 'Nolan Goal', 'Andrew Goal', 'Morgan Goal']
    
    # Ensure the DataFrame has these columns
    required_columns = feature_cols + ['Goal Differential']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing from the data: {missing_columns}")
    else:
        # Further check for any remaining NaN values
        if df[feature_cols + ['Goal Differential']].isna().sum().sum() > 0:
            st.error("There are still missing values in the dataset. Please check the data cleaning steps.")
        else:
            X = df[feature_cols]
            y = df['Goal Differential']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            st.subheader("Input Scenario")
            input_cols = st.columns(2)
            inputs = {
                "Us # Players": input_cols[0].number_input("Our Team Size", 1, 5, 3),
                "# Opponent Players": input_cols[0].number_input("Opponent Team Size", 1, 5, 3),
                "Nolan Goal": input_cols[1].number_input("Nolan's Goals", 0, 10, 1),
                "Andrew Goal": input_cols[1].number_input("Andrew's Goals", 0, 10, 1),
                "Morgan Goal": input_cols[1].number_input("Morgan's Goals", 0, 10, 1)
            }
            
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
