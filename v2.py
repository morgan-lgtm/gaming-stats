import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import boto3
import os

# Page Configuration
st.set_page_config(
    page_title="NHL Gaming Stats",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 0 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    .custom-metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data Function
@st.cache_data
def load_data():
    if os.getenv('IS_AWS', 'false').lower() == 'true':
        s3 = boto3.client('s3')
        try:
            obj = s3.get_object(Bucket=os.getenv('S3_BUCKET'), Key=os.getenv('CSV_KEY'))
            df = pd.read_csv(obj['Body'])
        except Exception as e:
            st.error(f"Error loading data from S3: {e}")
            return pd.DataFrame()
    else:
        try:
            df = pd.read_csv('data/nhl_stats.csv')
        except FileNotFoundError:
            st.error("Data file not found.")
            return pd.DataFrame()

    # Convert columns to correct data types
    numeric_columns = [
        'Us Goals', 'Opponent Goals',
        'Us Total Shots', 'Opponent Total Shots',
        'Us Hits', 'Opponent Hits',
        'Us TOA', 'Opponent TOA',
        'Us Passing Rate', 'Opponent Passing Rate',
        'Us Faceoffs Won', 'Opponent Faceoffs Won',
        'Us Penalty Minutes', 'Opponent Penalty Minutes',
        'Us Power Play Minutes', 'Opponent Power Play Minutes',
        'Us Shorthanded Goals', 'Opponent Shorthanded Goals',
        'Nolan Goal', 'Andrew Goal', 'Morgan Goal'
    ]

    # Replace 'Missing' with NaN and convert to numeric
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('Missing', np.nan), errors='coerce')

    # Basic data processing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Goal Differential'] = df['Us Goals'] - df['Opponent Goals']
    df['Result'] = df['Win / Loss'].map({'Win': 'Win', 'Loss': 'Loss', 'Quit': 'Forfeit'})
    
    # Calculate important metrics
    df['Shot Efficiency'] = np.where(
        df['Us Total Shots'] > 0,
        (df['Us Goals'] / df['Us Total Shots'] * 100).round(1),
        0
    )

    total_toa = df['Us TOA'] + df['Opponent TOA']
    df['Possession Score'] = np.where(
        total_toa > 0,
        (df['Us TOA'] / total_toa * 100).round(1),
        50
    )

    df['Power Play Efficiency'] = np.where(
        df['Us Power Play Minutes'] > 0,
        (df['Us Goals'] / df['Us Power Play Minutes']).round(2),
        0
    )
    
    return df

# Load data
df = load_data()
if df.empty:
    st.stop()

# Main Navigation
tabs = st.tabs([
    "📊 Overview",
    "🏒 Team Analysis",
    "👥 Player Stats",
    "📈 Performance Trends",
    "🎯 Predictions"
])

# Overview Tab
with tabs[0]:
    st.title("NHL Gaming Stats Dashboard")
    
    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    wins = (df['Result'] == 'Win').sum()
    losses = (df['Result'] == 'Loss').sum()
    forfeits = (df['Result'] == 'Forfeit').sum()
    total_games = len(df)
    
    with col1:
        st.markdown('<div class="custom-metric-container">', unsafe_allow_html=True)
        st.metric(
            "Record",
            f"{wins}-{losses}-{forfeits}",
            f"Total Games: {total_games}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="custom-metric-container">', unsafe_allow_html=True)
        win_rate = wins/total_games * 100
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            f"{win_rate - 50:.1f}% vs 50%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="custom-metric-container">', unsafe_allow_html=True)
        avg_goals = df['Us Goals'].mean()
        st.metric(
            "Avg Goals For",
            f"{avg_goals:.1f}",
            f"{avg_goals - df['Opponent Goals'].mean():.1f} vs Opponents"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="custom-metric-container">', unsafe_allow_html=True)
        shot_efficiency = df['Shot Efficiency'].mean()
        st.metric(
            "Shot Efficiency",
            f"{shot_efficiency:.1f}%",
            f"{shot_efficiency - 10:.1f}% vs Target"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Results Distribution and Performance Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Results Distribution")
        results_data = df['Result'].value_counts()
        fig_pie = px.pie(
            values=results_data.values,
            names=results_data.index,
            title="Win/Loss/Forfeit Distribution",
            color=results_data.index,
            color_discrete_map={
                'Win': '#2ecc71',
                'Loss': '#e74c3c',
                'Forfeit': '#95a5a6'
            },
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Performance Trends")
        rolling_avg = df['Goal Differential'].rolling(window=5).mean()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['Goal Differential'],
            name='Goal Differential',
            mode='markers',
            marker=dict(
                size=8,
                color=df['Goal Differential'],
                colorscale='RdYlBu',
                showscale=True
            )
        ))
        fig_trend.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=rolling_avg,
            name='5-Game Average',
            line=dict(color='black', width=2)
        ))
        fig_trend.update_layout(
            title="Goal Differential Trend",
            xaxis_title="Game Number",
            yaxis_title="Goal Differential",
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Team Performance Summary
    st.subheader("Team Performance Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        shot_efficiency = df['Shot Efficiency'].mean()
        fig_gauge1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=shot_efficiency,
            title={'text': "Shot Efficiency %"},
            delta={'reference': 10},
            gauge={
                'axis': {'range': [0, 20]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 5], 'color': "#ff5733"},
                    {'range': [5, 10], 'color': "#ffd700"},
                    {'range': [10, 20], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            }
        ))
        fig_gauge1.update_layout(height=300)
        st.plotly_chart(fig_gauge1, use_container_width=True)

    with col2:
        possession = df['Possession Score'].mean()
        fig_gauge2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=possession,
            title={'text': "Possession %"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff5733"},
                    {'range': [40, 60], 'color': "#ffd700"},
                    {'range': [60, 100], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge2.update_layout(height=300)
        st.plotly_chart(fig_gauge2, use_container_width=True)

    with col3:
        win_rate = (wins/total_games) * 100
        fig_gauge3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=win_rate,
            title={'text': "Win Rate %"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#9b59b6"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff5733"},
                    {'range': [40, 60], 'color': "#ffd700"},
                    {'range': [60, 100], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge3.update_layout(height=300)
        st.plotly_chart(fig_gauge3, use_container_width=True)

    # Recent Games Summary
    st.subheader("Recent Games Performance")
    recent_games = df.tail(5).copy()
    recent_games['Game_Date'] = recent_games['Date'].dt.strftime('%Y-%m-%d')
    
    fig_recent = go.Figure()
    fig_recent.add_trace(go.Bar(
        x=recent_games['Game_Date'],
        y=recent_games['Us Goals'],
        name='Goals For',
        marker_color='#2ecc71'
    ))
    fig_recent.add_trace(go.Bar(
        x=recent_games['Game_Date'],
        y=recent_games['Opponent Goals'],
        name='Goals Against',
        marker_color='#e74c3c'
    ))
    fig_recent.update_layout(
        title="Last 5 Games Breakdown",
        barmode='group',
        xaxis_title="Game Date",
        yaxis_title="Goals",
        hovermode='x unified'
    )
    st.plotly_chart(fig_recent, use_container_width=True)

# Team Analysis Tab
with tabs[1]:
    st.title("Team Analysis")
    
    # Team Performance Metrics
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = {
        "Avg Goals For": df['Us Goals'].mean(),
        "Avg Goals Against": df['Opponent Goals'].mean(),
        "Avg Shot Efficiency": df['Shot Efficiency'].mean(),
        "Avg Possession": df['Possession Score'].mean(),
        "Power Play Efficiency": df['Power Play Efficiency'].mean(),
        "Faceoff Win %": (df['Us Faceoffs Won'].sum() / 
                         (df['Us Faceoffs Won'].sum() + df['Opponent Faceoffs Won'].sum()) * 100),
    }
    
    for col, (metric, value) in zip([col1, col2, col3, col4], list(metrics.items())[:4]):
        col.metric(metric, f"{value:.1f}")

    # Detailed Team Stats
    st.subheader("Detailed Team Statistics")
    
    # Create comparison plots
    comparison_metrics = {
        'Goals': ['Us Goals', 'Opponent Goals'],
        'Shots': ['Us Total Shots', 'Opponent Total Shots'],
        'Hits': ['Us Hits', 'Opponent Hits'],
        'Time on Attack': ['Us TOA', 'Opponent TOA']
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(comparison_metrics.keys())
    )

    row, col = 1, 1
    for metric_name, (us_metric, opp_metric) in comparison_metrics.items():
        fig.add_trace(
            go.Box(y=df[us_metric], name="Us", marker_color='#2ecc71'),
            row=row, col=col
        )
        fig.add_trace(
            go.Box(y=df[opp_metric], name="Opponent", marker_color='#e74c3c'),
            row=row, col=col
        )
        
        if col == 2:
            row += 1
            col = 1
        else:
            col += 1

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Team vs Opponent Statistics Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Streak Analysis
    st.subheader("Win/Loss Streaks")
    
    def calculate_streaks(df):
        streaks = []
        current_streak = 1
        current_result = df.iloc[0]['Result']
        
        for i in range(1, len(df)):
            if df.iloc[i]['Result'] == current_result:
                current_streak += 1
            else:
                streaks.append({'Result': current_result, 'Length': current_streak})
                current_streak = 1
                current_result = df.iloc[i]['Result']
        
        streaks.append({'Result': current_result, 'Length': current_streak})
        return pd.DataFrame(streaks)

    streak_df = calculate_streaks(df)
    
    fig_streaks = px.bar(
        streak_df,
        x=streak_df.index,
        y='Length',
        color='Result',
        color_discrete_map={
            'Win': '#2ecc71',
            'Loss': '#e74c3c',
            'Forfeit': '#95a5a6'
        },
        title="Team Streaks Analysis"
    )
    st.plotly_chart(fig_streaks, use_container_width=True)

# Player Stats Tab
with tabs[2]:
    st.title("Player Statistics")
    
    players = ['Nolan', 'Andrew', 'Morgan']
    
    # Player Summary Cards
    st.subheader("Player Performance Summary")
    cols = st.columns(len(players))
    
    for idx, player in enumerate(players):
        with cols[idx]:
            total_goals = df[f'{player} Goal'].sum()
            games_played = len(df[df[f'{player} Goal'] >= 0])
            goals_per_game = total_goals / games_played if games_played > 0 else 0
            games_with_goals = len(df[df[f'{player} Goal'] > 0])
            
            st.markdown(f"""
            <div class="stat-card">
                <h3>{player}</h3>
                <p>Total Goals: {total_goals}</p>
                <p>Goals/Game: {goals_per_game:.2f}</p>
                <p>Games with Goals: {games_with_goals}</p>
                <p>Goal Rate: {(games_with_goals/games_played*100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Player Goal Distribution
    st.subheader("Player Goal Distribution")
    
    goal_data = pd.DataFrame({
        'Player': [p for p in players for _ in range(len(df))],
        'Goals': pd.concat([df[f'{p} Goal'] for p in players]).values,
        'Game': np.tile(range(len(df)), len(players))
    })
    
    fig_goals = px.line(
        goal_data,
        x='Game',
        y='Goals',
        color='Player',
        title="Player Goals Over Time",
        color_discrete_sequence=['#2ecc71', '#3498db', '#9b59b6']
    )
    st.plotly_chart(fig_goals, use_container_width=True)
    
    # Player Synergy Analysis
    st.subheader("Player Synergy Analysis")
    
    synergy_data = []
    for p1 in players:
        for p2 in players:
            if p1 != p2:
                both_scoring = df[(df[f'{p1} Goal'] > 0) & (df[f'{p2} Goal'] > 0)]
                synergy_data.append({
                    'Player 1': p1,
                    'Player 2': p2,
                    'Games Together': len(both_scoring),
                    'Win Rate': (both_scoring['Result'] == 'Win').mean() * 100 if len(both_scoring) > 0 else 0,
                    'Avg Goals': both_scoring['Us Goals'].mean() if len(both_scoring) > 0 else 0
                })
    
    synergy_df = pd.DataFrame(synergy_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_synergy = px.scatter(
            synergy_df,
            x='Games Together',
            y='Win Rate',
            size='Avg Goals',
            color='Player 1',
            hover_data=['Player 2'],
            title="Player Synergy Impact"
        )
        st.plotly_chart(fig_synergy, use_container_width=True)
    
    with col2:
        st.dataframe(
            synergy_df.sort_values('Win Rate', ascending=False)
            .style.format({
                'Win Rate': '{:.1f}%',
                'Avg Goals': '{:.2f}'
            })
        )

# Performance Trends Tab
with tabs[3]:
    st.title("Performance Trends")
    
    # Time Period Selection
    time_period = st.selectbox(
        "Select Time Period",
        ["All Time", "Last 10 Games", "Last 5 Games", "Custom Range"]
    )
    
    if time_period == "Last 10 Games":
        df_period = df.tail(10)
    elif time_period == "Last 5 Games":
        df_period = df.tail(5)
    elif time_period == "Custom Range":
        date_range = st.date_input(
            "Select Date Range",
            [df['Date'].min(), df['Date'].max()]
        )
        df_period = df[
            (df['Date'].dt.date >= date_range[0]) &
            (df['Date'].dt.date <= date_range[1])
        ]
    else:
        df_period = df
    
    # Performance Metrics Over Time
    st.subheader("Performance Metrics Trends")
    
    metrics = st.multiselect(
        "Select Metrics to Display",
        ["Goal Differential", "Shot Efficiency", "Possession Score", "Power Play Efficiency"],
        default=["Goal Differential", "Shot Efficiency"]
    )
    
    fig_trends = go.Figure()
    
    for metric in metrics:
        rolling_avg = df_period[metric].rolling(window=3).mean()
        fig_trends.add_trace(go.Scatter(
            x=list(range(len(df_period))),
            y=df_period[metric],
            name=metric,
            mode='lines+markers',
            line=dict(width=1),
            marker=dict(size=6)
        ))
        fig_trends.add_trace(go.Scatter(
            x=list(range(len(df_period))),
            y=rolling_avg,
            name=f"{metric} (3-game avg)",
            line=dict(width=2, dash='dash')
        ))
    
    fig_trends.update_layout(
        title="Performance Metrics Over Time",
        xaxis_title="Game Number",
        yaxis_title="Value",
        hovermode='x unified'
    )
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Performance by Opponent Size
    st.subheader("Performance by Opponent Team Size")
    
    opp_size_stats = df_period.groupby('# Opponent Players').agg({
        'Goal Differential': 'mean',
        'Win / Loss': lambda x: (x == 'Win').mean() * 100,
        'Us Goals': 'mean',
        'Opponent Goals': 'mean'
    }).round(2)
    
    fig_opp_size = go.Figure()
    fig_opp_size.add_trace(go.Bar(
        x=opp_size_stats.index,
        y=opp_size_stats['Goal Differential'],
        name='Avg Goal Differential',
        marker_color='#2ecc71'
    ))
    fig_opp_size.add_trace(go.Scatter(
        x=opp_size_stats.index,
        y=opp_size_stats['Win / Loss'],
        name='Win Rate %',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig_opp_size.update_layout(
        title="Performance vs Opponent Team Size",
        xaxis_title="Number of Opponent Players",
        yaxis_title="Average Goal Differential",
        yaxis2=dict(
            title="Win Rate %",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    st.plotly_chart(fig_opp_size, use_container_width=True)

# Predictions Tab
with tabs[4]:
    st.title("Game Predictions")
    
    # Prediction Model Setup
    st.subheader("Game Outcome Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Team Setup")
        num_players = st.number_input("Number of Our Players", 1, 5, 3)
        opp_players = st.number_input("Number of Opponent Players", 1, 5, 3)
        
        st.markdown("### Expected Performance")
        exp_shots = st.number_input("Expected Shots", 0, 100, 20)
        exp_hits = st.number_input("Expected Hits", 0, 50, 10)
        exp_toa = st.number_input("Expected Time on Attack (minutes)", 0.0, 20.0, 5.0)
    
    with col2:
        st.markdown("### Additional Factors")
        exp_passing = st.number_input("Expected Passing %", 0.0, 100.0, 70.0)
        exp_faceoffs = st.number_input("Expected Faceoffs Won", 0, 50, 15)
        exp_penalties = st.number_input("Expected Penalty Minutes", 0, 20, 2)
        
        first_game = st.checkbox("First Game of Night")
        last_game = st.checkbox("Last Game of Night")
    
    # Prepare prediction features
    feature_cols = [
        'Us # Players', '# Opponent Players',
        'Us Total Shots', 'Us Hits', 'Us TOA',
        'Us Passing Rate', 'Us Faceoffs Won', 'Us Penalty Minutes',
        'First Game of Night', 'Last Game of Night'
    ]
    
    X = df[feature_cols]
    y = df['Goal Differential']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Make prediction
    prediction_input = np.array([[
        num_players, opp_players,
        exp_shots, exp_hits, exp_toa,
        exp_passing, exp_faceoffs, exp_penalties,
        int(first_game), int(last_game)
    ]])
    
    prediction = model.predict(prediction_input)[0]
    
    # Display prediction
    st.markdown("### Prediction Result")
    
    if prediction > 0:
        st.success(f"Predicted to win by {abs(prediction):.1f} goals")
    else:
        st.error(f"Predicted to lose by {abs(prediction):.1f} goals")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.markdown("### Key Factors in Prediction")
    fig_importance = px.bar(
        importance,
        x='Feature',
        y='Importance',
        title="Feature Importance in Prediction",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_importance, use_container_width=True)

if __name__ == "__main__":
    pass