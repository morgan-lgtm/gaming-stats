import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="NHL Gaming Stats",
    page_icon="🏒",
    layout="wide",
)

# Custom CSS for stunning visuals
st.markdown("""
    <style>
    .main {
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0 20px;
        font-weight: 600;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(52, 152, 219, 0.5);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: white;
    }
    .mvp-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.2), rgba(52, 152, 219, 0.2));
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .player-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.2), rgba(52, 152, 219, 0.2));
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Set background
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images6.alphacoders.com/982/thumb-1920-982416.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/nhl_stats.csv')
        
        # Handle Missing values and data types
        for column in df.columns:
            df[column] = df[column].replace('Missing', np.nan)
        
        # Convert columns
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_columns = ['Us Goals', 'Opponent Goals', 'Us # Players', '# Opponent Players',
                         'Nolan Goal', 'Andrew Goal', 'Morgan Goal', 'Us Total Shots',
                         'Us Hits', 'Us TOA', 'Us Passing Rate', 'Us Faceoffs Won']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create metrics
        df['IsWin'] = (df['Win / Loss'] == 'Win').astype(int)
        df['Goal Differential'] = df['Us Goals'] - df['Opponent Goals']
        df['Shot Efficiency'] = np.where(df['Us Total Shots'] > 0,
                                       (df['Us Goals'] / df['Us Total Shots'] * 100), np.nan)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_mvp(df, num_games=5):
    """Calculate MVP based on recent games"""
    recent_games = df.tail(num_games)
    players = ['Nolan', 'Andrew', 'Morgan']
    
    mvp_stats = []
    for player in players:
        goals = recent_games[f'{player} Goal'].sum()
        wins_with_goals = recent_games[
            (recent_games[f'{player} Goal'] > 0) & 
            (recent_games['Win / Loss'] == 'Win')
        ].shape[0]
        
        mvp_stats.append({
            'Player': player,
            'Goals': goals,
            'Win Contributions': wins_with_goals,
            'Impact Score': goals * 1.5 + wins_with_goals * 2
        })
    
    return max(mvp_stats, key=lambda x: x['Impact Score'])

# Load data
df = load_data()

if not df.empty:
    st.title("🏒 NHL Gaming Analytics Hub")
    
    # Top Row: MVP and Key Metrics
    col1, col2, col3, col4 = st.columns([1.2, 0.9, 0.9, 0.9])
    
    with col1:
        mvp_data = calculate_mvp(df)
        st.markdown(f"""
            <div class="mvp-card">
                <h2 style="color: #2ecc71; text-align: center; font-size: 24px;">
                    🏆 WEEKLY MVP
                </h2>
                <h1 style="color: #ffffff; text-align: center; font-size: 42px;">
                    {mvp_data['Player']}
                </h1>
                <div style="text-align: center; color: #ffffff;">
                    <p style="font-size: 20px;">
                        {mvp_data['Goals']} Goals | {mvp_data['Win Contributions']} Win Contributions
                    </p>
                    <p style="font-size: 16px; color: #2ecc71;">
                        Last 5 Games Impact Score: {mvp_data['Impact Score']:.1f}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics
    wins = (df['Win / Loss'] == 'Win').sum()
    losses = (df['Win / Loss'] == 'Loss').sum()
    forfeits = (df['Win / Loss'] == 'Quit').sum()
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Record", f"{wins}-{losses}-{forfeits}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        win_rate = wins/len(df) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_goals = df['Us Goals'].mean()
        st.metric("Avg Goals", f"{avg_goals:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Create tabs
    tabs = st.tabs(["📊 Overview", "📈 Performance", "👥 Players", "🎯 Predictions"])

    # Overview Tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Results Distribution
            results = df['Win / Loss'].value_counts()
            fig_pie = px.pie(
                values=results.values,
                names=results.index,
                title="Results Distribution",
                color=results.index,
                color_discrete_map={
                    'Win': '#2ecc71',
                    'Loss': '#e74c3c',
                    'Quit': '#95a5a6'
                },
                hole=0.4
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            fig_pie.update_layout(
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0.3)',
                plot_bgcolor='rgba(0,0,0,0.3)',
                font_color='white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Efficiency Gauge Charts
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=("", "")
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=df['Shot Efficiency'].mean(),
                    title={'text': "Shot Efficiency %"},
                    gauge={
                        'axis': {'range': [0, 20]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, 5], 'color': "#ff5733"},
                            {'range': [5, 10], 'color': "#ffd700"},
                            {'range': [10, 20], 'color': "#2ecc71"}
                        ]
                    }
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=win_rate,
                    title={'text': "Win Rate %"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ff5733"},
                            {'range': [40, 60], 'color': "#ffd700"},
                            {'range': [60, 100], 'color': "#2ecc71"}
                        ]
                    }
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0.3)',
                plot_bgcolor='rgba(0,0,0,0.3)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Performance Tab
    with tabs[1]:
        # Performance Timeline
        df['DailyGoalDiff'] = df.groupby('Date')['Goal Differential'].transform('mean')
        daily_stats = df.groupby('Date')['Goal Differential'].mean().reset_index()
        daily_stats['Weekly_MA'] = daily_stats['Goal Differential'].rolling(window=7, min_periods=1).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_stats['Date'],
            y=daily_stats['Weekly_MA'],
            name='Weekly Trend',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=2)
        ))
        
        for result, color in [('Win', '#2ecc71'), ('Loss', '#e74c3c'), ('Quit', '#95a5a6')]:
            mask = df['Win / Loss'] == result
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'Date'],
                y=df.loc[mask, 'Goal Differential'],
                mode='markers',
                name=result,
                marker=dict(size=12, color=color, line=dict(color='white', width=1))
            ))
        
        fig.update_layout(
            title="Performance Timeline",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0.3)',
            plot_bgcolor='rgba(0,0,0,0.3)',
            height=400,
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Players Tab
    with tabs[2]:
        st.title("Player Performance Center")
        
        # Player Selection
        players = ['Nolan', 'Andrew', 'Morgan']
        selected_player = st.selectbox(
            "Select Player for Detailed Analysis",
            players,
            index=0,
            key='player_select'
        )
        
        # Calculate player stats
        def get_player_stats(player):
            stats = {
                'Total Goals': df[f'{player} Goal'].sum(),
                'Games Played': len(df[df[f'{player} Goal'] >= 0]),
                'Games With Goals': len(df[df[f'{player} Goal'] > 0]),
                'Max Goals in Game': df[f'{player} Goal'].max(),
                'Win Rate With Goals': (df[
                    (df[f'{player} Goal'] > 0) & 
                    (df['Win / Loss'] == 'Win')
                ].shape[0] / len(df[df[f'{player} Goal'] > 0])) * 100 if len(df[df[f'{player} Goal'] > 0]) > 0 else 0
            }
            stats['Goals Per Game'] = stats['Total Goals'] / stats['Games Played'] if stats['Games Played'] > 0 else 0
            return stats

        col1, col2 = st.columns([1, 2])
        
        with col1:
            player_stats = get_player_stats(selected_player)
            st.markdown(f"""
                <div class="player-card">
                    <h1 style="color: #2ecc71; font-size: 36px; margin-bottom: 20px;">
                        {selected_player}
                    </h1>
                    <div style="font-size: 24px; color: white; margin-bottom: 15px;">
                        {player_stats['Total Goals']} Goals
                    </div>
                    <div style="color: #bdc3c7; font-size: 18px;">
                        {player_stats['Games Played']} Games Played<br>
                        {player_stats['Goals Per Game']:.2f} Goals/Game<br>
                        {player_stats['Win Rate With Goals']:.1f}% Win Rate When Scoring
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Scoring Rate Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=player_stats['Games With Goals'] / player_stats['Games Played'] * 100,
                title={'text': "Scoring Rate %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ecc71"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ff5733"},
                        {'range': [30, 70], 'color': "#ffd700"},
                        {'range': [70, 100], 'color': "#2ecc71"}
                    ]
                }
            ))
            
            fig.update_layout(
                height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Goal Timeline
            fig = go.Figure()
            
            # Clean data for visualization
            timeline_df = pd.DataFrame({
                'Date': df['Date'],
                'Goals': df[f'{selected_player} Goal']
            }).fillna(0)  # Replace NaN with 0 for visualization
            
            fig.add_trace(go.Scatter(
                x=timeline_df['Date'],
                y=timeline_df['Goals'],
                mode='markers+lines',
                name='Goals',
                line=dict(color='#2ecc71', width=1),
                marker=dict(
                    size=timeline_df['Goals'] * 5 + 5,  # Now uses cleaned data
                    color='#2ecc71',
                    line=dict(color='white', width=1)
                )
            ))
            
            # Add moving average
            ma = timeline_df['Goals'].rolling(window=3).mean()
            fig.add_trace(go.Scatter(
                x=timeline_df['Date'],
                y=ma,
                mode='lines',
                name='3-Game Average',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Goal Timeline",
                xaxis_title="Date",
                yaxis_title="Goals",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                font_color='white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(0,0,0,0.3)"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Player Comparison Radar Chart
            categories = ['Goals/Game', 'Win Rate', 'Consistency', 'Big Games']
            fig = go.Figure()
            
            for player in players:
                stats = get_player_stats(player)
                goals_per_game = stats['Goals Per Game'] * 10
                win_rate = stats['Win Rate With Goals']
                consistency = (stats['Games With Goals'] / stats['Games Played'] * 100 
                             if stats['Games Played'] > 0 else 0)
                big_games = (len(df[df[f'{player} Goal'] >= 2]) / stats['Games Played'] * 100 
                           if stats['Games Played'] > 0 else 0)
                
                fig.add_trace(go.Scatterpolar(
                    r=[goals_per_game, win_rate, consistency, big_games],
                    theta=categories,
                    name=player,
                    fill='toself'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# Predictions Tab
    with tabs[3]:
        st.header("Win Prediction Analysis")
        
        features = [
            'Us Total Shots', 
            'Us Hits', 
            'Us TOA',
            'Us Passing Rate', 
            'Us Faceoffs Won',
            '# Opponent Players',
            'Opponent Total Shots', 
            'Opponent Hits', 
            'Opponent TOA',
            'Opponent Passing Rate', 
            'Opponent Faceoffs Won',
        ]
        
        model_data = df[features + ['IsWin']].dropna()
        
        if len(model_data) > 0:
            X = model_data[features]
            y = model_data['IsWin']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(random_state=42)
            model.fit(X_scaled, y)
            
            # Create DataFrame with raw coefficients to show direction of impact
            importance_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_[0],
                'Absolute Impact': np.abs(model.coef_[0])
            })
            
            # Sort by absolute impact but keep coefficient sign
            importance_df = importance_df.sort_values('Absolute Impact', ascending=True)
            
            # Create color scale based on coefficient direction
            colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in importance_df['Coefficient']]
            
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Coefficient'],
                orientation='h',
                marker_color=colors,
                text=importance_df['Coefficient'].round(3),
                textposition='auto',
            ))
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="white", opacity=0.5)
            
            fig.update_layout(
                title={
                    'text': "What Drives Winning vs. Losing?",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Impact on Win Probability (negative = increases losing)",
                yaxis_title="Game Factors",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0.3)',
                plot_bgcolor='rgba(0,0,0,0.3)',
                height=600,
                font=dict(
                    color='white',
                    size=14
                ),
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretations
            st.markdown("""
                <div class="metric-card">
                    <h3 style='color: white;'>Key Insights:</h3>
                    <div style='display: flex; gap: 20px;'>
                        <div style='flex: 1;'>
                            <h4 style='color: #2ecc71;'>Factors That Help Winning:</h4>
                            <ul style='color: white;'>
            """, unsafe_allow_html=True)
            
            # Generate insights
            positive_factors = importance_df[importance_df['Coefficient'] > 0].sort_values('Coefficient', ascending=False)
            negative_factors = importance_df[importance_df['Coefficient'] < 0].sort_values('Coefficient')
            
            for _, row in positive_factors.iterrows():
                st.markdown(f"<li>{row['Feature']}</li>", unsafe_allow_html=True)
            
            st.markdown("""
                            </ul>
                        </div>
                        <div style='flex: 1;'>
                            <h4 style='color: #e74c3c;'>Factors That Lead to Losing:</h4>
                            <ul style='color: white;'>
            """, unsafe_allow_html=True)
            
            for _, row in negative_factors.iterrows():
                st.markdown(f"<li>{row['Feature']}</li>", unsafe_allow_html=True)
            
            st.markdown("""
                            </ul>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Insufficient data for prediction analysis.")