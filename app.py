import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page Config
st.set_page_config(page_title="NHL Gaming Hub", page_icon="🏒", layout="wide")

# Enhanced CSS for mobile-friendly layout
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
    .trend-up { color: #2ecc71; }
    .trend-down { color: #e74c3c; }

    /* Mobile-friendly adjustments */
    @media only screen and (max-width: 768px) {
        .streak-container, .record-container {
            flex-direction: column;
            align-items: stretch;
        }
        .streak-card, .record-card {
            margin-bottom: 10px;
        }
        .streak-value {
            font-size: 20px;
        }
        .stat-card h3 {
            font-size: 18px;
        }
        .stat-card div {
            font-size: 16px;
        }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/nhl_stats.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')  # Ensure data is sorted chronologically

        # All numeric columns
        numeric_cols = [
            'Us # Players', '# Opponent Players', 'Us Goals', 'Opponent Goals',
            'Nolan Goal', 'Andrew Goal', 'Morgan Goal',
            'Us Total Shots', 'Opponent Total Shots',
            'Us Hits', 'Opponent Hits',
            'Us TOA', 'Opponent TOA',
            'Us Passing Rate', 'Opponent Passing Rate',
            'Us Faceoffs Won', 'Opponent Faceoffs Won',
            'Us Penalty Minutes', 'Opponent Penalty Minutes',
            'Us Power Play Minutes', 'Opponent Power Play Minutes',
            'Us Shorthanded Goals', 'Opponent Shorthanded Goals'
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate derived metrics
        df['IsWin'] = (df['Win / Loss'] == 'Win').astype(int)
        df['Shot Efficiency'] = np.where(
            df['Us Total Shots'] > 0,
            (df['Us Goals'] / df['Us Total Shots'] * 100),
            0
        )
        df['Power Play Efficiency'] = np.where(
            df['Us Power Play Minutes'] > 0,
            df['Us Goals'] / df['Us Power Play Minutes'],
            0
        )
        df['Goal Differential'] = df['Us Goals'] - df['Opponent Goals']

        # Calculate games since last win
        games_since_win = []
        counter = 0
        for win in df['IsWin']:
            if win == 1:
                counter = 0
            else:
                counter += 1
            games_since_win.append(counter)
        df['Games_Since_Win'] = games_since_win

        # Calculate games since last goal for each player
        for player in ['Nolan', 'Andrew', 'Morgan']:
            games_since_goal = []
            counter = 0
            for goals in df[f'{player} Goal']:
                if goals > 0:
                    counter = 0
                else:
                    counter +=1
                games_since_goal.append(counter)
            df[f'Games_Since_{player}_Goal'] = games_since_goal

        # Daily averages for metrics
        metrics_to_aggregate = [
            'Us TOA', 'Us Passing Rate', 'Us Penalty Minutes',
            'Us Hits', 'Us Faceoffs Won', 'Us Power Play Minutes',
            'Shot Efficiency', 'Power Play Efficiency', 'IsWin',
            'Us Goals', 'Goal Differential', 'Nolan Goal',
            'Andrew Goal', 'Morgan Goal', 'Games_Since_Win',
            'Games_Since_Nolan_Goal', 'Games_Since_Andrew_Goal',
            'Games_Since_Morgan_Goal', '# Opponent Players'
        ]

        daily_metrics = df.groupby('Date')[metrics_to_aggregate].mean().reset_index()

        # Get current streaks (from last row of sorted dataframe)
        current_streaks = {
            'Games_Since_Win': int(df['Games_Since_Win'].iloc[-1]),
            'Games_Since_Nolan_Goal': int(df['Games_Since_Nolan_Goal'].iloc[-1]),
            'Games_Since_Andrew_Goal': int(df['Games_Since_Andrew_Goal'].iloc[-1]),
            'Games_Since_Morgan_Goal': int(df['Games_Since_Morgan_Goal'].iloc[-1])
        }

        # Calculate total wins and losses
        total_wins = df['IsWin'].sum()
        total_losses = len(df) - total_wins

        # Return the additional metrics
        return daily_metrics, current_streaks, total_wins, total_losses

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), {}, 0, 0

def create_metric_timeline(df, metric, title, color_scale=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[metric],
        name='Daily Average',
        line=dict(color='#2ecc71', width=3)
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0.1)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_title="Date",
        yaxis_title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def create_special_teams_analysis(daily_metrics):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Power Play Time vs Goals',
            'Penalty Minutes Trend',
            'Power Play Efficiency Trend',
            'Team Performance'
        )
    )

    # Power Play Analysis
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Us Power Play Minutes'],
            y=daily_metrics['Us Goals'],
            mode='markers',
            name='PP Time vs Goals',
            marker=dict(
                color=daily_metrics['IsWin'],
                colorscale='RdYlGn',
                showscale=True
            )
        ),
        row=1, col=1
    )

    # Penalty Minutes Trend
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Date'],
            y=daily_metrics['Us Penalty Minutes'],
            name='Penalty Minutes',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )

    # Power Play Efficiency Trend
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Date'],
            y=daily_metrics['Power Play Efficiency'],
            name='PP Efficiency',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )

    # Win Rate Trend
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Date'],
            y=daily_metrics['IsWin'],
            name='Win Rate',
            line=dict(color='blue', width=2)
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0.1)',
        plot_bgcolor='rgba(0,0,0,0.1)'
    )
    return fig

def main():
    st.title("🏒 NHL Gaming Analytics Hub")

    daily_metrics, current_streaks, total_wins, total_losses = load_data()
    if daily_metrics.empty:
        st.warning("No data available.")
        return

    # Display total wins and losses
    st.markdown(f"""
        <style>
        .record-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .record-card {{
            flex: 1;
            min-width: 140px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            font-size: 24px;
            font-weight: bold;
        }}
        </style>
        <div class="record-container">
            <div class="record-card">
                Total Wins: {total_wins}
            </div>
            <div class="record-card">
                Total Losses: {total_losses}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Add streak metrics to the top
    st.markdown(f"""
        <style>
        .streak-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .streak-card {{
            flex: 1;
            min-width: 140px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }}
        .streak-value {{
            font-size: 20px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .streak-alert {{
            color: #e74c3c;
        }}
        .streak-good {{
            color: #2ecc71;
        }}
        </style>

        <div class="streak-container">
            <div class="streak-card">
                <h4>Games Since Last Win</h4>
                <div class="streak-value {'streak-alert' if current_streaks['Games_Since_Win'] > 3 else 'streak-good'}">
                    {current_streaks['Games_Since_Win']}
                </div>
            </div>
            <div class="streak-card">
                <h4>Games Since Nolan Goal</h4>
                <div class="streak-value {'streak-alert' if current_streaks['Games_Since_Nolan_Goal'] > 5 else 'streak-good'}">
                    {current_streaks['Games_Since_Nolan_Goal']}
                </div>
            </div>
            <div class="streak-card">
                <h4>Games Since Andrew Goal</h4>
                <div class="streak-value {'streak-alert' if current_streaks['Games_Since_Andrew_Goal'] > 5 else 'streak-good'}">
                    {current_streaks['Games_Since_Andrew_Goal']}
                </div>
            </div>
            <div class="streak-card">
                <h4>Games Since Morgan Goal</h4>
                <div class="streak-value {'streak-alert' if current_streaks['Games_Since_Morgan_Goal'] > 5 else 'streak-good'}">
                    {current_streaks['Games_Since_Morgan_Goal']}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📊 Overview", "⚡ Special Teams", "📈 Detailed Metrics", "🎮 Player Stats"])

    with tabs[0]:
        st.header("Team Performance Overview")

        # Key performance indicators
        kpi_metrics = {
            'Win Rate': {'metric': 'IsWin', 'suffix': '%', 'multiply': 100},
            'Goals/Game': {'metric': 'Us Goals', 'suffix': '', 'multiply': 1},
            'Shot Efficiency': {'metric': 'Shot Efficiency', 'suffix': '%', 'multiply': 1},
            'Time on Attack': {'metric': 'Us TOA', 'suffix': ' min', 'multiply': 1}
        }

        cols = st.columns(1) if st.sidebar.checkbox("Show KPIs Vertically", False) else st.columns(len(kpi_metrics))
        recent_days = daily_metrics.tail(5)
        previous_days = daily_metrics.iloc[-10:-5]

        for col, (name, meta) in zip(cols, kpi_metrics.items()):
            current = recent_days[meta['metric']].mean() * meta['multiply']
            previous = previous_days[meta['metric']].mean() * meta['multiply']
            trend = "↗" if current > previous else "↘"
            trend_class = "trend-up" if current > previous else "trend-down"

            col.markdown(f"""
                <div class="stat-card">
                    <h3>{name}</h3>
                    <div style="font-size: 24px; font-weight: bold;">
                        {current:.1f}{meta['suffix']} {trend}
                    </div>
                    <div class="{trend_class}">
                        {abs(current - previous):.1f}{meta['suffix']} change
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Time series of key metrics
        st.plotly_chart(create_metric_timeline(
            daily_metrics, 'Us TOA', 'Time on Attack (minutes)', 'Viridis'
        ), use_container_width=True)

        st.plotly_chart(create_metric_timeline(
            daily_metrics, 'Us Passing Rate', 'Passing Rate (%)', 'RdYlBu'
        ), use_container_width=True)

    with tabs[1]:
        st.header("Special Teams Analysis")
        st.plotly_chart(create_special_teams_analysis(daily_metrics), use_container_width=True)

        # Additional special teams metrics
        cols = st.columns(1) if st.sidebar.checkbox("Show Metrics Vertically", False) else st.columns(3)
        with cols[0]:
            pp_efficiency = daily_metrics['Power Play Efficiency'].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Power Play Efficiency</h3>
                    <div style="font-size: 24px;">{pp_efficiency:.2f} goals/minute</div>
                </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            avg_penalty_minutes = daily_metrics['Us Penalty Minutes'].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Avg Penalty Minutes</h3>
                    <div style="font-size: 24px;">{avg_penalty_minutes:.1f}</div>
                </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            avg_powerplay_time = daily_metrics['Us Power Play Minutes'].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Avg Power Play Time</h3>
                    <div style="font-size: 24px;">{avg_powerplay_time:.1f} minutes</div>
                </div>
            """, unsafe_allow_html=True)

    with tabs[2]:
        st.header("Detailed Metrics")

        # All available metrics
        available_metrics = [
            ('Us TOA', 'Time on Attack', 'Viridis'),
            ('Us Passing Rate', 'Passing Rate', 'RdYlBu'),
            ('Us Penalty Minutes', 'Penalty Minutes', 'Reds'),
            ('Us Hits', 'Hits', 'Blues'),
            ('Us Faceoffs Won', 'Faceoffs Won', 'Greens'),
            ('Us Power Play Minutes', 'Power Play Time', 'Plasma'),
            ('Shot Efficiency', 'Shot Efficiency', 'Viridis')
        ]

        selected_metrics = st.multiselect(
            "Select metrics to display",
            [m[1] for m in available_metrics],
            default=[m[1] for m in available_metrics[:3]],
            key="metrics_selector"
        )

        for metric, title, colorscale in available_metrics:
            if title in selected_metrics:
                st.plotly_chart(
                    create_metric_timeline(daily_metrics, metric, title, colorscale),
                    use_container_width=True
                )

        # Additional comparisons
        st.header("Additional Comparisons")

        # Us TOA vs. Number of Opponent Players
        fig1 = px.scatter(
            daily_metrics,
            x='# Opponent Players',
            y='Us TOA',
            color='IsWin',
            title='Us Time on Attack vs Number of Opponent Players',
            labels={'# Opponent Players': 'Number of Opponent Players', 'Us TOA': 'Us Time on Attack'},
            template='plotly_dark'
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Us Penalty Minutes vs Us Goals
        fig2 = px.scatter(
            daily_metrics,
            x='Us Penalty Minutes',
            y='Us Goals',
            color='IsWin',
            title='Us Penalty Minutes vs Us Goals',
            labels={'Us Penalty Minutes': 'Us Penalty Minutes', 'Us Goals': 'Us Goals'},
            template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[3]:
        st.header("Player Statistics")

        player = st.selectbox(
            "Select Player",
            ['Nolan', 'Andrew', 'Morgan'],
            key="player_selector"
        )

        # Player's recent performance
        cols = st.columns(1) if st.sidebar.checkbox("Show Player Metrics Vertically", False) else st.columns(3)

        recent_days = daily_metrics.tail(5)
        previous_days = daily_metrics.iloc[-10:-5]

        with cols[0]:
            goals = recent_days[f'{player} Goal'].mean()
            prev_goals = previous_days[f'{player} Goal'].mean()
            trend = "↗" if goals > prev_goals else "↘"
            trend_class = "trend-up" if goals > prev_goals else "trend-down"

            st.markdown(f"""
                <div class="stat-card">
                    <h3>Goals per Game</h3>
                    <div style="font-size: 24px; font-weight: bold;">
                        {goals:.2f} {trend}
                    </div>
                    <div class="{trend_class}">
                        {abs(goals - prev_goals):.2f} change
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            win_rate = recent_days['IsWin'].mean() * 100
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Recent Win Rate</h3>
                    <div style="font-size: 24px; font-weight: bold;">
                        {win_rate:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            scoring_rate = (recent_days[f'{player} Goal'] > 0).mean() * 100
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Recent Scoring Rate</h3>
                    <div style="font-size: 24px; font-weight: bold;">
                        {scoring_rate:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Player's goal timeline
        st.plotly_chart(create_metric_timeline(
            daily_metrics, f'{player} Goal', f"{player}'s Goals per Game", 'RdYlGn'
        ), use_container_width=True)

if __name__ == "__main__":
    main()