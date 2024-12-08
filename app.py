import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import yaml  # Ensure PyYAML is installed
from scipy import stats  # For statistical significance

# Page Configuration
st.set_page_config(page_title="NHL Gaming Hub", page_icon="🏒", layout="wide")

# Enhanced CSS for Mobile-Friendly Layout
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

    /* Mobile-Friendly Adjustments */
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
def load_model_features(yaml_path='model_features.yaml'):
    """
    Load model features from a YAML configuration file.
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            features = config.get('features', [])
            if not features:
                st.warning("No features found in the YAML configuration.")
            # Strip leading/trailing spaces
            features = [str(feature).strip() for feature in features]
            return features
    except FileNotFoundError:
        st.error(f"YAML configuration file '{yaml_path}' not found.")
        return []
    except yaml.YAMLError as exc:
        st.error(f"Error parsing YAML file: {exc}")
        return []


@st.cache_data
def load_data():
    """
    Load and process NHL stats data from a CSV file.
    """
    try:
        df = pd.read_csv('data/nhl_stats.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')  # Ensure data is sorted chronologically

        # All numeric columns
        numeric_cols = [
            'Us Num Players', 'Num Opponent Players', 'Us Goals', 'Opponent Goals',
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

        # Derived metrics
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

        # Initialize "Games Since" metrics
        df['Games_Since_Win'] = 0
        df['Games_Since_Nolan_Goal'] = 0
        df['Games_Since_Andrew_Goal'] = 0
        df['Games_Since_Morgan_Goal'] = 0

        # Counters for "Games Since" metrics
        games_since_win = 0
        games_since_nolan = 0
        games_since_andrew = 0
        games_since_morgan = 0

        # Iterate through the dataframe in chronological order
        for idx, row in df.iterrows():
            # Update Games_Since_Win
            if row['IsWin'] == 1:
                games_since_win = 0
            else:
                games_since_win += 1
            df.at[idx, 'Games_Since_Win'] = games_since_win

            # Update Games_Since_Nolan_Goal
            if row['Nolan Goal'] > 0:
                games_since_nolan = 0
            else:
                games_since_nolan += 1
            df.at[idx, 'Games_Since_Nolan_Goal'] = games_since_nolan

            # Update Games_Since_Andrew_Goal
            if row['Andrew Goal'] > 0:
                games_since_andrew = 0
            else:
                games_since_andrew += 1
            df.at[idx, 'Games_Since_Andrew_Goal'] = games_since_andrew

            # Update Games_Since_Morgan_Goal
            if row['Morgan Goal'] > 0:
                games_since_morgan = 0
            else:
                games_since_morgan += 1
            df.at[idx, 'Games_Since_Morgan_Goal'] = games_since_morgan

        # Faceoff calculations
        df['Total Faceoffs'] = df['Us Faceoffs Won'] + df['Opponent Faceoffs Won']
        df['Faceoff Win Percentage'] = np.where(
            df['Total Faceoffs'] > 0,
            (df['Us Faceoffs Won'] / df['Total Faceoffs'] * 100),
            0
        )

        # Number of Games Previously Played
        df['Number of Games Previously Played'] = df.reset_index().index

        # Load features from YAML
        features = load_model_features('model_features.yaml')

        # Ensure all features from YAML are included in metrics_to_aggregate
        metrics_to_aggregate = [
            'Us TOA', 'Us Passing Rate', 'Us Penalty Minutes',
            'Us Hits', 'Opponent Hits',
            'Us Faceoffs Won', 'Opponent Faceoffs Won',
            'Faceoff Win Percentage',
            'Us Power Play Minutes',
            'Shot Efficiency', 'Power Play Efficiency', 'IsWin',
            'Us Goals', 'Goal Differential', 'Nolan Goal',
            'Andrew Goal', 'Morgan Goal', 'Games_Since_Win',
            'Games_Since_Nolan_Goal', 'Games_Since_Andrew_Goal',
            'Games_Since_Morgan_Goal', 'Num Opponent Players',
            'Number of Games Previously Played'
        ]

        for feature in features:
            if feature not in metrics_to_aggregate and feature in df.columns:
                metrics_to_aggregate.append(feature)

        # Aggregate metrics by Date
        daily_metrics = df.groupby('Date')[metrics_to_aggregate].mean().reset_index()

        # Current streaks (latest values)
        current_streaks = {
            'Games_Since_Win': int(df['Games_Since_Win'].iloc[-1]),
            'Games_Since_Nolan_Goal': int(df['Games_Since_Nolan_Goal'].iloc[-1]),
            'Games_Since_Andrew_Goal': int(df['Games_Since_Andrew_Goal'].iloc[-1]),
            'Games_Since_Morgan_Goal': int(df['Games_Since_Morgan_Goal'].iloc[-1])
        }

        # Total wins/losses
        total_wins = df['IsWin'].sum()
        total_losses = len(df) - total_wins

        # Total goals per player
        total_goals = {
            'Nolan': int(df['Nolan Goal'].sum()),
            'Andrew': int(df['Andrew Goal'].sum()),
            'Morgan': int(df['Morgan Goal'].sum())
        }

        return daily_metrics, current_streaks, total_wins, total_losses, total_goals, df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), {}, 0, 0, {}, pd.DataFrame()


def create_metric_timeline(df, metric, title, color_scale=None):
    """
    Create a time series plot for a given metric.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[metric],
        name='Session Average',
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
    """
    Create plots for special teams analysis.
    """
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


def create_correlation_plot(df, features, target='Goal Differential', significance_level=0.05):
    """
    Create a correlation plot for statistically significant features.
    Returns the correlation DataFrame and a dictionary of correlations.
    """
    if not features:
        st.warning("No features available for correlation analysis.")
        return None, {}

    # Ensure target exists in the dataframe
    if target not in df.columns:
        st.error(f"Target variable '{target}' not found in the data.")
        return None, {}

    # Calculate correlations and p-values
    correlations = {}
    p_values = {}
    for feature in features:
        if feature in df.columns:
            valid_data = df[[feature, target]].dropna()
            if len(valid_data) < 2:
                st.warning(f"Not enough data to compute correlation for '{feature}'.")
                continue
            corr, p_val = stats.pearsonr(valid_data[feature], valid_data[target])
            if p_val <= significance_level:
                correlations[feature] = corr
                p_values[feature] = p_val
        else:
            st.warning(f"Feature '{feature}' not found in the data and will be skipped.")

    if not correlations:
        st.warning("No statistically significant correlations found.")
        return None, {}

    # Convert to DataFrame
    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    corr_df['p-value'] = pd.Series(p_values)
    corr_df = corr_df.dropna().sort_values('Correlation', ascending=False)

    # Plotting
    fig = px.bar(
        corr_df,
        x='Correlation',
        y=corr_df.index,
        orientation='h',
        color='Correlation',
        color_continuous_scale='RdYlGn',
        title='Statistically Significant Feature Correlations with Goal Differential',
        labels={'Correlation': 'Correlation Coefficient', 'index': 'Features'},
        template='plotly_dark'
    )

    fig.update_layout(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(automargin=True),
        margin=dict(l=150, r=50, t=100, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)
    return corr_df, correlations


def main():
    st.title("🏒 NHL Gaming Analytics Hub")

    # Load Data
    daily_metrics, current_streaks, total_wins, total_losses, total_goals, full_df = load_data()
    if daily_metrics.empty:
        st.warning("No data available.")
        return

    # Add the Last Game Date Message at the Top
    try:
        last_game_date = daily_metrics['Date'].max().strftime('%B %d, %Y')
        st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.1); 
                        border-radius: 10px; 
                        padding: 10px; 
                        text-align: center; 
                        color: white; 
                        margin-bottom: 20px;">
                <h3>Last Game Played on: {last_game_date}</h3>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying last game date: {e}")

    # Display Total Wins and Losses
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
                Total Wins: {int(total_wins)}
            </div>
            <div class="record-card">
                Total Losses: {int(total_losses)}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Streak Metrics
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

    # Tabs
    tabs = st.tabs(["📊 Overview", "⚡ Special Teams", "📈 Detailed Metrics", "🎮 Player Stats", "📉 Goal Difference Prediction"])

    with tabs[0]:
        st.header("Team Performance Overview")

        # KPIs
        kpi_metrics = {
            'Win Rate': {'metric': 'IsWin', 'suffix': '%', 'multiply': 100},
            'Goals/Game': {'metric': 'Us Goals', 'suffix': '', 'multiply': 1},
            'Shot Efficiency': {'metric': 'Shot Efficiency', 'suffix': '%', 'multiply': 1},
            'Time on Attack': {'metric': 'Us TOA', 'suffix': ' min', 'multiply': 1}
        }

        show_kpis_vertically = st.sidebar.checkbox("Show KPIs Vertically", False)
        cols = st.columns(1) if show_kpis_vertically else st.columns(len(kpi_metrics))
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

        # Time Series of Key Metrics
        st.plotly_chart(create_metric_timeline(
            daily_metrics, 'Us TOA', 'Time on Attack (minutes)', 'Viridis'
        ), use_container_width=True)

        st.plotly_chart(create_metric_timeline(
            daily_metrics, 'Us Passing Rate', 'Passing Rate (%)', 'RdYlBu'
        ), use_container_width=True)

        st.plotly_chart(create_metric_timeline(
            daily_metrics, 'Faceoff Win Percentage', 'Faceoff Win Percentage (%)', 'Portland'
        ), use_container_width=True)

        # Bar Chart for Total Goals per Player
        st.subheader("Total Goals Scored by Each Player")
        goals_df = pd.DataFrame({
            'Player': list(total_goals.keys()),
            'Total Goals': list(total_goals.values())
        })

        fig_goals = px.bar(
            goals_df,
            x='Player',
            y='Total Goals',
            color='Player',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title='Total Goals Scored by Each Player',
            template='plotly_dark',
            text='Total Goals'
        )

        fig_goals.update_traces(texttemplate='%{text}', textposition='outside')
        fig_goals.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            yaxis=dict(title='Total Goals'),
            xaxis=dict(title='Player'),
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40)
        )

        st.plotly_chart(fig_goals, use_container_width=True)

    with tabs[1]:
        st.header("Special Teams Analysis")
        st.plotly_chart(create_special_teams_analysis(daily_metrics), use_container_width=True)

        show_metrics_vertically = st.sidebar.checkbox("Show Special Teams Metrics Vertically", False)
        cols = st.columns(1) if show_metrics_vertically else st.columns(3)
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

        # All Available Metrics
        available_metrics = [
            ('Us TOA', 'Time on Attack', 'Viridis'),
            ('Us Passing Rate', 'Passing Rate', 'RdYlBu'),
            ('Us Penalty Minutes', 'Penalty Minutes', 'Reds'),
            ('Us Hits', 'Hits', 'Blues'),
            ('Us Faceoffs Won', 'Faceoffs Won', 'Greens'),
            ('Us Power Play Minutes', 'Power Play Time', 'Plasma'),
            ('Shot Efficiency', 'Shot Efficiency', 'Viridis'),
            ('Faceoff Win Percentage', 'Faceoff Win Percentage (%)', 'Portland'),
            ('Number of Games Previously Played', 'Number of Games Previously Played', 'Blues')
        ]

        selected_metrics = st.multiselect(
            "Select metrics to display",
            [m[1] for m in available_metrics],
            default=[m[1] for m in available_metrics[:4]],
            key="metrics_selector"
        )

        metric_mapping = {title: (metric, colorscale) for metric, title, colorscale in available_metrics}

        for title in selected_metrics:
            metric, colorscale = metric_mapping.get(title, (None, None))
            if metric:
                st.plotly_chart(
                    create_metric_timeline(daily_metrics, metric, title, colorscale),
                    use_container_width=True
                )

        # Additional Comparisons
        st.header("Additional Comparisons")

        # Us TOA vs. Number of Opponent Players
        fig1 = px.scatter(
            daily_metrics,
            x='Num Opponent Players',
            y='Us TOA',
            color='IsWin',
            title='Us Time on Attack vs Number of Opponent Players',
            labels={'Num Opponent Players': 'Number of Opponent Players', 'Us TOA': 'Us Time on Attack'},
            template='plotly_dark',
            color_continuous_scale='RdYlGn',
            hover_data=['Date']
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
            template='plotly_dark',
            color_continuous_scale='RdYlGn',
            hover_data=['Date']
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[3]:
        st.header("🎮 Player Statistics")

        player = st.selectbox(
            "Select Player",
            ['Nolan', 'Andrew', 'Morgan'],
            key="player_selector"
        )

        # Player's Recent Performance
        show_player_metrics_vertically = st.sidebar.checkbox("Show Player Metrics Vertically", False)
        cols = st.columns(1) if show_player_metrics_vertically else st.columns(3)

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

        # Display Total Goals for the Selected Player
        st.markdown(f"""
            <div class="stat-card">
                <h3>Total Goals</h3>
                <div style="font-size: 24px; font-weight: bold;">
                    {total_goals.get(player, 0)} goals
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Player's Goal Timeline
        st.plotly_chart(create_metric_timeline(
            daily_metrics, f'{player} Goal', f"{player}'s Goals per Game", 'RdYlGn'
        ), use_container_width=True)

    with tabs[4]:
        st.header("📉 Goal Difference Prediction")

        # Load features from YAML file
        features = load_model_features('model_features.yaml')

        if features:
            st.subheader("Feature Correlations with Goal Differential")

            # Create the correlation plot
            corr_df, correlations = create_correlation_plot(daily_metrics, features, target='Goal Differential')

            if corr_df is not None and not corr_df.empty:
                # Display correlation values as a table
                st.subheader("Statistically Significant Correlation Coefficients")
                st.dataframe(corr_df.style.format({"Correlation": "{:.2f}", "p-value": "{:.3f}"}))

                # Business-Friendly Interpretations
                st.subheader("Stats Insights")
                for feature, corr in correlations.items():
                    if corr > 0:
                        direction = 'positive'
                        tendency = 'increase'
                    else:
                        direction = 'negative'
                        tendency = 'decrease'
                    st.markdown(f"- **{feature}** has a **{direction}** correlation with goal differential. As **{feature}** increases, the goal differential tends to **{tendency}**. If goal differential decreases, this means that the opponent scores more goals. If the differential is increasing, it means we are scoring more goals. Decrease = greater opponent lead, Increase = Greater us lead")
            else:
                st.warning("No statistically significant correlations to display.")
        else:
            st.warning("No features available for analysis. Please check the YAML configuration.")


if __name__ == "__main__":
    main()