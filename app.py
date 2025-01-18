import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import yaml
from scipy import stats

# Page Configuration
st.set_page_config(page_title="NHL Gaming Hub", page_icon="🏒", layout="wide")

# Enhanced CSS
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
    @media only screen and (max-width: 768px) {
        .streak-container, .record-container {
            flex-direction: column;
            align-items: stretch;
        }
        .streak-card, .record-card {
            margin-bottom: 10px;
        }
        .streak-value { font-size: 20px; }
        .stat-card h3 { font-size: 18px; }
        .stat-card div { font-size: 16px; }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_features(yaml_path='model_features.yaml'):
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            features = config.get('features', [])
            return [str(feature).strip() for feature in features] if features else []
    except Exception as e:
        st.error(f"Error loading features: {e}")
        return []

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/nhl_stats.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Convert numeric columns
        numeric_cols = ['Us Num Players', 'Num Opponent Players', 'Us Goals', 'Opponent Goals',
                       'Nolan Goal', 'Andrew Goal', 'Morgan Goal', 'Us Total Shots', 
                       'Opponent Total Shots', 'Us Hits', 'Opponent Hits', 'Us TOA', 
                       'Opponent TOA', 'Us Passing Rate', 'Opponent Passing Rate',
                       'Us Faceoffs Won', 'Opponent Faceoffs Won', 'Us Penalty Minutes',
                       'Opponent Penalty Minutes', 'Us Power Play Minutes', 
                       'Opponent Power Play Minutes', 'Us Shorthanded Goals',
                       'Opponent Shorthanded Goals', 'Season Year']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Derived metrics
        df['IsWin'] = (df['Win / Loss'] == 'Win').astype(int)
        df['Shot Efficiency'] = np.where(df['Us Total Shots'] > 0,
                                       (df['Us Goals'] / df['Us Total Shots'] * 100), 0)
        df['Power Play Efficiency'] = np.where(df['Us Power Play Minutes'] > 0,
                                             df['Us Goals'] / df['Us Power Play Minutes'], 0)
        df['Goal Differential'] = df['Us Goals'] - df['Opponent Goals']

        # Calculate streaks
        streak_cols = ['Win', 'Nolan_Goal', 'Andrew_Goal', 'Morgan_Goal']
        for col in streak_cols:
            counter = 0
            df[f'Games_Since_{col}'] = 0
            for idx, row in df.iterrows():
                if (col == 'Win' and row['IsWin'] == 1) or \
                   (col != 'Win' and row[f'{col.split("_")[0]} Goal'] > 0):
                    counter = 0
                else:
                    counter += 1
                df.at[idx, f'Games_Since_{col}'] = counter

        # Faceoff metrics
        df['Total Faceoffs'] = df['Us Faceoffs Won'] + df['Opponent Faceoffs Won']
        df['Faceoff Win Percentage'] = np.where(df['Total Faceoffs'] > 0,
                                              (df['Us Faceoffs Won'] / df['Total Faceoffs'] * 100), 0)
        
        df['Number of Games Previously Played'] = df.reset_index().index
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_metric_timeline(df, metric, title, color_scale=None):
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

def create_special_teams_analysis(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Power Play Time vs Goals',
            'Penalty Minutes Trend',
            'Power Play Efficiency Trend',
            'Team Performance'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['Us Power Play Minutes'],
            y=df['Us Goals'],
            mode='markers',
            name='PP Time vs Goals',
            marker=dict(
                color=df['IsWin'],
                colorscale='RdYlGn',
                showscale=True
            )
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Us Penalty Minutes'],
            name='Penalty Minutes',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Power Play Efficiency'],
            name='PP Efficiency',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['IsWin'],
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
    if not features:
        st.warning("No features available for correlation analysis.")
        return None, {}

    if target not in df.columns:
        st.error(f"Target variable '{target}' not found in the data.")
        return None, {}

    correlations = {}
    p_values = {}
    for feature in features:
        if feature in df.columns:
            valid_data = df[[feature, target]].dropna()
            if len(valid_data) >= 2:
                corr, p_val = stats.pearsonr(valid_data[feature], valid_data[target])
                if p_val <= significance_level:
                    correlations[feature] = corr
                    p_values[feature] = p_val

    if not correlations:
        return None, {}

    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    corr_df['p-value'] = pd.Series(p_values)
    corr_df = corr_df.dropna().sort_values('Correlation', ascending=False)

    fig = px.bar(
        corr_df,
        x='Correlation',
        y=corr_df.index,
        orientation='h',
        color='Correlation',
        color_continuous_scale='RdYlGn',
        title='Statistically Significant Feature Correlations with Goal Differential',
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

    # Load data
    df = load_data()
    if df.empty:
        st.warning("No data available.")
        return

    # Add Season Year filter in sidebar
    available_years = sorted(df['Season Year'].unique())
    selected_year = st.sidebar.selectbox(
        "Select Season Year",
        options=available_years,
        index=len(available_years)-1
    )

    # Filter data based on selected year
    filtered_df = df[df['Season Year'] == selected_year]
    
    # Calculate metrics based on filtered data
    daily_metrics = filtered_df.copy()
    current_streaks = {
        'Games_Since_Win': int(filtered_df['Games_Since_Win'].iloc[-1]),
        'Games_Since_Nolan_Goal': int(filtered_df['Games_Since_Nolan_Goal'].iloc[-1]),
        'Games_Since_Andrew_Goal': int(filtered_df['Games_Since_Andrew_Goal'].iloc[-1]),
        'Games_Since_Morgan_Goal': int(filtered_df['Games_Since_Morgan_Goal'].iloc[-1])
    }
    total_wins = filtered_df['IsWin'].sum()
    total_losses = len(filtered_df) - total_wins
    total_goals = {
        'Nolan': int(filtered_df['Nolan Goal'].sum()),
        'Andrew': int(filtered_df['Andrew Goal'].sum()),
        'Morgan': int(filtered_df['Morgan Goal'].sum())
    }

    # Display last game date
    last_game_date = filtered_df['Date'].max().strftime('%B %d, %Y')
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

    # Display Total Wins and Losses
    st.markdown(f"""
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

    # Create tabs
    tabs = st.tabs(["📊 Overview", "⚡ Special Teams", "📈 Detailed Metrics", 
                    "🎮 Player Stats", "📉 Goal Difference Prediction", 
                    "🔗 Correlation Explorer"])

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
        recent_days = daily_metrics.tail(5) # Last line for previous sheet: recent_days = daily_metrics.tail(5)
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

        # Timeline plots
        metrics_to_plot = [
            ('Us TOA', 'Time on Attack (minutes)', 'Viridis'),
            ('Us Passing Rate', 'Passing Rate (%)', 'RdYlBu'),
            ('Faceoff Win Percentage', 'Faceoff Win Percentage (%)', 'Portland')
        ]

        for metric, title, color_scale in metrics_to_plot:
            st.plotly_chart(create_metric_timeline(
                daily_metrics, metric, title, color_scale
            ), use_container_width=True)

        # Player Goals Bar Chart
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
        
        special_teams_metrics = {
            'Power Play Efficiency': {'value': daily_metrics['Power Play Efficiency'].mean(), 'suffix': ' goals/minute'},
            'Avg Penalty Minutes': {'value': daily_metrics['Us Penalty Minutes'].mean(), 'suffix': ''},
            'Avg Power Play Time': {'value': daily_metrics['Us Power Play Minutes'].mean(), 'suffix': ' minutes'}
        }

        for col, (name, meta) in zip(cols, special_teams_metrics.items()):
            col.markdown(f"""
                <div class="stat-card">
                    <h3>{name}</h3>
                    <div style="font-size: 24px;">{meta['value']:.2f}{meta['suffix']}</div>
                </div>
            """, unsafe_allow_html=True)

    with tabs[2]:
        st.header("Detailed Metrics")

        available_metrics = [
            ('Us TOA', 'Time on Attack', 'Viridis'),
            ('Us Passing Rate', 'Passing Rate', 'RdYlBu'),
            ('Us Penalty Minutes', 'Penalty Minutes', 'Reds'),
            ('Us Hits', 'Hits', 'Blues'),
            ('Us Faceoffs Won', 'Faceoffs Won', 'Greens'),
            ('Us Power Play Minutes', 'Power Play Time', 'Plasma'),
            ('Shot Efficiency', 'Shot Efficiency', 'Viridis'),
            ('Faceoff Win Percentage', 'Faceoff Win Percentage (%)', 'Portland'),
            ('Number of Games Previously Played', 'Games Played', 'Blues')
        ]

        selected_metrics = st.multiselect(
            "Select metrics to display",
            [m[1] for m in available_metrics],
            default=[m[1] for m in available_metrics[:4]]
        )

        metric_mapping = {title: (metric, colorscale) for metric, title, colorscale in available_metrics}

        for title in selected_metrics:
            metric, colorscale = metric_mapping.get(title, (None, None))
            if metric:
                st.plotly_chart(
                    create_metric_timeline(daily_metrics, metric, title, colorscale),
                    use_container_width=True
                )

        # Additional comparisons
        comparison_plots = [
            {
                'x': 'Num Opponent Players',
                'y': 'Us TOA',
                'title': 'Us Time on Attack vs Number of Opponent Players'
            },
            {
                'x': 'Us Penalty Minutes',
                'y': 'Us Goals',
                'title': 'Us Penalty Minutes vs Us Goals'
            }
        ]

        for plot in comparison_plots:
            fig = px.scatter(
                daily_metrics,
                x=plot['x'],
                y=plot['y'],
                color='IsWin',
                title=plot['title'],
                template='plotly_dark',
                color_continuous_scale='RdYlGn',
                hover_data=['Date']
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.header("🎮 Player Statistics")

        player = st.selectbox(
            "Select Player",
            ['Nolan', 'Andrew', 'Morgan']
        )

        show_player_metrics_vertically = st.sidebar.checkbox("Show Player Metrics Vertically", False)
        cols = st.columns(1) if show_player_metrics_vertically else st.columns(3)

        recent_days = daily_metrics.tail(5)
        previous_days = daily_metrics.iloc[-10:-5]

        # Player stats in columns
        stats = [
            {
                'title': 'Goals per Game',
                'current': recent_days[f'{player} Goal'].mean(),
                'previous': previous_days[f'{player} Goal'].mean()
            },
            {
                'title': 'Recent Win Rate',
                'value': recent_days['IsWin'].mean() * 100
            },
            {
                'title': 'Recent Scoring Rate',
                'value': (recent_days[f'{player} Goal'] > 0).mean() * 100
            }
        ]

        for col, stat in zip(cols, stats):
            if 'previous' in stat:
                trend = "↗" if stat['current'] > stat['previous'] else "↘"
                trend_class = "trend-up" if stat['current'] > stat['previous'] else "trend-down"
                col.markdown(f"""
                    <div class="stat-card">
                        <h3>{stat['title']}</h3>
                        <div style="font-size: 24px; font-weight: bold;">
                            {stat['current']:.2f} {trend}
                        </div>
                        <div class="{trend_class}">
                            {abs(stat['current'] - stat['previous']):.2f} change
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                col.markdown(f"""
                    <div class="stat-card">
                        <h3>{stat['title']}</h3>
                        <div style="font-size: 24px; font-weight: bold;">
                            {stat['value']:.1f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Total goals display
        st.markdown(f"""
            <div class="stat-card">
                <h3>Total Goals</h3>
                <div style="font-size: 24px; font-weight: bold;">
                    {total_goals.get(player, 0)} goals
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Player's goal timeline
        st.plotly_chart(create_metric_timeline(
            daily_metrics, f'{player} Goal', f"{player}'s Goals per Game", 'RdYlGn'
        ), use_container_width=True)

    with tabs[4]:
        st.header("📉 Goal Difference Prediction")
        features = load_model_features()
        #features = None

        if features:
            st.subheader("Feature Correlations with Goal Differential")
            corr_df, correlations = create_correlation_plot(daily_metrics, features)

            if corr_df is not None and not corr_df.empty:
                st.subheader("Statistically Significant Correlation Coefficients")
                st.dataframe(corr_df.style.format({
                    "Correlation": "{:.2f}",
                    "p-value": "{:.3f}"
                }))

                st.subheader("Stats Insights")
                for feature, corr in correlations.items():
                    direction = 'positive' if corr > 0 else 'negative'
                    tendency = 'increase' if corr > 0 else 'decrease'
                    st.markdown(f"- **{feature}** has a **{direction}** correlation with goal differential. "
                              f"As **{feature}** increases, the goal differential tends to **{tendency}**.")

    with tabs[5]:
        st.header("🔗 Correlation Explorer")

        numeric_columns = [col for col in daily_metrics.columns 
                         if np.issubdtype(daily_metrics[col].dtype, np.number)]

        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Select X variable", options=numeric_columns, key="x_var")
        with col2:
            y_var = st.selectbox("Select Y variable", options=numeric_columns, key="y_var")

        if x_var and y_var:
            fig = px.scatter(
                daily_metrics,
                x=x_var,
                y=y_var,
                trendline="ols",
                title=f"Correlation between {x_var} and {y_var}",
                template='plotly_dark'
            )

            valid_data = daily_metrics[[x_var, y_var]].dropna()
            if len(valid_data) > 1:
                try:
                    corr, p_val = stats.pearsonr(valid_data[x_var], valid_data[y_var])
                    corr_text = f"**Pearson correlation coefficient:** {corr:.2f} (p-value: {p_val:.3f})"
                except:
                    print('failed')
            else:
                try:
                    corr_text = "**Not enough data to calculate correlation.**"
                except:
                    print('failed')

            #st.plotly_chart(fig, use_container_width=True)
            #st.markdown(corr_text)

if __name__ == "__main__":
    main()