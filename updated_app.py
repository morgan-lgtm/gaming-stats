import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

# Page Configuration
st.set_page_config(page_title="NHL Gaming Hub", page_icon="🏒", layout="wide")

# Basic CSS for clean look
st.markdown("""
    <style>
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/nhl_stats.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add game sequence numbers
        df['Game_Sequence'] = range(1, len(df) + 1)
        df['Season_Game_Number'] = df.groupby('Season Year').cumcount() + 1
        
        # Convert Season Year to category
        df['Season Year'] = df['Season Year'].astype(str).astype('category')
        
        # Convert numeric columns
        numeric_cols = [
            'Us Goals', 'Opponent Goals', 'Us Total Shots', 'Opponent Total Shots',
            'Us Hits', 'Opponent Hits', 'Us TOA', 'Opponent TOA', 'Us Passing Rate',
            'Opponent Passing Rate', 'Us Faceoffs Won', 'Opponent Faceoffs Won',
            'Us Penalty Minutes', 'Opponent Penalty Minutes', 'Us Power Play Minutes',
            'Opponent Power Play Minutes', 'Nolan Goal', 'Andrew Goal', 'Morgan Goal'
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate key metrics
        df['IsWin'] = (df['Win / Loss'] == 'Win').astype(int)
        df['Shot Efficiency'] = (df['Us Goals'] / df['Us Total Shots'] * 100).fillna(0)
        df['Goal Differential'] = df['Us Goals'] - df['Opponent Goals']
        df['Power Play Efficiency'] = (df['Us Goals'] / df['Us Power Play Minutes']).fillna(0)
        df['Faceoff Win Rate'] = (
            df['Us Faceoffs Won'] / (df['Us Faceoffs Won'] + df['Opponent Faceoffs Won']) * 100
        ).fillna(0)
        
        # Calculate rolling averages (5-game window) within each season
        df = df.sort_values(['Season Year', 'Game_Sequence'])
        seasons = df['Season Year'].unique()
        
        for season in seasons:
            season_mask = df['Season Year'] == season
            season_df = df[season_mask]
            
            df.loc[season_mask, 'GoalDiff_Rolling'] = season_df['Goal Differential'].rolling(5, min_periods=1).mean()
            df.loc[season_mask, 'ShotEff_Rolling'] = season_df['Shot Efficiency'].rolling(5, min_periods=1).mean()
            df.loc[season_mask, 'PowerPlay_Rolling'] = season_df['Power Play Efficiency'].rolling(5, min_periods=1).mean()
            df.loc[season_mask, 'FaceoffWin_Rolling'] = season_df['Faceoff Win Rate'].rolling(5, min_periods=1).mean()
            df.loc[season_mask, 'Goals_Rolling'] = season_df['Us Goals'].rolling(5, min_periods=1).mean()
            
            for player in ['Nolan', 'Andrew', 'Morgan']:
                df.loc[season_mask, f'{player}Goals_Rolling'] = season_df[f'{player} Goal'].rolling(5, min_periods=1).mean()
                df.loc[season_mask, f'{player}_Goal_Rate'] = season_df[f'{player} Goal'] / season_df['Us Goals']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_metric_card(title, value, delta=None):
    delta_html = ""
    if delta is not None:
        color = 'green' if delta > 0 else 'red'
        delta_html = f"<div style='color: {color}'>{delta:+.2f}</div>"
    
    return f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <div style='font-size: 24px'>{value:.2f}</div>
            {delta_html}
        </div>
    """

def plot_trend(df, metric, rolling_metric, title, show_individual=True):
    fig = go.Figure()
    
    if show_individual:
        fig.add_trace(go.Scatter(
            x=df['Season_Game_Number'],
            y=df[metric],
            name='Per Game',
            mode='markers',
            marker=dict(size=8, color='lightblue'),
            hovertemplate=f'Game {{x}}<br>{metric}: {{y:.2f}}<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=df['Season_Game_Number'],
        y=df[rolling_metric],
        name='5-Game Average',
        line=dict(width=3, color='yellow'),
        hovertemplate='Game %{x}<br>5-Game Avg: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Game Number in Season",
        yaxis_title=metric,
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

def season_comparison(df):
    """ Team-wide season performance comparison (Win Rate, Goals/Game, etc.). """
    seasons = sorted(df['Season Year'].unique())
    
    comparison_metrics = {}
    for season in seasons:
        season_df = df[df['Season Year'] == season]
        total_games = len(season_df)
        wins = season_df['IsWin'].sum()
        
        comparison_metrics[season] = {
            'Win Rate': (wins / total_games) * 100 if total_games > 0 else 0,
            'Goals per Game': season_df['Us Goals'].mean(),
            'Shot Efficiency': season_df['Shot Efficiency'].mean(),
            'Power Play Efficiency': season_df['Power Play Efficiency'].mean()
        }
    
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=['Win Rate (%)', 'Goals per Game',
                        'Shot Efficiency (%)', 'Power Play Efficiency']
    )
    
    metrics = ['Win Rate', 'Goals per Game', 'Shot Efficiency', 'Power Play Efficiency']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, (row, col) in zip(metrics, positions):
        values = [comparison_metrics[season][metric] for season in seasons]
        fig.add_trace(
            go.Bar(x=seasons, y=values, name=metric),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        title_text="Season Performance Comparison",
        showlegend=False,
        template='plotly_dark'
    )
    
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=12, color='white')
    
    return fig

def compare_player_seasons(df, player='Nolan'):
    """
    Bar-chart comparison for a single player's performance across all seasons:
    - Goals/Game
    - Total Goals
    - Goal Rate (% of team's goals)
    """
    players = ['Nolan', 'Andrew', 'Morgan']
    if player not in players:
        raise ValueError(f"Player '{player}' not recognized. Must be one of {players}.")
    
    seasons = sorted(df['Season Year'].unique())
    
    # Collect metrics per season
    metric_data = {
        'Goals per Game': [],
        'Total Goals': [],
        'Goal Rate (%)': []
    }
    
    for season in seasons:
        season_df = df[df['Season Year'] == season]
        total_us_goals = season_df['Us Goals'].sum()
        player_goals = season_df[f'{player} Goal'].sum()
        
        goals_per_game = season_df[f'{player} Goal'].mean()
        total_goals = player_goals
        goal_rate_pct = (player_goals / total_us_goals * 100) if total_us_goals > 0 else 0
        
        metric_data['Goals per Game'].append(goals_per_game)
        metric_data['Total Goals'].append(total_goals)
        metric_data['Goal Rate (%)'].append(goal_rate_pct)
    
    # Create a 3-row subplot
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Goals per Game", "Total Goals", "Goal Rate (%)"],
        vertical_spacing=0.1
    )
    
    # 1) Goals per Game
    fig.add_trace(
        go.Bar(
            x=seasons,
            y=metric_data['Goals per Game'],
            marker_color='rgb(31,119,180)',
            name='Goals/Game'
        ),
        row=1, col=1
    )
    
    # 2) Total Goals
    fig.add_trace(
        go.Bar(
            x=seasons,
            y=metric_data['Total Goals'],
            marker_color='rgb(255,127,14)',
            name='Total Goals'
        ),
        row=2, col=1
    )
    
    # 3) Goal Rate (%)
    fig.add_trace(
        go.Bar(
            x=seasons,
            y=metric_data['Goal Rate (%)'],
            marker_color='rgb(44,160,44)',
            name='Goal Rate (%)'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text=f"{player}'s Performance Across Seasons",
        template='plotly_dark',
        showlegend=False  # Single player => no separate legend needed
    )
    
    fig.update_yaxes(title_text="Goals/Game", row=1, col=1)
    fig.update_yaxes(title_text="Total Goals", row=2, col=1)
    fig.update_yaxes(title_text="Goal Rate (%)", row=3, col=1)
    
    return fig

def main():
    st.title("🏒 NHL Gaming Analytics Hub")
    
    # Load data
    df = load_data()
    if df.empty:
        st.warning("No data available.")
        return

    # Season selector
    selected_year = st.sidebar.selectbox(
        "Select Season Year",
        options=sorted(df['Season Year'].unique()),
        index=len(df['Season Year'].unique()) - 1
    )
    
    # Filter data by selected season
    filtered_df = df[df['Season Year'] == selected_year]
    
    # Show/hide individual game points
    show_individual_games = st.sidebar.checkbox("Show Individual Game Data Points", True)
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview",
        "👥 Player Analysis",
        "⚡ Special Teams",
        "📈 Season Comparison",
        "🔍 Metric Explorer"
    ])
    
    # ---- 1) OVERVIEW ----
    with tabs[0]:
        st.header("Team Performance Overview")
        
        recent_games = filtered_df.tail(5)
        previous_games = (
            filtered_df.iloc[-10:-5] 
            if len(filtered_df) >= 10 else filtered_df.head(5)
        )
        
        metrics = {
            'Win Rate': (
                recent_games['IsWin'].mean() * 100, 
                previous_games['IsWin'].mean() * 100
            ),
            'Goals/Game': (
                recent_games['Us Goals'].mean(),
                previous_games['Us Goals'].mean()
            ),
            'Shot Efficiency': (
                recent_games['Shot Efficiency'].mean(),
                previous_games['Shot Efficiency'].mean()
            ),
            'Power Play Efficiency': (
                recent_games['Power Play Efficiency'].mean(),
                previous_games['Power Play Efficiency'].mean()
            )
        }
        
        cols = st.columns(4)
        for col, (metric, (current, previous)) in zip(cols, metrics.items()):
            col.markdown(
                create_metric_card(metric, current, current - previous),
                unsafe_allow_html=True
            )
        
        # Performance trends
        st.plotly_chart(
            plot_trend(
                filtered_df,
                'Goal Differential',
                'GoalDiff_Rolling',
                'Goal Differential Trend',
                show_individual_games
            ),
            use_container_width=True
        )
        
        st.plotly_chart(
            plot_trend(
                filtered_df,
                'Shot Efficiency',
                'ShotEff_Rolling',
                'Shot Efficiency Trend',
                show_individual_games
            ),
            use_container_width=True
        )
    
    # ---- 2) PLAYER ANALYSIS ----
    with tabs[1]:
        st.header("Player Performance")
        
        # Select a single season, single player breakdown
        player = st.selectbox("Select Player", ['Nolan', 'Andrew', 'Morgan'])
        
        # Compare recent vs. previous for that player in the chosen season
        recent_player = filtered_df.tail(5)
        previous_player = (
            filtered_df.iloc[-10:-5] 
            if len(filtered_df) >= 10 else filtered_df.head(5)
        )
        
        player_metrics = {
            'Recent Goals/Game': (
                recent_player[f'{player} Goal'].mean(),
                previous_player[f'{player} Goal'].mean()
            ),
            'Season Goals': (
                filtered_df[f'{player} Goal'].sum(),
                None
            ),
            'Goal Rate': (
                recent_player[f'{player}_Goal_Rate'].mean() * 100,
                previous_player[f'{player}_Goal_Rate'].mean() * 100
            )
        }
        
        cols = st.columns(3)
        for col, (metric, (current, previous)) in zip(cols, player_metrics.items()):
            delta = (current - previous) if (previous is not None) else None
            col.markdown(
                create_metric_card(metric, current, delta),
                unsafe_allow_html=True
            )
        
        # Single-season trend for the player's goals
        st.plotly_chart(
            plot_trend(
                filtered_df,
                f'{player} Goal',
                f'{player}Goals_Rolling',
                f"{player}'s Goal Trend",
                show_individual_games
            ),
            use_container_width=True
        )
        
        # New: Compare this player across *all* seasons
        st.subheader(f"{player} Across All Seasons")
        multi_season_fig = compare_player_seasons(df, player)
        st.plotly_chart(multi_season_fig, use_container_width=True)
    
    # ---- 3) SPECIAL TEAMS ----
    with tabs[2]:
        st.header("Special Teams Analysis")
        
        recent_games = filtered_df.tail(5)
        previous_games = (
            filtered_df.iloc[-10:-5] 
            if len(filtered_df) >= 10 else filtered_df.head(5)
        )
        
        special_teams_metrics = {
            'Power Play Efficiency': (
                recent_games['Power Play Efficiency'].mean(),
                previous_games['Power Play Efficiency'].mean()
            ),
            'Penalty Minutes/Game': (
                recent_games['Us Penalty Minutes'].mean(),
                previous_games['Us Penalty Minutes'].mean()
            ),
            'PP Minutes/Game': (
                recent_games['Us Power Play Minutes'].mean(),
                previous_games['Us Power Play Minutes'].mean()
            )
        }
        
        cols = st.columns(3)
        for col, (metric, (current, previous)) in zip(cols, special_teams_metrics.items()):
            col.markdown(
                create_metric_card(metric, current, current - previous),
                unsafe_allow_html=True
            )
        
        st.plotly_chart(
            plot_trend(
                filtered_df,
                'Power Play Efficiency',
                'PowerPlay_Rolling',
                'Power Play Efficiency Trend',
                show_individual_games
            ),
            use_container_width=True
        )
    
    # ---- 4) SEASON COMPARISON (TEAM) ----
    with tabs[3]:
        st.header("Season Performance Comparison")
        
        # Overall season comparison for the team
        st.plotly_chart(season_comparison(df), use_container_width=True)
        # (Removed the old multi-player season comparison function call)
    
    # ---- 5) METRIC EXPLORER ----
    with tabs[4]:
        st.header("Metric Explorer")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        col1, col2 = st.columns(2)
        x_var = col1.selectbox("Select X variable", options=numeric_cols)
        y_var = col2.selectbox("Select Y variable", options=numeric_cols)
        
        if x_var and y_var:
            # Build scatter plot with optional regression line
            plot_df = filtered_df[[x_var, y_var]].dropna()
            if len(plot_df) >= 2:
                try:
                    fig = px.scatter(
                        plot_df,
                        x='Season_Game_Number' if x_var == 'Game_Sequence' else x_var,
                        y=y_var,
                        trendline="ols",
                        title=f"Correlation between {x_var} and {y_var}"
                    )
                    fig.update_layout(
                        template='plotly_dark',
                        xaxis_title=x_var,
                        yaxis_title=y_var,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
                
                # Correlation stats
                valid_data = filtered_df[[x_var, y_var]].dropna()
                if len(valid_data) >= 2:
                    try:
                        corr, p_val = stats.pearsonr(valid_data[x_var], valid_data[y_var])
                        st.markdown(f"**Correlation:** {corr:.2f} (p-value: {p_val:.3f})")
                    except Exception as e:
                        st.warning(f"Could not calculate correlation: {str(e)}")
                else:
                    st.warning("Not enough data points for correlation (need at least 2).")
            else:
                st.warning("Not enough valid data points to create scatter plot.")

if __name__ == "__main__":
    main()