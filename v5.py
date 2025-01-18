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
    """Load model features from a YAML configuration file."""
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            features = config.get('features', [])
            if not features:
                st.warning("No features found in the YAML configuration.")
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
    """Load and process NHL stats data from a CSV file."""
    try:
        df = pd.read_csv('data/nhl_stats.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Convert Season Year to integer
        df['Season Year'] = pd.to_numeric(df['Season Year'], errors='coerce').astype('Int64')

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

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def filter_data_by_season(df, selected_season):
    """Filter the dataframe by selected season year."""
    if selected_season == "All Seasons":
        return df
    return df[df['Season Year'] == int(selected_season)]

def process_data(df):
    """Process the filtered dataframe to calculate all required metrics."""
    daily_metrics = df.copy()

    # Derived metrics
    daily_metrics['IsWin'] = (daily_metrics['Win / Loss'] == 'Win').astype(int)
    daily_metrics['Shot Efficiency'] = np.where(
        daily_metrics['Us Total Shots'] > 0,
        (daily_metrics['Us Goals'] / daily_metrics['Us Total Shots'] * 100),
        0
    )
    daily_metrics['Power Play Efficiency'] = np.where(
        daily_metrics['Us Power Play Minutes'] > 0,
        daily_metrics['Us Goals'] / daily_metrics['Us Power Play Minutes'],
        0
    )
    daily_metrics['Goal Differential'] = daily_metrics['Us Goals'] - daily_metrics['Opponent Goals']

    # Initialize "Games Since" metrics
    daily_metrics['Games_Since_Win'] = 0
    daily_metrics['Games_Since_Nolan_Goal'] = 0
    daily_metrics['Games_Since_Andrew_Goal'] = 0
    daily_metrics['Games_Since_Morgan_Goal'] = 0

    # Calculate streaks
    games_since_win = 0
    games_since_nolan = 0
    games_since_andrew = 0
    games_since_morgan = 0

    for idx, row in daily_metrics.iterrows():
        if row['IsWin'] == 1:
            games_since_win = 0
        else:
            games_since_win += 1
        daily_metrics.at[idx, 'Games_Since_Win'] = games_since_win

        if row['Nolan Goal'] > 0:
            games_since_nolan = 0
        else:
            games_since_nolan += 1
        daily_metrics.at[idx, 'Games_Since_Nolan_Goal'] = games_since_nolan

        if row['Andrew Goal'] > 0:
            games_since_andrew = 0
        else:
            games_since_andrew += 1
        daily_metrics.at[idx, 'Games_Since_Andrew_Goal'] = games_since_andrew

        if row['Morgan Goal'] > 0:
            games_since_morgan = 0
        else:
            games_since_morgan += 1
        daily_metrics.at[idx, 'Games_Since_Morgan_Goal'] = games_since_morgan

    # Faceoff calculations
    daily_metrics['Total Faceoffs'] = daily_metrics['Us Faceoffs Won'] + daily_metrics['Opponent Faceoffs Won']
    daily_metrics['Faceoff Win Percentage'] = np.where(
        daily_metrics['Total Faceoffs'] > 0,
        (daily_metrics['Us Faceoffs Won'] / daily_metrics['Total Faceoffs'] * 100),
        0
    )

    # Current streaks
    current_streaks = {
        'Games_Since_Win': int(daily_metrics['Games_Since_Win'].iloc[-1]),
        'Games_Since_Nolan_Goal': int(daily_metrics['Games_Since_Nolan_Goal'].iloc[-1]),
        'Games_Since_Andrew_Goal': int(daily_metrics['Games_Since_Andrew_Goal'].iloc[-1]),
        'Games_Since_Morgan_Goal': int(daily_metrics['Games_Since_Morgan_Goal'].iloc[-1])
    }

    # Calculate totals
    total_wins = daily_metrics['IsWin'].sum()
    total_losses = len(daily_metrics) - total_wins
    
    total_goals = {
        'Nolan': int(daily_metrics['Nolan Goal'].sum()),
        'Andrew': int(daily_metrics['Andrew Goal'].sum()),
        'Morgan': int(daily_metrics['Morgan Goal'].sum())
    }

    return daily_metrics, current_streaks, total_wins, total_losses, total_goals

def create_metric_timeline(df, metric, title, color_scale=None):
    """Create a time series plot for a given metric."""
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
    """Create plots for special teams analysis."""
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

    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Date'],
            y=daily_metrics['Us Penalty Minutes'],
            name='Penalty Minutes',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Date'],
            y=daily_metrics['Power Play Efficiency'],
            name='PP Efficiency',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )

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
    """Create a correlation plot for statistically significant features."""
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
            if len(valid_data) < 2:
                st.warning(f"Not enough data to compute correlation for '{feature}'.")
                continue
            corr, p_val = stats.pearsonr(valid_data[feature], valid_data[target])
            if p_val <= significance_level:
                correlations[feature] = corr
                p_values[feature] = p_val

    if not correlations:
        st.warning("No statistically significant correlations found.")
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

    # Load the initial data
    full_df = load_data()
    if full_df.empty:
        st.warning("No data available.")
        return
    
    # Get unique seasons and add "All Seasons" option
    seasons = ["All Seasons"] + sorted(full_df['Season Year'].unique().tolist())
    
    # Add season filter in the sidebar
    st.sidebar.header("Filters")
    selected_season = st.sidebar.selectbox(
        "Select Season",
        seasons,
        index=0  # Default to "All Seasons"
    )

    # Filter data based on selected season
    filtered_df = filter_data_by_season(full_df, selected_season)
    
    # Process the filtered data
    daily_metrics, current_streaks, total_wins, total_losses, total_goals = process_data(filtered_df)

    # Display season information
    if selected_season != "All Seasons":
        st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.1); 
                        border-radius: 10px; 
                        padding: 10px; 
                        text-align: center; 
                        color: white; 
                        margin-bottom: 20px;">
                <h3>Viewing Season: {selected_season}</h3>
            </div>
        """, unsafe_allow_html=True)

    # Add the Last Game Date Message at the Top
    try:
        last_game_date = daily_metrics['Date'].max().strftime('%B %d, %Y')
        st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.1); 
                        border-radius: 10px; 
                        padding: 10px; 
                        text-align: center; 
                        color: white