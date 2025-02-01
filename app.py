import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ---------------------------------------------------------------
# Page Configuration (centered layout for mobile friendliness)
# ---------------------------------------------------------------
st.set_page_config(page_title="NHL Gaming Hub", page_icon="🏒", layout="centered")

# ---------------------------------------------------------------
# Custom CSS for Mobile-Friendly Metric Cards
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-row {
        display: flex;
        overflow-x: auto;
        gap: 10px;
        padding: 10px 0;
    }
    .metric-card {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        min-width: 120px;
        text-align: center;
        flex: 0 0 auto;
    }
    .metric-card h4 {
        margin: 0;
        font-size: 1em;
        color: #ddd;
    }
    .metric-value {
        font-size: 1.5em;
        color: #fff;
    }
    .metric-delta {
        font-size: 0.9em;
        color: #0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Helper Functions for Metric Cards
# ---------------------------------------------------------------
def metric_card(title, value, delta=None):
    """Return HTML for a single metric card."""
    delta_html = f"<div class='metric-delta'>{delta:+.1f}</div>" if delta is not None else ""
    return f"<div class='metric-card'><h4>{title}</h4><div class='metric-value'>{value}</div>{delta_html}</div>"

def metric_row(metrics):
    """
    Given a list of (title, value, delta) tuples, returns HTML for a horizontal row of metric cards.
    Each delta can be None if not applicable.
    """
    cards = "".join(metric_card(title, value, delta) for title, value, delta in metrics)
    return f"<div class='metric-row'>{cards}</div>"

# ---------------------------------------------------------------
# Data Loading and Preprocessing
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Let pandas auto-detect delimiter (works for both comma or tab)
        df = pd.read_csv('data/nhl_stats.csv', sep=None, engine='python')
        # Strip extra whitespace from column names.
        df.columns = df.columns.str.strip()
        
        # Check for the Date column.
        if "Date" not in df.columns:
            st.write("Columns found in CSV:", df.columns.tolist())
            raise KeyError("The 'Date' column is not found in the CSV file. Please verify the header and delimiter.")
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Convert listed columns to numeric
        numeric_cols = [
            'Us Goals', 'Opponent Goals', 'Us Total Shots', 'Opponent Total Shots',
            'Us Hits', 'Opponent Hits', 'Us TOA', 'Opponent TOA',
            'Us Passing Rate', 'Opponent Passing Rate',
            'Us Faceoffs Won', 'Opponent Faceoffs Won',
            'Us Penalty Minutes', 'Opponent Penalty Minutes',
            'Us Power Play Minutes', 'Opponent Power Play Minutes',
            'Nolan Goal', 'Andrew Goal', 'Morgan Goal',
            'Us Shorthanded Goals', 'Opponent Shorthanded Goals'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process Season Year as categorical
        if 'Season Year' in df.columns:
            df['Season Year'] = pd.to_numeric(df['Season Year'], errors='coerce')
            df = df.dropna(subset=['Season Year'])
            df['Season Year'] = df['Season Year'].astype(int).astype(str)
            df['Season Year'] = df['Season Year'].astype('category')
            df = df.sort_values(['Season Year', 'Date'])
        else:
            df = df.sort_values('Date')
        
        # Create a sequential game number.
        df['Game_Number'] = np.arange(1, len(df) + 1)
        
        # ----------------- Existing Team Metrics -----------------
        df['IsWin'] = (df['Win / Loss'] == 'Win').astype(int)
        df['Shot Efficiency'] = (df['Us Goals'] / df['Us Total Shots'] * 100).fillna(0)
        df['Goal Differential'] = df['Us Goals'] - df['Opponent Goals']
        if 'Us Power Play Minutes' in df.columns:
            df['Power Play Efficiency'] = np.where(df['Us Power Play Minutes'] > 0,
                                                   df['Us Goals'] / df['Us Power Play Minutes'],
                                                   0)
        if ('Us Faceoffs Won' in df.columns) and ('Opponent Faceoffs Won' in df.columns):
            tot_faceoffs = df['Us Faceoffs Won'] + df['Opponent Faceoffs Won']
            df['Faceoff Win Rate'] = np.where(tot_faceoffs > 0,
                                              (df['Us Faceoffs Won'] / tot_faceoffs) * 100,
                                              0)
        
        # ----------------- New Additional Metrics -----------------
        df['Shot_Ratio'] = np.where((df['Us Total Shots'] + df['Opponent Total Shots']) > 0,
                                    df['Us Total Shots'] / (df['Us Total Shots'] + df['Opponent Total Shots']) * 100,
                                    0)
        df['Hit_Differential'] = df['Us Hits'] - df['Opponent Hits']
        df['TOA_Differential'] = df['Us TOA'] - df['Opponent TOA']
        df['Passing_Differential'] = df['Us Passing Rate'] - df['Opponent Passing Rate']
        df['Faceoff_Differential'] = df['Us Faceoffs Won'] - df['Opponent Faceoffs Won']
        df['Shorthanded_Differential'] = df['Us Shorthanded Goals'] - df['Opponent Shorthanded Goals']
        
        # ----------------- Rolling Averages (5-game window) -----------------
        window = 5
        if 'Season Year' in df.columns:
            df['Goals_Rolling'] = df.groupby('Season Year')['Us Goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df['WinRate_Rolling'] = df.groupby('Season Year')['IsWin'].transform(lambda x: x.rolling(window, min_periods=1).mean() * 100)
            if 'Power Play Efficiency' in df.columns:
                df['PowerPlay_Rolling'] = df.groupby('Season Year')['Power Play Efficiency'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            if 'Faceoff Win Rate' in df.columns:
                df['FaceoffWin_Rolling'] = df.groupby('Season Year')['Faceoff Win Rate'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            if 'Us Penalty Minutes' in df.columns:
                df['PenaltyMinutes_Rolling'] = df.groupby('Season Year')['Us Penalty Minutes'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df['Shot_Ratio_Rolling'] = df.groupby('Season Year')['Shot_Ratio'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df['Hit_Diff_Rolling'] = df.groupby('Season Year')['Hit_Differential'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df['TOA_Diff_Rolling'] = df.groupby('Season Year')['TOA_Differential'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df['Passing_Diff_Rolling'] = df.groupby('Season Year')['Passing_Differential'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df['Faceoff_Diff_Rolling'] = df.groupby('Season Year')['Faceoff_Differential'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df['Shorthanded_Diff_Rolling'] = df.groupby('Season Year')['Shorthanded_Differential'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        else:
            df['Goals_Rolling'] = df['Us Goals'].rolling(window, min_periods=1).mean()
            df['WinRate_Rolling'] = df['IsWin'].rolling(window, min_periods=1).mean() * 100
            if 'Power Play Efficiency' in df.columns:
                df['PowerPlay_Rolling'] = df['Power Play Efficiency'].rolling(window, min_periods=1).mean()
            if 'Faceoff Win Rate' in df.columns:
                df['FaceoffWin_Rolling'] = df['Faceoff Win Rate'].rolling(window, min_periods=1).mean()
            if 'Us Penalty Minutes' in df.columns:
                df['PenaltyMinutes_Rolling'] = df['Us Penalty Minutes'].rolling(window, min_periods=1).mean()
            df['Shot_Ratio_Rolling'] = df['Shot_Ratio'].rolling(window, min_periods=1).mean()
            df['Hit_Diff_Rolling'] = df['Hit_Differential'].rolling(window, min_periods=1).mean()
            df['TOA_Diff_Rolling'] = df['TOA_Differential'].rolling(window, min_periods=1).mean()
            df['Passing_Diff_Rolling'] = df['Passing_Differential'].rolling(window, min_periods=1).mean()
            df['Faceoff_Diff_Rolling'] = df['Faceoff_Differential'].rolling(window, min_periods=1).mean()
            df['Shorthanded_Diff_Rolling'] = df['Shorthanded_Differential'].rolling(window, min_periods=1).mean()
        
        # ----------------- Player Metrics (existing) -----------------
        for player in ['Nolan', 'Andrew', 'Morgan']:
            if f'{player} Goal' in df.columns:
                df[f'{player}_Rolling'] = df[f'{player} Goal'].rolling(window, min_periods=1).mean()
                df[f'{player}_Goal_Rate'] = np.where(df['Us Goals'] > 0, df[f'{player} Goal'] / df['Us Goals'], 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ---------------------------------------------------------------
# Season Selector (placed at the top for mobile friendliness)
# ---------------------------------------------------------------
if 'Season Year' in df.columns:
    seasons = sorted(df['Season Year'].unique(), key=lambda x: int(x))
    selected_season = st.selectbox("Select Season", options=seasons, index=len(seasons)-1)
    df = df[df['Season Year'] == selected_season]
else:
    selected_season = "All Seasons"

# ---------------------------------------------------------------
# Display the last updated date based on the most recent "Date"
# ---------------------------------------------------------------
last_update = df["Date"].max()
if pd.isnull(last_update):
    last_update_str = "N/A"
else:
    last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S")
st.markdown(
    f"<div style='text-align: center; font-size: 0.9em; color: gray;'>Last updated: {last_update_str}</div>",
    unsafe_allow_html=True
)


# ---------------------------------------------------------------
# Main App Title and Tabs
# ---------------------------------------------------------------
st.title("🏒 NHL Stats Report")
tabs = st.tabs(["Team Overview", "Player Trends", "Special Teams", "Metric Explorer"])

# ------------------------------
# 1. Team Overview
# ------------------------------
with tabs[0]:
    st.header("Overall Team Trends")
    
    # --- Team & Player Total Goals (Custom Metric Row) ---
    team_total = f"{df['Us Goals'].sum():.0f}"
    nolan_total = f"{df['Nolan Goal'].sum():.0f}" if 'Nolan Goal' in df.columns else "N/A"
    andrew_total = f"{df['Andrew Goal'].sum():.0f}" if 'Andrew Goal' in df.columns else "N/A"
    morgan_total = f"{df['Morgan Goal'].sum():.0f}" if 'Morgan Goal' in df.columns else "N/A"
    html_goals = metric_row([
        ("Team Total Goals", team_total, None),
        ("Nolan Total Goals", nolan_total, None),
        ("Andrew Total Goals", andrew_total, None),
        ("Morgan Total Goals", morgan_total, None)
    ])
    st.markdown(html_goals, unsafe_allow_html=True)
    
    # --- Recent Game Metrics (4 Metrics) ---
    if len(df) >= 10:
        recent = df.tail(5)
        previous = df.iloc[-10:-5]
    else:
        recent = df.tail(5)
        previous = df.head(5)
    
    win_rate = recent['IsWin'].mean() * 100
    win_rate_delta = (recent['IsWin'].mean() - previous['IsWin'].mean()) * 100
    goals_game = recent['Us Goals'].mean()
    goals_game_delta = recent['Us Goals'].mean() - previous['Us Goals'].mean()
    shot_eff = recent['Shot Efficiency'].mean()
    shot_eff_delta = recent['Shot Efficiency'].mean() - previous['Shot Efficiency'].mean()
    goal_diff = recent['Goal Differential'].mean()
    goal_diff_delta = recent['Goal Differential'].mean() - previous['Goal Differential'].mean()
    html_recent = metric_row([
        ("Win Rate (%)", f"{win_rate:.1f}", win_rate_delta),
        ("Goals/Game", f"{goals_game:.1f}", goals_game_delta),
        ("Shot Efficiency (%)", f"{shot_eff:.1f}", shot_eff_delta),
        ("Goal Differential", f"{goal_diff:.1f}", goal_diff_delta)
    ])
    st.markdown(html_recent, unsafe_allow_html=True)
    
    # --- Additional Team Metrics (6 Metrics, 2 rows of 3) ---
    shot_ratio = recent['Shot_Ratio'].mean()
    shot_ratio_delta = recent['Shot_Ratio'].mean() - previous['Shot_Ratio'].mean()
    hit_diff = recent['Hit_Differential'].mean()
    hit_diff_delta = recent['Hit_Differential'].mean() - previous['Hit_Differential'].mean()
    toa_diff = recent['TOA_Differential'].mean()
    toa_diff_delta = recent['TOA_Differential'].mean() - previous['TOA_Differential'].mean()
    passing_diff = recent['Passing_Differential'].mean()
    passing_diff_delta = recent['Passing_Differential'].mean() - previous['Passing_Differential'].mean()
    faceoff_diff = recent['Faceoff_Differential'].mean()
    faceoff_diff_delta = recent['Faceoff_Differential'].mean() - previous['Faceoff_Differential'].mean()
    shorthanded_diff = recent['Shorthanded_Differential'].mean()
    shorthanded_diff_delta = recent['Shorthanded_Differential'].mean() - previous['Shorthanded_Differential'].mean()
    html_additional1 = metric_row([
        ("Shot Ratio (%)", f"{shot_ratio:.1f}", shot_ratio_delta),
        ("Hit Differential", f"{hit_diff:.1f}", hit_diff_delta),
        ("TOA Differential", f"{toa_diff:.1f}", toa_diff_delta)
    ])
    html_additional2 = metric_row([
        ("Passing Differential", f"{passing_diff:.1f}", passing_diff_delta),
        ("Faceoff Differential", f"{faceoff_diff:.1f}", faceoff_diff_delta),
        ("Shorthanded Diff", f"{shorthanded_diff:.1f}", shorthanded_diff_delta)
    ])
    st.markdown(html_additional1, unsafe_allow_html=True)
    st.markdown(html_additional2, unsafe_allow_html=True)
    
    # --- Goals per Game Trend Chart (unchanged) ---
    fig_goals = go.Figure()
    fig_goals.add_trace(go.Scatter(
        x=df['Game_Number'],
        y=df['Us Goals'],
        mode='markers+lines',
        name="Game Goals",
        marker=dict(color='cyan')
    ))
    fig_goals.add_trace(go.Scatter(
        x=df['Game_Number'],
        y=df['Goals_Rolling'],
        mode='lines',
        name="5-Game Average",
        line=dict(width=3, color='yellow')
    ))
    fig_goals.update_layout(xaxis_title="Game Number", yaxis_title="Goals", template="plotly_dark")
    st.plotly_chart(fig_goals, use_container_width=True)
    
    # --- Rolling Win Rate Trend Chart (unchanged) ---
    fig_win = go.Figure()
    fig_win.add_trace(go.Scatter(
        x=df['Game_Number'],
        y=df['WinRate_Rolling'],
        mode='lines+markers',
        name="Win Rate (5-Game Rolling)",
        marker=dict(color='lime')
    ))
    fig_win.update_layout(xaxis_title="Game Number", yaxis_title="Win Rate (%)", template="plotly_dark")
    st.plotly_chart(fig_win, use_container_width=True)
    
    # --- Player Comparison Over Time Chart (unchanged) ---
    st.subheader("Player Comparison Over Time")
    cum_nolan = df['Nolan Goal'].cumsum() if 'Nolan Goal' in df.columns else None
    cum_andrew = df['Andrew Goal'].cumsum() if 'Andrew Goal' in df.columns else None
    cum_morgan = df['Morgan Goal'].cumsum() if 'Morgan Goal' in df.columns else None
    fig_comp = go.Figure()
    if cum_nolan is not None:
        fig_comp.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=cum_nolan,
            mode='lines+markers',
            name="Nolan",
            line=dict(color='blue')
        ))
    if cum_andrew is not None:
        fig_comp.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=cum_andrew,
            mode='lines+markers',
            name="Andrew",
            line=dict(color='orange')
        ))
    if cum_morgan is not None:
        fig_comp.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=cum_morgan,
            mode='lines+markers',
            name="Morgan",
            line=dict(color='magenta')
        ))
    fig_comp.update_layout(title="Cumulative Player Goals Over Time", xaxis_title="Game Number", yaxis_title="Cumulative Goals", template="plotly_dark")
    st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------------
# 2. Player Trends
# ------------------------------
with tabs[1]:
    st.header("Individual Player Trends")
    player = st.selectbox("Select Player", options=['Nolan', 'Andrew', 'Morgan'])
    if f'{player} Goal' not in df.columns:
        st.error(f"Data for {player} is not available.")
    else:
        st.markdown(f"### {player}'s Goals Over Time")
        fig_player = go.Figure()
        fig_player.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df[f'{player} Goal'],
            mode='markers',
            name="Game Goals",
            marker=dict(color='magenta')
        ))
        fig_player.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df[f'{player}_Rolling'],
            mode='lines',
            name="5-Game Average",
            line=dict(width=3, color='orange')
        ))
        fig_player.update_layout(xaxis_title="Game Number", yaxis_title="Goals", template="plotly_dark")
        st.plotly_chart(fig_player, use_container_width=True)
        player_total = df[f'{player} Goal'].sum()
        player_avg = df[f'{player} Goal'].mean()
        player_rate = df[f'{player}_Goal_Rate'].mean() * 100 if f'{player}_Goal_Rate' in df.columns else 0
        st.markdown(f"**Total Goals for {player}:** {player_total:.0f}  •  **Average per Game:** {player_avg:.2f}  •  **Goal Contribution:** {player_rate:.1f}%")

# ------------------------------
# 3. Special Teams
# ------------------------------
with tabs[2]:
    st.header("Special Teams Metrics")
    has_powerplay = 'Power Play Efficiency' in df.columns
    has_faceoffs = 'Faceoff Win Rate' in df.columns
    has_penalties = 'Us Penalty Minutes' in df.columns
    if len(df) >= 10:
        recent = df.tail(5)
        previous = df.iloc[-10:-5]
    else:
        recent = df.tail(5)
        previous = df.head(5)
    # Use a custom metric row for the three special teams metrics.
    metrics_special = []
    if has_powerplay:
        pp_val = f"{recent['Power Play Efficiency'].mean():.2f}"
        pp_delta = recent['Power Play Efficiency'].mean() - previous['Power Play Efficiency'].mean()
        metrics_special.append(("Power Play Eff.", pp_val, pp_delta))
    if has_faceoffs:
        fo_val = f"{recent['Faceoff Win Rate'].mean():.1f}%"
        fo_delta = recent['Faceoff Win Rate'].mean() - previous['Faceoff Win Rate'].mean()
        metrics_special.append(("Faceoff Win Rate", fo_val, fo_delta))
    if has_penalties:
        pen_val = f"{recent['Us Penalty Minutes'].mean():.1f}"
        pen_delta = recent['Us Penalty Minutes'].mean() - previous['Us Penalty Minutes'].mean()
        metrics_special.append(("Penalty Minutes/Game", pen_val, pen_delta))
    if metrics_special:
        st.markdown(metric_row(metrics_special), unsafe_allow_html=True)
    
    # Special Teams Trend Charts (unchanged)
    if has_powerplay:
        st.markdown("#### Power Play Efficiency Trend")
        fig_pp = go.Figure()
        fig_pp.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df['Power Play Efficiency'],
            mode='markers+lines',
            name="Game PP Eff.",
            marker=dict(color='gold')
        ))
        fig_pp.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df['PowerPlay_Rolling'],
            mode='lines',
            name="5-Game Average",
            line=dict(width=3, color='darkgoldenrod')
        ))
        fig_pp.update_layout(xaxis_title="Game Number", yaxis_title="Power Play Efficiency", template="plotly_dark")
        st.plotly_chart(fig_pp, use_container_width=True)
    if has_faceoffs:
        st.markdown("#### Faceoff Win Rate Trend")
        fig_fo = go.Figure()
        fig_fo.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df['Faceoff Win Rate'],
            mode='markers+lines',
            name="Game Faceoff Win Rate",
            marker=dict(color='violet')
        ))
        fig_fo.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df['FaceoffWin_Rolling'],
            mode='lines',
            name="5-Game Average",
            line=dict(width=3, color='purple')
        ))
        fig_fo.update_layout(xaxis_title="Game Number", yaxis_title="Faceoff Win Rate (%)", template="plotly_dark")
        st.plotly_chart(fig_fo, use_container_width=True)
    if has_penalties:
        st.markdown("#### Penalty Minutes Trend")
        fig_pm = go.Figure()
        fig_pm.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df['Us Penalty Minutes'],
            mode='markers+lines',
            name="Game Penalty Minutes",
            marker=dict(color='red')
        ))
        fig_pm.add_trace(go.Scatter(
            x=df['Game_Number'],
            y=df['PenaltyMinutes_Rolling'],
            mode='lines',
            name="5-Game Average",
            line=dict(width=3, color='darkred')
        ))
        fig_pm.update_layout(xaxis_title="Game Number", yaxis_title="Penalty Minutes", template="plotly_dark")
        st.plotly_chart(fig_pm, use_container_width=True)

# ------------------------------
# 4. Metric Explorer
# ------------------------------
with tabs[3]:
    st.header("Metric Explorer")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    col_x, col_y = st.columns(2)
    x_var = col_x.selectbox("Select X variable", options=numeric_columns, index=0)
    y_var = col_y.selectbox("Select Y variable", options=numeric_columns, index=1)
    if x_var and y_var:
        plot_df = df[[x_var, y_var, 'Game_Number']].dropna()
        if len(plot_df) >= 2:
            try:
                fig_scatter = px.scatter(
                    plot_df,
                    x='Game_Number' if x_var == 'Game_Number' else x_var,
                    y=y_var,
                    trendline="ols",
                    title=f"Correlation between {x_var} and {y_var}"
                )
                fig_scatter.update_layout(template="plotly_dark", xaxis_title=x_var, yaxis_title=y_var)
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating scatter plot: {e}")
            valid_data = plot_df[[x_var, y_var]].dropna()
            if len(valid_data) >= 2:
                try:
                    corr, p_val = stats.pearsonr(valid_data[x_var], valid_data[y_var])
                    st.markdown(f"**Correlation:** {corr:.2f} (p-value: {p_val:.3f})")
                except Exception as e:
                    st.warning(f"Could not calculate correlation: {e}")
            else:
                st.warning("Not enough data points for correlation (need at least 2).")
        else:
            st.warning("Not enough valid data points to create scatter plot.")