import math
from random import randint
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import trimesh
import random
from pybaseball import statcast
from random import randint
from pybaseball import pitching_stats_range
from pybaseball import batting_stats_range
import datetime
from datetime import timedelta
from datetime import datetime
from pybaseball import schedule_and_record
import requests
import pybaseball as pb
import pickle
st.set_page_config(
    page_title="MLB Visualizer",  # This sets the browser tab title
    page_icon="⚾",               # This sets the page icon to a baseball emoji
    layout="wide"                # This sets the page layout to wide
)
def generate_random_color():
    """Generate a random hex color."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def display_player_image(player_id, width2, caption2):
    # Construct the URL for the player image using the player ID
    image_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{player_id}/headshot/67/current"
    
    # Check if the image URL returns a successful response
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center;">'
    f'<img src="{image_url}" style="width: {width2}px;">'
    f'<p style="text-align: center; font-size: 30px;">{caption2}</p>'  # Adjust font-size as needed
    f'</div>',
    unsafe_allow_html=True
)
    
        # st.image(image_url, width=width2, caption=caption2)
    else:
        image_url = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{"Image Unavailable"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
def plot_points_on_stl(stl_file_path, scatter_points):
    # Load the STL file using trimesh
    mesh = trimesh.load(stl_file_path)
    vertices = mesh.vertices
    faces = mesh.faces
    z_min, z_max = -50, 30

    # Scale and translate vertices
    vertices[:, 2] = np.clip(vertices[:, 2], z_min, z_max)  # Clip to z-axis range
    if mesh.visual and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors[:, :3] / 255  # Normalize colors to [0, 1]
    else:
        colors = np.tile(np.array([0.8, 0.8, 0.8]), (vertices.shape[0], 1))  # Default to light gray
    # Create a Plotly figure
    fig = go.Figure()

    # Add the STL mesh as a surface
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.5,
        hoverinfo='none',
        color='grey',  # Make base color transparent
        # vertexcolor=['green']  # Use vertex colors
    ))

    fig.add_trace(go.Scatter3d(
        x=scatter_points['hc_x'],
        y=scatter_points['hc_y'],  # Assuming this is the y-axis
        z=len(scatter_points['hc_x'])*[0],  # Assuming this is the z-axis
        mode='markers',
        marker=dict(size=3, color=scatter_points['color']),
        hoverinfo='text',
        hovertext=df['des']
    ))
    x_range = [min(vertices[:, 0]) - 10, max(vertices[:, 0]) + 10]  # Adjust padding as needed
    y_range = [min(vertices[:, 1]) - 10, max(vertices[:, 1]) + 10]  # Adjust padding as needed

    fig.update_layout(
        height=800,  # Set the desired height in pixels
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            aspectmode='auto',
            xaxis=dict(
                range=[-150, 100],
                showbackground=False,
                backgroundcolor='black',
                showticklabels=False,
                showgrid=False,        # Turn off grid
                  showline=False,  # Turn off tick labels
                ticks=""               # Turn off ticks
            ),
            yaxis=dict(
                range=[-50, 200],
                showbackground=False,
                backgroundcolor='black',
                showticklabels=False,  # Turn off tick labels
                showgrid=False,        # Turn off grid
                 showline=False,  # Turn off tick labels
                ticks=""               # Turn off ticks
            ),
            zaxis=dict(
                range=[-60, 60],
                showbackground=False,
                backgroundcolor='black',
                showticklabels=False,  # Turn off tick labels
                showgrid=False,        # Turn off grid
                 showline=False,  # Turn off tick labels
                ticks=""               # Turn off ticks
            )
        ),
        showlegend=False  # Turn off the legend
    )

    # Render the plot in Streamlit
    
    return fig


st.markdown("""
    <style>
    .big-font {
        font-size: 100px !important;
        text-align: center;
    }
    </style>
    <p class="big-font">MLB Visualizer</p>
    """, unsafe_allow_html=True)


currentdate = datetime.now().date()

# Date input from the user
type = st.selectbox('Hitter or Pitcher',['Hitter','Pitcher'])
name = st.text_input('Enter a player name ex: Shohei Ohtani')
if name:
    nameparts = name.split(' ')
    first = nameparts[0]
    last = nameparts[1]
    playerdata = pb.playerid_lookup(first=first,last=last)
    playerid = playerdata['key_mlbam'].iloc[0]
    col1,col2 = st.columns(2)
    with col1:
        start = st.date_input('Select a start date', value=currentdate)
    with col2:
        end = st.date_input('Select an end date', value=currentdate)
    # Display the selected date in YYYY-MM-DD format
    formattedstart = start.strftime('%Y-%m-%d')
    formattedend = end.strftime('%Y-%m-%d')

# date2 = st.text_input('Enter an end date',placeholder='YYYY-MM-DD',value=currentdate)
# try:
#     # Parse the input date
#     parsed_date = datetime.strptime(date, '%Y-%m-%d')
    
#     # Extract year, month (in word form), and day
#     year = parsed_date.year
#     month = parsed_date.strftime('%B')  # Full month name
#     day = parsed_date.day
    
#     st.write(f"Year: {year}, Month: {month}, Day: {day}")
# except ValueError:
#     st.error("Please enter a valid date in YYYY-MM-DD format.")
@st.cache_data
def load_data(start,end,playerid,type):
    if type == 'Hitter':
        df = pb.statcast_batter(start_dt=formattedstart,end_dt=formattedend,player_id=playerid)
    else:
        df = pb.statcast_pitcher(start_dt=formattedstart,end_dt=formattedend,player_id=playerid)
    return df
if formattedstart:
    df = load_data(formattedstart,formattedend,playerid,type)
    playerteam = df['home_team'].value_counts().reset_index()['home_team'].iloc[0]
    # if type == 'Hitter':
    #     df = df[df['description'] == 'hit_into_play']
    # else:
    #     # df = df[df['type'] == 'B']
    #     df = df
   
    # st.write(df.columns)
    home = df[df['home_team'] == playerteam]
    away = df[df['home_team'] != playerteam]
    homeaway = st.sidebar.selectbox('Home or Away',['Home','Away'])
    if homeaway == 'Home':
        df = home
    else:
        df = away
    if not df.empty:
        if homeaway == 'Away':
            unique_matchups = df.groupby('game_date').apply(
                lambda x: [f"{row['home_team']} vs {row['away_team']}" for index, row in x.iterrows()]
            ).explode().unique()

            # Create a selectbox in Streamlit
            selected_matchup = st.selectbox("Select a Matchup", unique_matchups)
            teams = selected_matchup.split(' vs ')
            hteam = teams[0]
            ateam = teams[1]
            df = df[(df['home_team'] == hteam) & (df['away_team'] == ateam)]
        else:
            df = df
        hittypes = st.sidebar.multiselect('Select hits',df['events'].unique())
        if hittypes:
            df = df[df['events'].isin(hittypes)]
        pitchtypes = st.sidebar.multiselect('Select pitches',df['pitch_name'].unique()) 
        if pitchtypes:
            df = df[df['pitch_name'].isin(pitchtypes)]
        types = st.sidebar.multiselect('Select pitch result', df['type'].unique())
        if types:
            df = df[df['type'].isin(types)]
    # df = df[(df['events'] == 'home_run') | (df['events'] == 'triple') | (df['events'] == 'double') | (df['events'] == 'single')]
    
    #     hit_dist_min, hit_dist_max = st.sidebar.slider(
    #     "Hit Distance (ft)", 
    #     min_value=int(df['hit_distance_sc'].min()), 
    #     max_value=int(df['hit_distance_sc'].max()), 
    #     value=(int(df['hit_distance_sc'].min()), int(df['hit_distance_sc'].max()))
    # )
    #     df = df[
    #     (df['hit_distance_sc'] >= hit_dist_min) & 
    #     (df['hit_distance_sc'] <= hit_dist_max)
    # ]
    #     launch_angle_min, launch_angle_max = st.sidebar.slider(
    #     "Launch Angle", 
    #     min_value=int(df['launch_angle'].min()), 
    #     max_value=int(df['launch_angle'].max()), 
    #     value=(int(df['launch_angle'].min()), int(df['launch_angle'].max()))
    # )
    #     df = df[
    #     (df['launch_angle'] >= launch_angle_min) & 
    #     (df['launch_angle'] <= launch_angle_max)
    # ]
    #     launch_speed_min, launch_speed_max = st.sidebar.slider(
    #     "Launch Speed", 
    #     min_value=int(df['launch_speed'].min()), 
    #     max_value=int(df['launch_speed'].max()), 
    #     value=(int(df['launch_speed'].min()), int(df['launch_speed'].max()))
    # )
    #     df = df[
    #     (df['launch_speed'] >= launch_speed_min) & 
    #     (df['launch_speed'] <= launch_speed_max)
    # ]
        expectedFilter = st.checkbox('Expected Stats')
        pitchers = df['player_name'].unique()
        formatted_names = [f"{name.split(', ')[1]} {name.split(', ')[0]}" for name in pitchers]

      
        finalfeats = ['launch_speed', 'launch_speed_angle', 'bat_speed', 'hyper_speed','launch_angle','swing_path_tilt','attack_direction','zone']

        modelinput = df[finalfeats]
        with open('mlb_hit_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        preds = loaded_model.predict_proba(modelinput)[:, 1]
        # df['home_run'] = np.where(df['events'] == 'home_run',1,0)
        df['hit'] = np.where(df['events'].isin(['single', 'double', 'triple', 'home_run']), 1, 0)

        df['xHit'] = preds
        
        finalfeats2 = [
            'zone',
            'plate_z',
            'balls',
            'vy0',
            'api_break_z_with_gravity',
            'effective_speed',
            'release_speed',
            'az',
            'pfx_z',
            'api_break_x_batter_in',
            'ay',
            'strikes',
            'plate_x',
            'release_spin_rate',
            'pitch_number',
            'vz0',
            'pitch_type',
            'stand',
            'api_break_x_arm',
            'vx0',
            'pitch_name',
            'age_bat',
            'spin_axis',
            'n_thruorder_pitcher',
            'release_pos_y',
            'release_extension',
            'inning_topbot',
            'sz_top',
            'release_pos_x',
            'n_priorpa_thisgame_player_at_bat',
            'p_throws']
        modelinput2 = df[finalfeats2]

        bool_like_cols = [
            col for col in modelinput2.select_dtypes(include='object').columns
            if modelinput2[col].dropna().isin([True, False]).all()
        ]

        modelinput2[bool_like_cols] = modelinput2[bool_like_cols].astype(bool)

        modelinput2[bool_like_cols] = modelinput2[bool_like_cols].astype(int)
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()

        # Loop through only non-numeric columns
        for col in modelinput2.select_dtypes(include=['object', 'category']).columns:
            modelinput2[col] = le.fit_transform(modelinput2[col])
        with open('strikeprobabilitymodel.pkl', 'rb') as f2:
            loaded_model2 = pickle.load(f2)
        preds2 = loaded_model2.predict_proba(modelinput2)[:, 1]
        df['strike'] = np.where(df['type'] == 'S', 1, 0)
        df['xStrike'] = preds2

      
    
        byPlayer = df[df['description'] == 'hit_into_play'].groupby('player_name').agg({'xHit': 'sum', 'hit': 'sum'}).reset_index()
        byPlayer = byPlayer.sort_values(by='xHit', ascending=False).reset_index(drop=True)
        byPlayer['xHit'] = byPlayer['xHit'].apply(lambda x: round(x, 3))    
        # totalpitches = len(df)
        # totalabs = len(df['des'].unique())
        # byPlayer['totalPitches'] = totalpitches
        # byPlayer['totalABs'] = totalabs
        # byPlayer['xBA'] = byPlayer['xHit']/totalabs
        # byPlayer['BA'] = byPlayer['hit']/totalabs
        st.subheader(f'xHits: {byPlayer['xHit'].iloc[0]}')
        st.subheader(f'Hits: {byPlayer['hit'].iloc[0]}')

     
        names_list = []

        for index, row in df.iterrows():
            text = row['des']
            words = text.split()
            first_two_words = ' '.join(words[:2])
            names_list.append(first_two_words)
        unique_names_set = set(names_list)

        unique_names_list = list(unique_names_set)
        def first_two_words(text):
            words = text.split()
            return ' '.join(words[:2])

        # df = df[df['des'].apply(lambda x: first_two_words(x) in selectp)]
        ids = df['batter'].unique()
        import pandas as pd

        # df = pd.read_csv('/Users/ryan/Desktop/FantasyPython/giantsstadium.csv')
        unique_pitch_types = df['pitch_name'].unique()

        color_mapping = {pitch_type: generate_random_color() for pitch_type in unique_pitch_types}

        df['color'] = df['pitch_name'].map(color_mapping)


        df['hc_x'] = df['hc_x']-125.42
        df['hc_x'] = df['hc_x']*.70
        df['hc_y'] = 206.27-df['hc_y']
        df['hc_y'] = df['hc_y']*.70
        st.write(df.columns)
        # st.write(df)
        hometeam = df['home_team'].iloc[0].lower()
        if hometeam == 'phi':
            stl_file_path = 'stlfolder/simplify_stadium_phi.stl'
        else:
            stl_file_path = f'stlfolder/stadium_{hometeam}.stl'


        scatter_points = df[['hc_x', 'hc_y','color']]  # Make sure these columns exist in your DataFrame

        display_player_image(playerid,200,'')
        # Call the function to plot STL with scatter points
        fig = plot_points_on_stl(stl_file_path, scatter_points)
        x_values = []
        y_values = []
        z_values = []
        # Loop through each row in the 'location' column
        plays = []
        pitchers = []
        dists = []
        pitchtypes = []
        colors = []
        innings = []
        dates = []
        types = []
        xHits = []
        xStrikes = []
        for index, row in df.iterrows():
            # if 'homer' in row['des']:
            #     x_values.append(row['hc_x']*3)
            # elif 'pop' in row['des']:
            #     x_values.append(row['hc_x']*3)
            # else:
            # # Append the value from column 'x' to the list
            # if 206.26-row['hc_y'] > 150:
            #     row['hc_y'] = randint(58,60)
            # if 'homer' in row['des'] or 'grand slam' in row['des']:
            #     row['hc_y'] = randint(35,40)


            x_values.append(row['hc_x'])
            y_values.append(row['hc_y'])
            z_values.append(0)
            plays.append(row['des'])
            pitchers.append(row['pitcher'])
            dists.append(206-row['hc_y'])
            pitchtypes.append(row['pitch_name'])
            innings.append(row['inning'])
            dates.append(row['game_date'])
            colors.append(row['color'])
            xHits.append(row['xHit'])
            xStrikes.append(row['xStrike'])
            types.append(row['type'])

        x_values2 = []
        y_values2 = []
        z_values2 = []
        # Loop through each row in the 'location' column
        for index, row in df.iterrows():
            # Append the value from column 'x' to the list
            x_values2.append(row['plate_x'])
            y_values2.append(0)
            z_values2.append(row['plate_z'])
        x_valuesg = []
        y_valuesg = []
        z_valuesg = []
  
        def calculate_distance(x1, y1, x2, y2):
            """Calculate the distance between two points (x1, y1) and (x2, y2)."""
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def generate_arc_points(p1, p2, apex, num_points=100):
            """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
            t = np.linspace(0, 1, num_points)
            x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
            y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
            z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
            return x, y, z

        # Example lists of x and y coordinates
        x_coords = x_values
        y_coords = y_values
        z_value = 0  # Fixed z value
        x_coords2 = x_values2
        y_coords2 = y_values2

        x_coordsg = x_valuesg
        y_coordsg = y_valuesg
        z_valueg = 0  # Fixed z value


        launch_angles = df['launch_angle'].tolist()
        dess = df['des'].tolist()

        plays2 = len(plays)
        for i in range(len(x_coords)):
            des = plays[i]
            pitcher = pitchers[i]
            ys = dists[i]
            pitch = pitchtypes[i]
            inning = innings[i]
            x1 = x_coords[i]
            y1 = y_coords[i]
            x2 = x_coords2[i]
            y2 = y_coords2[i]
            launch_angle = launch_angles[i]
            des = dess[i]
            color = colors[i]
            date = dates[i]
            xHit = xHits[i]
            if launch_angle < 0:
                launch_angle = 0
            
            # Define the start and end points
            p1 = np.array([x1, y1, z_value])
            p2 = np.array([x2, y2, z_value])
            
            # Adjust the apex height based on the launch angle
            # Example conversion: Adjust this scaling factor as needed
            if 'homer' in des.lower():
                height_scaling_factor = 0.5 # Adjust this factor to change how launch angle affects height
            else:
                height_scaling_factor = 0.5
            h = height_scaling_factor * np.tan(np.radians(launch_angle)) * np.linalg.norm(p2 - p1)
            
            # Adjust the apex position based on the calculated height
            apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])
            
            # Generate arc points
            x, y, z = generate_arc_points(p1, p2, apex)
            
            # Add arc trace to figure
            if expectedFilter:
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(width=5,color=color),
                    name=f'Arc {i}',
                    opacity=df['xHit'].iloc[i],
                    hoverinfo='text',
                    hovertext=f'{des}<br>Pitch Type: {pitch}<br>Inning: {inning}<br>Pitcher: {pitcher}<br>Date: {date}<br>xHit: {xHit}',

                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(width=5,color=color),
                    name=f'Arc {i}',
                    hoverinfo='text',
                    hovertext=f'{des}<br>Pitch Type: {pitch}<br>Inning: {inning}<br>Pitcher: {pitcher}<br>Date: {date}<br>xHit: {xHit}',

                ))
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig,use_container_width=True)


        import plotly.graph_objects as go
        import numpy as np


        # Create a 3D scatter plot
        fig = go.Figure()



        # Plot plate positions
        # df2 = df[df['type'] == 'X']
        df2 = df
        if expectedFilter:
            fig.add_trace(go.Scatter3d(
                x=df2['plate_x'],
                y=[0] * len(df2),  # y is always 0 for plate positions
                z=df2['plate_z'],
                mode='markers',
                marker=dict(size=8, color=df2['xStrike'],colorscale='hot'),
                name='Plate Position',
                # opacity=df2['xStrike'],
                hoverinfo='text',
                hovertext=[f"{pitchname} - {t} - xStrike: {xS}" for pitchname, t, xS in zip(df2['pitch_name'], df2['type'], df2['xStrike'])]
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=df2['plate_x'],
                y=[0] * len(df2),  # y is always 0 for plate positions
                z=df2['plate_z'],
                mode='markers',
                marker=dict(size=8, color=df2['color'],line=dict(color='black', width=1)),
                name='Plate Position',
                # opacity=df2['xStrike'],
                hoverinfo='text',
                hovertext=[f"{pitchname} - {t} - xStrike: {xS}" for pitchname, t, xS in zip(df2['pitch_name'], df2['type'], df2['xStrike'])]
            ))
        # Add lines connecting release points to adjusted plate positions
        def plot_curve(x_start, y_start, z_start, x_end, y_end, z_end, pfx_x, pfx_z, pitch_name):
            t = np.linspace(0, 1, 100)  # 100 points for smooth curve
            
            # Adjust pfx_x and pfx_z based on pitch_name
            if pitch_name in ["Changeup","Sinker"]:
                # pfx_x = -pfx_x  # Flip the effect along the x-axis
                pfx_z = -pfx_z  # Optionally, flip the effect along the z-axis if needed
            pfx_x=-pfx_x
            x_curve = x_start + (x_end - x_start) * t + pfx_x * t * (1 - t)
            y_curve = y_start + (y_end - y_start) * t
            z_curve = z_start + (z_end - z_start) * t + pfx_z * t * (1 - t)
            
            return x_curve, y_curve, z_curve

        for i in range(len(df2)):
            x_start = df2['release_pos_x'].iloc[i]
            y_start = 100
            z_start = df2['release_pos_z'].iloc[i]
            x_end = df2['plate_x'].iloc[i]
            y_end = 0
            z_end = df2['plate_z'].iloc[i]
            
            pfx_x = df2['pfx_x'].iloc[i]
            pfx_z = df2['pfx_z'].iloc[i]
            pitch2 = pitchtypes[i]
            pitcher2 = pitchers[i]
            inning2 = innings[i]
            date2 = dates[i]
            des2 = plays[i]
            xStrike = xStrikes[i]
            type2 = types[i]

            
            x_curve, y_curve, z_curve = plot_curve(x_start, y_start, z_start, x_end, y_end, z_end, pfx_x, pfx_z,pitch2)
            
            if expectedFilter:
                fig.add_trace(go.Scatter3d(
                    x=x_curve,
                    y=y_curve,
                    z=z_curve,
                    mode='lines',
                    line=dict(color='black', width=4),
                    name=f'Pitch Path {i}',
                    hoverinfo='text',
                    hovertext=f'{pitch2}<br>{type2}<br>Inning: {inning2}<br>Pitcher: {pitcher2}<br>Date: {date2}<br>xStrike: {xStrike}',
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=x_curve,
                    y=y_curve,
                    z=z_curve,
                    mode='lines',
                    line=dict(color=df2['color'].iloc[i], width=4),
                    name=f'Pitch Path {i}',
                    hoverinfo='text',
                    hovertext=f'{pitch2}<br>{type2}<br>Inning: {inning2}<br>Pitcher: {pitcher2}<br>Date: {date2}<br>xStrike: {xStrike}',
                ))


        fig.add_trace(go.Scatter3d(
            x=[-.7, .9],    # X coordinates of the line start and end
            y=[0, 0],     # Y coordinates of the line start and end (same value for both points)
            z=[1.6, 1.6],     # Z coordinates of the line start and end
            mode='lines', # Set mode to 'lines' to draw a line
            line=dict(color='grey', width=10), # Line style
                hoverinfo='none',

            name='Strike Zone Sides' # Legend name
        ))
        fig.add_trace(go.Scatter3d(
            x=[-.7, -.7],    # X coordinates of the line start and end
            y=[0, 0],     # Y coordinates of the line start and end (same value for both points)
            z=[1.6, 3.4],     # Z coordinates of the line start and end
            mode='lines', # Set mode to 'lines' to draw a line
            line=dict(color='grey', width=10), # Line style
                hoverinfo='none',

            name='Strike Zone Sides' # Legend name
        ))
        fig.add_trace(go.Scatter3d(
            x=[.9, .9],    # X coordinates of the line start and end
            y=[0, 0],     # Y coordinates of the line start and end (same value for both points)
            z=[1.6, 3.4],     # Z coordinates of the line start and end
            mode='lines', # Set mode to 'lines' to draw a line
            line=dict(color='grey', width=10), # Line style
                hoverinfo='none',

            name='Strike Zone Sides' # Legend name
        ))
        fig.add_trace(go.Scatter3d(
            x=[-.7, .9],    # X coordinates of the line start and end
            y=[0, 0],     # Y coordinates of the line start and end (same value for both points)
            z=[3.4, 3.4],     # Z coordinates of the line start and end
            mode='lines', # Set mode to 'lines' to draw a line
            line=dict(color='grey', width=10), # Line style
            hoverinfo='none',
            name='Strike Zone Sides' # Legend name
        ))

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Horizontal Position (x)',
                yaxis_title='Height (y)',
                zaxis_title='Vertical Position (z)'
            ),
        )
        fig.update_layout(
        
                    
            height=700,
            scene=dict(
                xaxis=dict(
                    title='',
                    range=[-10, 10],
                    showticklabels=False,
                                showgrid=False,  # Set the range for the x-axis
                ),
                yaxis=dict(
                    title='',
                    range=[0, 100] ,
                    
                    showticklabels=False,
                                showgrid=False, # Set the range for the y-axis
                ),
                zaxis=dict(
                    title='',
                    range=[-6, 12],
                    # showbackground=True,
                    # backgroundcolor='#006400',

                                showticklabels=False,
                                showgrid=False,
                )

            ),
            showlegend=False
        )
        with col2:
            st.plotly_chart(fig)
            
     
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
                hits_df = df[df['description'] == 'hit_into_play'].copy()
                if not hits_df.empty:
                    fig_scatter = px.scatter(
                        hits_df, 
                        x='launch_speed', 
                        y='launch_angle',
                        # color='events',
                        # size='hit_distance_sc' if 'hit_distance_sc' in df.columns else None,
                        hover_data=['player_name', 'pitch_name', 'game_date'],
                        title="Launch Angle vs Exit Velocity",
                        labels={'launch_speed': 'Exit Velocity (mph)', 'launch_angle': 'Launch Angle (°)'}
                    )
                    
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            pitch_counts = df['pitch_name'].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=pitch_counts.index,
                values=pitch_counts.values,
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                title='Pitch Type Breakdown'
            )])
            
            fig_pie.update_layout(
                title="Pitch Type Breakdown",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
      
        
        # Row 3: Velocity Distribution and Expected Stats Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            if 'release_speed' in df.columns:
                fig_violin = go.Figure()
                
                for pitch_type in df['pitch_name'].unique()[:6]:  # Limit to top 6 pitch types
                    pitch_data = df[df['pitch_name'] == pitch_type]
                    
                    fig_violin.add_trace(go.Violin(
                        y=pitch_data['release_speed'],
                        name=pitch_type,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                fig_violin.update_layout(
                    title="Velocity by Pitch Type",
                    yaxis_title="Velocity (mph)",
                    height=400
                )
                
                st.plotly_chart(fig_violin, use_container_width=True)
        with col2:
            byPitch = df.groupby('pitch_name').agg({'xStrike': 'mean', 'strike': 'mean'}).reset_index()
            byPitch = byPitch.sort_values(by='xStrike', ascending=False).reset_index(drop=True)
            byPitch['xStrike'] = byPitch['xStrike'].apply(lambda x: round(x, 3))
            fig_bar = go.Figure(data=[go.Bar(
                x=byPitch['pitch_name'],
                y=byPitch['xStrike'],
                name='Expected Strike Probability',
                marker_color='indianred'
            ),go.Bar(
                x=byPitch['pitch_name'],
                y=byPitch['strike'],
                name='Actual Strike Probability',
                marker_color='lightsalmon'
            )])
            fig_bar.update_layout(
                title="Expected vs Actual Strike Probability by Pitch Type",
                yaxis_title="Probability",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
      
