from matplotlib import patches, pyplot as plt
from pybaseball import statcast
from random import randint
from pybaseball import pitching_stats_range
from pybaseball import batting_stats_range

# Fetch the DataFrame for the specified date range
# df = statcast(start_dt="2024-06-24", end_dt="2024-06-25")

# # Save the DataFrame to a CSV file
# df.to_csv('sb.csv', index=False)
import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import math
import pandas as pd
from datetime import datetime,timedelta
import requests
from scipy.interpolate import CubicSpline
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
# Get the current date and time
now = datetime.now()
previous_date = now - timedelta(days=1)


# Format the date as yyyy-mm-dd
currentdate = previous_date.strftime('%Y-%m-%d')

date = st.text_input('Enter a start date',placeholder='YYYY-MM-DD',value=currentdate)
date2 = st.text_input('Enter an end date',placeholder='YYYY-MM-DD',value=currentdate)
df = statcast(start_dt=date, end_dt=date2)

# Save the DataFrame to a CSV file
df.to_csv('sb.csv', index=False)


df = pd.read_csv('sb.csv')
df = df.fillna(0)
# df2 = df.tail(1000)
pitchers = df['player_name'].unique()
formatted_names = [f"{name.split(', ')[1]} {name.split(', ')[0]}" for name in pitchers]
filterby = st.selectbox('Filter by',['Batter','Pitcher','Type Hit'])
if filterby == 'Pitcher':
    data = pitching_stats_range(date,date2)
    selectp = st.multiselect('Select a pitcher',formatted_names)
    selectp2 = []
    for name in selectp:
        names = name.split(' ')
        selectp2.append(names[1] + ', ' + names[0])
    df2 = df[df['player_name'].isin(selectp2)] 
    df2.dropna(subset=['hc_x','hc_y'])
    df2 = df2.drop_duplicates(subset='des')
    df2 = df2[~df2['des'].str.contains('walks', case=False, na=False)]
    df2 = df2[~df2['des'].str.contains('strike', case=False, na=False)]
    df2 = df2[~df2['des'].str.contains('hit by pitch', case=False, na=False)]
    df2 = df2 = df2[~df2['des'].str.contains('strike', case=False, na=False)]
    df2 = df2[~((df2['des'].str.contains('ground', case=False, na=False)) & (df2['hc_x'] < 50))]
    df2 = df2[~((df2['des'].str.contains("fielder's choice", case=False, na=False)))]
    df2 = df2[~((df2['des'].str.contains("error", case=False, na=False)))]
    df2 = df2[~((df2['des'].str.contains("challenged", case=False, na=False)))]
    df2 = df2[~((df2['des'].str.contains("fielding error", case=False, na=False)))]
    df2 = df2[~((df2['des'].str.contains("caught", case=False, na=False)))]

    ids = df2['pitcher'].unique()
elif filterby == 'Type Hit':
    hittypes = df['events'].unique()
    typehit = st.multiselect('Select a type of hit',hittypes)
    df2 = df[df['events'].isin(typehit)]
else:
    data = batting_stats_range(date,date2)

    df2 = df

    # df2 = df[df['des'].str.contains('Rafael Devers', case=False, na=False)]
    df2.dropna(subset=['hc_x','hc_y'])
    df2 = df2.drop_duplicates(subset='des')
    # df2 = df2[~df2['des'].str.contains('walks', case=False, na=False)]
    # df2 = df2[~df2['des'].str.contains('strike', case=False, na=False)]
    # df2 = df2[~df2['des'].str.contains('hit by pitch', case=False, na=False)]
    # df2 = df2 = df2[~df2['des'].str.contains('strike', case=False, na=False)]
    df2 = df2[df2['type'] == 'X']
    # df2 = df2[~((df2['des'].str.contains('ground', case=False, na=False)) & (df2['hc_x'] < 50))]
    # df2 = df2[~((df2['des'].str.contains("fielder's choice", case=False, na=False)))]
    # df2 = df2[~((df2['des'].str.contains("error", case=False, na=False)))]
    # df2 = df2[~((df2['des'].str.contains("challenged", case=False, na=False)))]
    # df2 = df2[~((df2['des'].str.contains("fielding error", case=False, na=False)))]
    # df2.loc[(206.27-df2['hc_y'] > 150), 'hc_y'] = 206.27-randint(148,152)


    # df2['color'] = [generate_random_color() for _ in range(len(df2))]
    names_list = []

    # Loop through the DataFrame and extract the first two words
    for index, row in df2.iterrows():
        text = row['des']
        words = text.split()
        first_two_words = ' '.join(words[:2])
        names_list.append(first_two_words)
    unique_names_set = set(names_list)

    # Convert the set back to a list if needed
    unique_names_list = list(unique_names_set)
    selectp = st.multiselect('Select a player',unique_names_list)
    def first_two_words(text):
        words = text.split()
        return ' '.join(words[:2])

    # Apply the function to the 'des' column and filter based on the variable
    df2 = df2[df2['des'].apply(lambda x: first_two_words(x) in selectp)]
    ids = df2['batter'].unique()

# st.write(len(df2))

# df2 = df2.head(50)
# dfg = dfg.head(50)
df2['z'] = 0
unique_pitch_types = df2['pitch_name'].unique()

# Generate a color for each pitch type
color_mapping = {pitch_type: generate_random_color() for pitch_type in unique_pitch_types}

# Add the color mapping to the DataFrame
df2['color'] = df2['pitch_name'].map(color_mapping)
# dfg['z'] = 0
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
for index, row in df2.iterrows():
    # if 'homer' in row['des']:
    #     x_values.append(row['hc_x']*3)
    # elif 'pop' in row['des']:
    #     x_values.append(row['hc_x']*3)
    # else:
    # # Append the value from column 'x' to the list
    if 206.26-row['hc_y'] > 150:
        row['hc_y'] = randint(58,60)
    if 'homer' in row['des'] or 'grand slam' in row['des']:
        row['hc_y'] = randint(35,40)


    x_values.append(row['hc_x']-125.42)
    y_values.append(206.27-row['hc_y'])
    z_values.append(0)
    plays.append(row['des'])
    pitchers.append(row['player_name'])
    dists.append(206-row['hc_y'])
    pitchtypes.append(row['pitch_name'])
    colors.append(row['color'])
    innings.append(row['inning'])
x_values2 = []
y_values2 = []
z_values2 = []
# Loop through each row in the 'location' column
for index, row in df2.iterrows():
    # Append the value from column 'x' to the list
    x_values2.append(row['plate_x'])
    y_values2.append(0)
    z_values2.append(row['plate_z'])

x_valuesg = []
y_valuesg = []
z_valuesg = []
# Loop through each row in the 'location' column
# for loc_list in dfg['location']:
#     for coord in loc_list:
#         x = loc_list[0]
#         y=loc_list[1]
#         x_valuesg.append(x)
#         y_valuesg.append(y)
#         z_valuesg.append(0)

# x_values2g = []
# y_values2g = []
# z_values2g = []
# # Loop through each row in the 'location' column
# for loc_list in dfg['pass_end_location']:
#     for coord in loc_list:
#         x = loc_list[0]
#         y=loc_list[1]
#         x_values2g.append(x)
#         y_values2g.append(y)
#         z_values2g.append(0)
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
# x_coords2g = x_values2g
# y_coords2g = y_values2g

# Create figure
fig = go.Figure()

# Loop through pairs of points to create arcs
launch_angles = df2['launch_angle'].tolist()

plays2 = len(plays)
for i in range(len(x_coords)):
    des = plays[i]
    pitcher = pitchers[i]
    ys = dists[i]
    pitch = pitchtypes[i]
    color = colors[i]
    inning = innings[i]
    x1 = x_coords[i]
    y1 = y_coords[i]
    x2 = x_coords2[i]
    y2 = y_coords2[i]
    launch_angle = launch_angles[i]
    
    # Define the start and end points
    p1 = np.array([x1, y1, z_value])
    p2 = np.array([x2, y2, z_value])
    
    # Adjust the apex height based on the launch angle
    # Example conversion: Adjust this scaling factor as needed
    height_scaling_factor = 0.1  # Adjust this factor to change how launch angle affects height
    h = height_scaling_factor * np.tan(np.radians(launch_angle)) * np.linalg.norm(p2 - p1)
    
    # Adjust the apex position based on the calculated height
    apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])
    
    # Generate arc points
    x, y, z = generate_arc_points(p1, p2, apex)
    
    # Add arc trace to figure
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(width=5,color=color),
        name=f'Arc {i}',
        hoverinfo='text',
        hovertext=f'{des}<br>Pitch Type: {pitch}<br>Inning: {inning}<br>Pitcher: {pitcher}',

    ))
for i in range(len(x_coords)):
    x1 = x_coords[i]
    y1 = y_coords[i]
    x2 = x_coords2[i]
    y2 = y_coords2[i]
    des = plays[i]
    pitcher = pitchers[i]
    ys = dists[i]
    pitch = pitchtypes[i]
    color = colors[i]
    inning = innings[i]
    # Define the start and end points
    p1 = np.array([x1, y1, z_value])
    p2 = np.array([x2, y2, z_value])
    
    # Apex will be above the line connecting p1 and p2
    distance = calculate_distance(x1, y1, x2, y2)
    if 'homer' in des:
        h = randint(12,16)
    elif 'homers' in des:
        h = randint(12,16)
    elif 'grand slam' in des:
        h = randint(13,16)
    elif 'ground-rule' in des:
        h = randint(3,5)
    elif 'ground' in des:
        h = 0
    elif 'line' in des:
        h = randint(2,3)
    elif 'pop' in des:
        h = randint(12,16)
    elif 'flies out sharply' in des:
        h = randint(6,8)
    elif 'flies' in des:
        h = randint(12,16)
    elif 'on a fly ball' in des:
        h = randint(3,5)
    elif 'sacrifice fly' in des:
        h = randint(12,16)
    elif 'triples' in des:
        h = randint(0,6)
    elif 'doubles' in des:
        h = randint(3,6)
    elif 'singles on a fly ball' in des:
        h = randint(8,10)
    elif 'inside-the-park home run' in des:
        h = randint(2,6)
    elif 'bunt' in des:
        h = randint(0,1)
    else: 
        h = distance
    apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
    
    # Generate arc points
    x, y, z = generate_arc_points(p1, p2, apex)
    
    # Add arc trace
    radius_x = 180   # Radius along the x-axis
    radius_y = 60   # Radius along the y-axis
    center_z = 0     # Z coordinate for both start and end of the semicircle

    # Generate points for the semicircle
    theta = np.linspace(0, np.pi, 100)  # Angle from 0 to pi (semicircle)
    x1 = radius_x * np.cos(theta)        # X coordinates (renamed)
    y1 = radius_y * np.sin(theta)        # Y coordinates (renamed)
    max_y1 = np.max(y1)

# Calculate the maximum value of y1 + 100
    max_y1_plus_100 = max_y1 + 100
    # st.write(max_y1_plus_100)

    # Add the semicircle to the plot

    fig.add_trace(go.Scatter3d(
        x=x1,                           # Use x1 for x coordinates
        y=y1+100,                           # Use y1 for y coordinates
        z=np.full_like(x1, center_z),   # Z coordinates are constant
        mode='lines',
        hoverinfo='none',

        line=dict(color='white', width=4),
        name='Oval Semicircle'
    ))

    start = 0
    stop = 0.8
    step = 0.01

# Generate values using numpy.arange()
    fig.add_trace(go.Scatter3d(
        x=[8,188],
        y=[0,100],
        z=[0,0],
        mode='lines',
        hoverinfo='none',

        line=dict(width=25,color='#6d451f')
        # name=f'Endpoints {i + 1}'
    ))
    fig.add_trace(go.Scatter3d(
        x=[-8,-188],
        y=[0,100],
        z=[0,0],
        mode='lines',
        hoverinfo='none',

        line=dict(width=25,color='#6d451f')
        # name=f'Endpoints {i + 1}'
    ))
    fig.add_trace(go.Scatter3d(
        x=x1,                           # Use x1 for x coordinates
        y=y1+95,                           # Use y1 for y coordinates
        z=np.full_like(x1, 0),   # Z coordinates are constant
        mode='lines',
        hoverinfo='none',
        line=dict(color='#6d451f', width=20),
        name='Oval Semicircle'))
    for value in np.arange(start, stop + step, step):
        fig.add_trace(go.Scatter3d(
        x=x1,                           # Use x1 for x coordinates
        y=y1+100,                           # Use y1 for y coordinates
        z=np.full_like(x1, value),   # Z coordinates are constant
        mode='lines',
        hoverinfo='none',
        line=dict(color='gray', width=4),
        name='Oval Semicircle'
    ))
    radius_x_small = 70   # Radius along the x-axis for the smaller semicircle
    radius_y_small = 25   # Radius along the y-axis for the smaller semicircle
    y_center_small = 80   # Y coordinate for the center of the smaller semicircle

    # Generate points for the smaller semicircle
    theta_small = np.linspace(0, np.pi, 100)  # Angle from 0 to pi (semicircle)
    x_small = radius_x_small * np.cos(theta_small)        # X coordinates for the smaller semicircle
    y_small = radius_y_small * np.sin(theta_small) + y_center_small  # Y coordinates shifted to y = 80
    fig.add_trace(go.Scatter3d(
    x=x_small,                               # X coordinates for the smaller semicircle
    y=y_small-40,                               # Y coordinates for the smaller semicircle
    z=np.full_like(x_small, center_z),       # Z coordinates are constant
    mode='lines',
    hoverinfo='none',

    line=dict(color='#6d451f', width=4),        # Different color for the smaller semicircle
    name='Smaller Semicircle'
))
    for i in range(1,11): 
        radius_x_small = 70-i   # Radius along the x-axis for the smaller semicircle
        
        radius_y_small = 25-i   # Radius along the y-axis for the smaller semicircle
        y_center_small = 80   # Y coordinate for the center of the smaller semicircle

        # Generate points for the smaller semicircle
        theta_small = np.linspace(0, np.pi, 100)  # Angle from 0 to pi (semicircle)
        x_small = radius_x_small * np.cos(theta_small)        # X coordinates for the smaller semicircle
        y_small = radius_y_small * np.sin(theta_small) + y_center_small  # Y coordinates shifted to y = 80

        fig.add_trace(go.Scatter3d(
        x=x_small,                               # X coordinates for the smaller semicircle
        y=y_small-40-i,                               # Y coordinates for the smaller semicircle
        z=np.full_like(x_small, center_z),       # Z coordinates are constant
        mode='lines',
        hoverinfo='none',

        line=dict(color='#6d451f', width=40),        # Different color for the smaller semicircle
        name='Smaller Semicircle'
    ))
    for i in range(1,15):
        fig.add_trace(go.Scatter3d(
            x=[40+i,0+i],
            y=[24.5+i,44.5+i],
            z=[0,0],
            mode='lines',
            hoverinfo='none',

            line=dict(width=25,color='#6d451f')
            # name=f'Endpoints {i + 1}'
        ))
        fig.add_trace(go.Scatter3d(
            x=[-40-i,0-i],
            y=[24.5+i,44.5+i],
            z=[0,0],
            mode='lines',
            hoverinfo='none',

            line=dict(width=25,color='#6d451f')
            # name=f'Endpoints {i + 1}'
        ))
    fig.add_trace(go.Scatter3d(
        x=[0,180],
        y=[0,100],
        z=[0,0],
        mode='lines',
        hoverinfo='none',

        line=dict(width=6,color='white')
        # name=f'Endpoints {i + 1}'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0,-180],
        y=[0,100],
        z=[0,0],
        mode='lines',
        hoverinfo='none',

        line=dict(width=6,color='white')
        # name=f'Endpoints {i + 1}'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[20],
        z=[0],
        mode='markers',
        marker=dict(size=10, color='tan'),
        # name=f'Endpoints {i + 1}'
        hoverinfo='none',
        # hovertext=pitcher
    ))
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[20],
        z=[0],
        mode='markers',
        marker=dict(size=6, color='white',symbol='square'),
        # name=f'Endpoints {i + 1}'
        hoverinfo='none',
    ))
    fig.add_trace(go.Scatter3d(
        x=[40],
        y=[22.5],
        z=[0.05],
        mode='markers',
        marker=dict(size=6, color='white',symbol='square'),
        # name=f'Endpoints {i + 1}'
        hoverinfo='none',
        hovertext='First Base'
    ))
    fig.add_trace(go.Scatter3d(
        x=[-40],
        y=[22.5],
        z=[0.05],
        mode='markers',
        marker=dict(size=6, color='white',symbol='square'),
        # name=f'Endpoints {i + 1}'
        hoverinfo='none',
        hovertext='Third Base'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[42.5],
        z=[0.05],
        mode='markers',
        marker=dict(size=6, color='white',symbol='square'),
        # name=f'Endpoints {i + 1}'
        hoverinfo='none',
        hovertext='Second Base'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0.05],
        mode='markers',
        marker=dict(size=6, color='white',symbol='square'),
        # name=f'Endpoints {i + 1}'
        hoverinfo='none',
        hovertext='Home Base'
    ))
    random_color = generate_random_color()
    # fig.add_trace(go.Scatter3d(
    #     x=x, y=y, z=z,
    #     mode='lines',
    #     line=dict(width=6,color=color),
    #     name=f'Arc {i + 1}',
    #     hoverinfo='text',
    #     hovertext=f'{des}<br>Pitch Type: {pitch}<br>Inning: {inning}<br>Pitcher: {pitcher}<br>{ys}',

    #     # opacity=0.5
    # ))
    
    # Add start and end points
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p1[0]],
        y=[p1[1], p1[1]],
        z=[0.05, 0.05],
        mode='markers',
        marker=dict(size=2, color='white'),
        # name=f'Endpoints {i + 1}'
        hoverinfo='text',
        hovertext=f'{des}<br>Pitch Type: {pitch}<br>Inning: {inning}<br>Pitcher: {pitcher}',
    ))
#     fig.add_trace(go.Scatter3d(
#     x=[p1[0]],  # X coordinates
#     y=[p1[1]],  # Y coordinates
#     z=[0],    # Z coordinates
#     mode='text',  # Display both markers and text
#     text='⚾',  # Text labels (emoji or symbol)
#     textposition='middle center',  # Position of text relative to the marker
#     hoverinfo='text',
#       textfont=dict(
#         size=8,  # Adjust the size of the text
#     ),  # Display text on hover
#     hovertext=f'{des} ({str(ys)} ft)'

# ))

# Update layout
fig.update_layout(
    title='',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)
fig.update_layout(
   
               
    height=1200,
    scene=dict(
        xaxis=dict(
            title='',
            range=[-200, 200],
              showticklabels=False,
                        showgrid=False,  # Set the range for the x-axis
        ),
        yaxis=dict(
            title='',
            range=[-10, 180] ,
             showticklabels=False,
                        showgrid=False, # Set the range for the y-axis
        ),
        zaxis=dict(
            title='',
            range=[0, 18],
             showbackground=True,
                        backgroundcolor='#006400',
                        showticklabels=False,
                        showgrid=False,
        )

    ),
    title='3D Baseball Hit Chart',

     showlegend=False
)

# st.subheader(f'{hitters} Hits Chart')
col1, col2 = st.columns(2)
with col1:
    if filterby != 'Type Hit':
        for id in reversed(ids):
            display_player_image(id,250,'')
    st.plotly_chart(fig,use_container_width=True)

pitch_codes = ["FF", "CU", "CH", "FC", "EP", "FO", "KN", "KC", "SC", "SI", "SL", "FS", "FT", "ST", "SV", "SIFT", "CUKC", "ALL"] # note: all doesn't work in words, we'll have some special handling
pitch_names = ["4-Seamer", "Curveball", "Changeup", "Cutter", "Eephus", "Forkball", "Knuckleball", "Knuckle-curve", "Screwball", "Sinker", "Slider", "Splitter", "2-Seamer", "Sweeper", "Slurve", "Sinker", "Curveball"]
pitch_names_upper = [p.upper() for p in pitch_names]

# including all the codes to themselves makes this simpler later
pitch_name_to_code_map = dict(zip(pitch_codes + pitch_names_upper, pitch_codes + pitch_codes))
pitch_code_to_name_map = dict(zip(pitch_codes, pitch_names))

def plot_strike_zone(data: pd.DataFrame, title: str = '', colorby: str = 'pitch_type', legend_title: str = '',
                     annotation: str = 'pitch_type',axis=None):  
    # some things to auto adjust formatting
    # make the markers really visible when fewer pitches
    alpha_markers = min(0.8, 0.5 + 1 / data.shape[0])
    alpha_text = alpha_markers + 0.2
    
    # define Matplotlib figure and axis
    if axis is None:
        fig, axis = plt.subplots()

    # add home plate to plot 
    home_plate_coords = [[-0.71, 0], [-0.85, -0.5], [0, -1], [0.85, -0.5], [0.71, 0]]
    axis.add_patch(patches.Polygon(home_plate_coords,
                                   edgecolor = 'darkgray',
                                   facecolor = 'lightgray',
                                   zorder = 0.1))
    
    # add strike zone to plot, technically the y coords can vary by batter
    axis.add_patch(patches.Rectangle((-0.71, 1.5), 2*0.71, 2,
                 edgecolor = 'lightgray',
                 fill=False,
                 lw=3,
                 zorder = 0.1))
    
    # legend_title = ""
    color_label = ""
    
    # to avoid the SettingWithCopyWarning error
    sub_data = data.copy().reset_index(drop=True)
    if colorby == 'pitch_type':
        color_label = 'pitch_type'
        
        if not legend_title:
            legend_title = 'Pitch Type'
            
    elif colorby == 'description':
        values = sub_data.loc[:, 'description'].str.replace('_', ' ').str.title()
        sub_data.loc[:, 'desc'] = values
        color_label = 'desc'
        
        if not legend_title:
            legend_title = 'Pitch Description'
    elif colorby == 'pitcher':
        color_label = 'player_name'
        
        if not legend_title:
            legend_title = 'Pitcher'
            
    elif colorby == "events":
        # only things where something happened
        sub_data = sub_data[sub_data['events'].notna()]
        sub_data['event'] = sub_data['events'].str.replace('_', ' ').str.title()
        color_label = 'event'
        
        if not legend_title:
            legend_title = 'Outcome'
    
    else:
        color_label = colorby
        if not legend_title:
            legend_title = colorby
        
    scatters = []
    for color in sub_data[color_label].unique():
        color_sub_data = sub_data[sub_data[color_label] == color]
        scatters.append(axis.scatter(
            color_sub_data["plate_x"],
            color_sub_data['plate_z'],
            s = 10**2,
            label = pitch_code_to_name_map[color] if color_label == 'pitch_type' else color,
            alpha = alpha_markers
        ))
        
        # add an annotation at the center of the marker
        if annotation:
            for i, pitch_coord in zip(color_sub_data.index, zip(color_sub_data["plate_x"], color_sub_data['plate_z'])):
                label_formatted = color_sub_data.loc[i, annotation]
                label_formatted = label_formatted if not pd.isna(label_formatted) else ""
                
                # these are numbers, format them that way
                if annotation in ["release_speed", "effective_speed", "launch_speed"] and label_formatted != "":
                    label_formatted = "{:.0f}".format(label_formatted)
                
                axis.annotate(label_formatted,
                            pitch_coord,
                            size = 7,
                            ha = 'center',
                            va = 'center',
                            alpha = alpha_text)

    axis.set_xlim(-4, 4)
    axis.set_ylim(-1.5, 7)
    axis.axis('off')

    axis.legend(handles=scatters, title=legend_title, bbox_to_anchor=(0.7, 1), loc='upper left')
    
    plt.title(title)
    return plt
# fig = plot_strike_zone(data=df2)
# st.pyplot(fig,use_container_width=True)



import plotly.graph_objects as go
import numpy as np


# Create a 3D scatter plot
fig = go.Figure()

# Plot release points
# fig.add_trace(go.Scatter3d(
#     x=df2['release_pos_x'],
#     y=[40] * len(df2),  # y is always 15 for release points
#     z=df2['release_pos_z'],
#     mode='markers',
#     marker=dict(size=6, color=df2['color']),
#     name='Release Point'
# ))

# Plot plate positions
df2 = df2[df2['type'] == 'X']
fig.add_trace(go.Scatter3d(
    x=df2['plate_x'],
    y=[0] * len(df2),  # y is always 0 for plate positions
    z=df2['plate_z'],
    mode='markers',
    marker=dict(size=8, color=df2['color']),
    name='Plate Position',
     hoverinfo='text',
     hovertext=df2['pitch_name']
))
# Add lines connecting release points to adjusted plate positions
def plot_curve(x_start, y_start, z_start, x_end, y_end, z_end, pfx_x, pfx_z):
    t = np.linspace(0, 1, 100)  # 100 points for smooth curve
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
    
    x_curve, y_curve, z_curve = plot_curve(x_start, y_start, z_start, x_end, y_end, z_end, pfx_x, pfx_z)
    
    fig.add_trace(go.Scatter3d(
        x=x_curve,
        y=y_curve,
        z=z_curve,
        mode='lines',
        line=dict(color=df2['color'].iloc[i], width=4),
        name=f'Pitch Path {i}',
        hoverinfo='text',
        hovertext=f'{pitch2}<br>Inning: {inning2}<br>Pitcher: {pitcher2}',
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
    title='3D Baseball Pitch Chart'
)
fig.update_layout(
   
               
    height=1200,
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
            range=[0, 18],
            # showbackground=True,
            # backgroundcolor='#006400',

                        showticklabels=False,
                        showgrid=False,
        )

    ),
     showlegend=False
)
# Show plot
import streamlit as st
with col2:
    data = data[data['Name'].isin(selectp)]
    if filterby != 'Pitcher':
        for index, row in data.iterrows():
            st.write(f"Name: {row['Name']}")
            st.write(f"Age: {row['Age']}")
            st.write(f"Plate Appearances: {row['PA']}")
            st.write(f"At Bats: {row['AB']}")
            st.write(f"Hits: {row['H']}")
            st.write(f"Batting Average: {row['BA']}")
            st.write(f"OBP: {row['OBP']}")
            st.write(f"SLG: {row['SLG']}")
            st.write(f"OPS: {row['OPS']}")
    else:
        for index, row in data.iterrows():
            st.write(f"Name: {row['Name']}")
            st.write(f"Age: {row['Age']}")
            st.write(f"Games: {row['G']}")
            st.write(f"Wins: {row['W']}")
            st.write(f"Losses: {row['L']}")
            st.write(f"Innings Pitched: {row['IP']}")
            st.write(f"Strikeouts: {row['SO']}")
            st.write(f"ERA: {row['ERA']}")
            st.write(f"WHIP: {row['WHIP']}")
    st.plotly_chart(fig)
    







# fig.show()
# import k3d
# import vtk
# import ipywidgets as widgets

# reader = vtk.vtkGLTFReader() 
# reader.SetFileName('/Users/ryan/Downloads/Stadium_MIL.glb')
# reader.Update() 

# plot = k3d.plot()
# mb = reader.GetOutput()

# iterator = mb.NewIterator()

# vtk_polyobjects = []
# while not iterator.IsDoneWithTraversal():
#     item = iterator.GetCurrentDataObject()
#     vtk_polyobjects.append(item)
#     iterator.GoToNextItem()

    
# for obj in vtk_polyobjects:
#     plot += k3d.vtk_poly_data(obj, color=0x222222)
# plot.display()

# debug_info = widgets.HTML()
# import pyvista as pv
# import pandas as pd
# import numpy as np
# import streamlit as st

# # Load the mesh
# mesh = pv.read('/Users/ryan/Downloads/Stadium_MIL.glb')

# # Read and preprocess data
# df = pd.read_csv('/Users/ryan/Desktop/FantasyPython/sb.csv')
# df = df.dropna(subset=['hc_x'])
# df = df[~df['des'].str.contains('ground', case=False, na=False)]
# df = df[~df['des'].str.contains('walk', case=False, na=False)]
# df = df[~df['des'].str.contains('strike', case=False, na=False)]
# df = df[~df['des'].str.contains('hit by pitch', case=False, na=False)]

# df2 = df.head(100)

# # Adjust values
# df2['hc_z'] = df['hc_y']
# x_values = (df2['hc_x'] - 125.42).tolist()
# y_values = (198.27 - df2['hc_z']).tolist()
# z_values = [0] * len(df2)  # Adding z=0 for all points

# # Create PyVista plotter
# plotter = pv.Plotter()

# # Add the mesh to the plotter
# plotter.add_mesh(mesh, color='white')

# # Add scatter plot of hc_x and hc_y
# scatter_points = np.array([x_values, y_values, z_values]).T
# scatter = pv.PolyData(scatter_points)
# plotter.add_mesh(scatter, color='blue', point_size=10, render_points_as_spheres=True)

# # Set up the camera position
# plotter.camera_position = 'iso'

# # Show the plot in Streamlit
# plotter.show()

# # import streamlit as st
# # import pyvista as pv
# # import pandas as pd
# # import numpy as np
# # from random import randint
# # import pyvista as pv
# # from stpyvista import stpyvista
# # # Example DataFrame
# # # Replace this with your actual DataFrame
# # data = {
# #     'hc_x': np.random.rand(100) * -100,
# #     'hc_z': np.random.rand(100) * 0
# # }
# # df = pd.DataFrame(data)
# # # df.to_csv('test.csv')
# # df2 = pd.read_csv('/Users/ryan/Desktop/FantasyPython/sb_normalized.csv')
# # df3 = pd.read_csv('sb.csv')
# # # df2 = df2.head(100)
# # df['hc_x'] = df2['hc_x']
# # df['hc_z'] = df2['hc_y']
# # df['hc_z'] = df['hc_z']
# # df['hc_x'] = df['hc_x']-60
# # df['des'] = df3['des']

# # # df.to_csv('/Users/ryan/Desktop/FantasyPython/test.csv')
# # # df = pd.read_csv('/Users/ryan/Desktop/FantasyPython/test.csv')

# # # Add a column for z with value 0
# # df['hc_y'] = 0
# # df.to_csv('test.csv')
# # # Convert DataFrame to NumPy array
# # df2 = pd.read_csv('sb.csv')
# # scatter_points = df[['hc_z', 'hc_y', 'hc_x']].to_numpy()
# # labels = df['des'].tolist()

# # # Load the mesh
# # mesh = pv.read('/Users/ryan/Downloads/Stadium_SF.glb')

# # # Create a PyVista plotter
# # plotter = pv.Plotter()

# # # Add the mesh to the plotter
# # plotter.add_mesh(mesh)

# # # Add the scatter plot to the plotter
# # scatter = pv.PolyData(scatter_points)
# # plotter.add_mesh(scatter, color='red', point_size=15, render_points_as_spheres=True)

# # plotter.add_point_labels(scatter, labels, font_size=10, point_size=15)

# # # Optionally, set up the camera position
# # plotter.camera_position = 'xy'

# # # Show the plot
# # # stpyvista(plotter)
# # plotter.show()

# # # hc_x: 15 to -120 front and back
# # # hc_z: -60 t0 60left and right

# # # import pandas as pd
# # # import numpy as np

# # # # Read the CSV file into a DataFrame
# # # df = pd.read_csv('sb.csv')

# # # # Define the new bounds
# # # new_x_min, new_x_max = -60, 60
# # # new_y_min, new_y_max = -120, 15

# # # # Find the original bounds
# # # original_x_min, original_x_max = df['hc_x'].min(), df['hc_x'].max()
# # # original_y_min, original_y_max = df['hc_y'].min(), df['hc_y'].max()

# # # # Normalize the coordinates to the new range
# # # df['hc_x_normalized'] = np.interp(df['hc_x'], 
# # #                                    (original_x_min, original_x_max), 
# # #                                    (new_x_min, new_x_max))

# # # df['hc_y_normalized'] = np.interp(df['hc_y'], 
# # #                                    (original_y_min, original_y_max), 
# # #                                    (new_y_min, new_y_max))

# # # # Optionally, save the adjusted DataFrame to a new CSV file
# # # df.to_csv('sb_normalized.csv', index=False)
