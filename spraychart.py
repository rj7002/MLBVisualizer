import math
from random import randint
import streamlit as st
import numpy as np
import plotly.graph_objects as go
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
date_input = st.date_input('Select a date', value=currentdate)

# Display the selected date in YYYY-MM-DD format
formatted_date = date_input.strftime('%Y-%m-%d')
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

if formatted_date:
    df = statcast(start_dt=formatted_date)
    # st.write(df.columns)
    if not df.empty:
        
        unique_matchups = df.groupby('game_date').apply(
            lambda x: [f"{row['home_team']} vs {row['away_team']}" for index, row in x.iterrows()]
        ).explode().unique()

        # Create a selectbox in Streamlit
        selected_matchup = st.selectbox("Select a Matchup", unique_matchups)
        teams = selected_matchup.split(' vs ')
        hteam = teams[0]
        ateam = teams[1]
        df = df[(df['home_team'] == hteam) & (df['away_team'] == ateam)]
        filterby = st.selectbox('Filter by',['Batter','Pitcher','Hit Type'])
        pitchers = df['player_name'].unique()
        formatted_names = [f"{name.split(', ')[1]} {name.split(', ')[0]}" for name in pitchers]

        if filterby == 'Pitcher':
            # data = pitching_stats_range(date,date2)
            selectp = st.multiselect('Select a pitcher',formatted_names)
            selectp2 = []
            for name in selectp:
                names = name.split(' ')
                selectp2.append(names[1] + ', ' + names[0])
            df = df[df['player_name'].isin(selectp2)] 
            # names = selectp.split(' ')
            # selectp2 = []
            # selectp2 = (names[1] + ', ' + names[0])
            # df = df[df['player_name'] == selectp2]
            df.dropna(subset=['hc_x','hc_y'])
            df = df.drop_duplicates(subset='des')
            df = df[~df['des'].str.contains('walks', case=False, na=False)]
            df = df[~df['des'].str.contains('strike', case=False, na=False)]
            df = df[~df['des'].str.contains('hit by pitch', case=False, na=False)]
            df = df = df[~df['des'].str.contains('strike', case=False, na=False)]
            df = df[~((df['des'].str.contains('ground', case=False, na=False)) & (df['hc_x'] < 50))]
            df = df[~((df['des'].str.contains("fielder's choice", case=False, na=False)))]
            df = df[~((df['des'].str.contains("error", case=False, na=False)))]
            df = df[~((df['des'].str.contains("challenged", case=False, na=False)))]
            df = df[~((df['des'].str.contains("fielding error", case=False, na=False)))]
            df = df[~((df['des'].str.contains("caught", case=False, na=False)))]

            ids = df['pitcher'].unique()
        elif filterby == 'Hit Type':
            hittypes = df['events'].unique()
            selectp = st.multiselect('Select a type of hit',hittypes)
            df = df[df['events'].isin(selectp)]
        else:
            # data = batting_stats_range(date,date2)

            df = df

            # df = df[df['des'].str.contains('Rafael Devers', case=False, na=False)]
            df.dropna(subset=['hc_x','hc_y'])
            df = df.drop_duplicates(subset='hc_x')
            # df = df[~df['des'].str.contains('walks', case=False, na=False)]
            # df = df[~df['des'].str.contains('strike', case=False, na=False)]
            # df = df[~df['des'].str.contains('hit by pitch', case=False, na=False)]
            # df = df = df[~df['des'].str.contains('strike', case=False, na=False)]
            df = df[df['type'] == 'X']
            # df = df[~((df['des'].str.contains('ground', case=False, na=False)) & (df['hc_x'] < 50))]
            # df = df[~((df['des'].str.contains("fielder's choice", case=False, na=False)))]
            # df = df[~((df['des'].str.contains("error", case=False, na=False)))]
            # df = df[~((df['des'].str.contains("challenged", case=False, na=False)))]
            # df = df[~((df['des'].str.contains("fielding error", case=False, na=False)))]
            # df.loc[(206.27-df['hc_y'] > 150), 'hc_y'] = 206.27-randint(148,152)


            # df['color'] = [generate_random_color() for _ in range(len(df))]
            names_list = []

            # Loop through the DataFrame and extract the first two words
            for index, row in df.iterrows():
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
            df = df[df['des'].apply(lambda x: first_two_words(x) in selectp)]
            ids = df['batter'].unique()
        if selectp:
            import pandas as pd

            # df = pd.read_csv('/Users/ryan/Desktop/FantasyPython/giantsstadium.csv')
            unique_pitch_types = df['pitch_name'].unique()

            color_mapping = {pitch_type: generate_random_color() for pitch_type in unique_pitch_types}

            # Add the color mapping to the DataFrame
            df['color'] = df['pitch_name'].map(color_mapping)


            df['hc_x'] = df['hc_x']-125.42
            df['hc_x'] = df['hc_x']*.70
            df['hc_y'] = 206.27-df['hc_y']
            df['hc_y'] = df['hc_y']*.70
            # st.write(df.columns)
            # st.write(df)
            hometeam = df['home_team'].iloc[0].lower()
            if hometeam == 'phi':
                stl_file_path = f'simplify_stadium_{hometeam}.stl'
            else:
                stl_file_path = f'stadium_{hometeam}.stl'


            scatter_points = df[['hc_x', 'hc_y','color']]  # Make sure these columns exist in your DataFrame


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
                pitchers.append(row['player_name'])
                dists.append(206-row['hc_y'])
                pitchtypes.append(row['pitch_name'])
                innings.append(row['inning'])
                colors.append(row['color'])

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
                if launch_angle < 0:
                    launch_angle = 0
                
                # Define the start and end points
                p1 = np.array([x1, y1, z_value])
                p2 = np.array([x2, y2, z_value])
                
                # Adjust the apex height based on the launch angle
                # Example conversion: Adjust this scaling factor as needed
                if 'homer' in des.lower():
                    height_scaling_factor = 0.75 # Adjust this factor to change how launch angle affects height
                else:
                    height_scaling_factor = 0.5
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
            # for i in range(len(x_coords)):
            #     x1 = x_coords[i]
            #     y1 = y_coords[i]
            #     x2 = x_coords2[i]
            #     y2 = y_coords2[i]
            #     des = plays[i]
            #     pitcher = pitchers[i]
            #     ys = dists[i]
            #     pitch = pitchtypes[i]
            #     inning = innings[i]
            #     # Define the start and end points
            #     p1 = np.array([x1, y1, z_value])
            #     p2 = np.array([x2, y2, z_value])
                
            #     # Apex will be above the line connecting p1 and p2
            #     distance = calculate_distance(x1, y1, x2, y2)
            #     if 'homer' in des:
            #         h = randint(12,16) * 50
            #     elif 'homers' in des:
            #         h = randint(12,16) * 50
            #     elif 'grand slam' in des:
            #         h = randint(13,16) * 50
            #     elif 'ground-rule' in des:
            #         h = randint(3,5) * 10
            #     elif 'ground' in des:
            #         h = 0
            #     elif 'line' in des:
            #         h = randint(2,3) * 10
            #     elif 'pop' in des:
            #         h = randint(12,16) * 10
            #     elif 'flies out sharply' in des:
            #         h = randint(6,8) * 10
            #     elif 'flies' in des:
            #         h = randint(12,16) * 10
            #     elif 'on a fly ball' in des:
            #         h = randint(3,5) * 10
            #     elif 'sacrifice fly' in des:
            #         h = randint(12,16) * 10
            #     elif 'triples' in des:
            #         h = randint(0,6) * 10
            #     elif 'doubles' in des:
            #         h = randint(3,6) * 10
            #     elif 'singles on a fly ball' in des:
            #         h = randint(8,10) * 10
            #     elif 'inside-the-park home run' in des:
            #         h = randint(2,6) * 10
            #     elif 'bunt' in des:
            #         h = randint(0,1) * 10
            #     else: 
            #         h = distance
            #     apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
                
            #     # Generate arc points
            #     x, y, z = generate_arc_points(p1, p2, apex)
            if filterby != 'Hit Type':
                selectp = selectp[::-1]
                dfimage = pd.DataFrame({'player_id': ids,'names' : selectp})

                # Create a new column in the layout for images
                cols = st.columns(len(dfimage))

                # Display player images in separate columns
                for col, row in zip(cols, dfimage.itertuples()):
                    with col:
                            display_player_image(row.player_id,200,'')
            import matplotlib.pyplot as plt

            # Extract unique pitches and their corresponding colors
            unique_pitches = df[['pitch_name', 'color']].drop_duplicates()
            
            # Create legend handles
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=row['color'], markersize=10, label=row['pitch_name'])
                for _, row in unique_pitches.iterrows()
            ]
            
            # Create a figure for the legend with transparent background
            legendfig, ax = plt.subplots(figsize=(6, 6), facecolor='none')  # 'none' ensures transparency
            ax.axis('off')  # Turn off the axes
            
            # Add the legend to the figure
            legend = ax.legend(handles=handles, title="Pitch Legend", loc='upper center', frameon=False)  # `frameon=False` for no box
            
            # Display the legend figure in Streamlit
            col1, col2 = st.columns(2)
            st.pyplot(legendfig, transparent=True)
            with col1:
                st.plotly_chart(fig,use_container_width=True)


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
            df2 = df[df['type'] == 'X']
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
            def plot_curve(x_start, y_start, z_start, x_end, y_end, z_end, pfx_x, pfx_z, pitch_name):
                t = np.linspace(0, 1, 100)  # 100 points for smooth curve
                
                # Adjust pfx_x and pfx_z based on pitch_name
                if pitch_name in ["Curveball", "Knuckle Curve"]:
                    pfx_x = -pfx_x  # Flip the effect along the x-axis
                    pfx_z = -pfx_z  # Optionally, flip the effect along the z-axis if needed

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
                des2 = plays[i]

                
                x_curve, y_curve, z_curve = plot_curve(x_start, y_start, z_start, x_end, y_end, z_end, pfx_x, pfx_z,pitch2)
                
                fig.add_trace(go.Scatter3d(
                    x=x_curve,
                    y=y_curve,
                    z=z_curve,
                    mode='lines',
                    line=dict(color=df2['color'].iloc[i], width=4),
                    name=f'Pitch Path {i}',
                    hoverinfo='text',
                    hovertext=f'{des2}<br>{pitch2}<br>Inning: {inning2}<br>Pitcher: {pitcher2}',
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
                        range=[0, 18],
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
    else:
        st.error('No data found')
