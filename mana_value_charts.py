import pathlib as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

file = pl.Path("mana_value_simulation_results.csv")
df = pd.read_csv(file, engine='c')

df = df[
    (df['off_color_pips'] == 0) 
    & (df['mana_value'] == 3) 
    ]
print(df.head())

# Create a scatter plot with the following settings:
# - x-axis: 'average_cast_turn'
# - y-axis: 'on_color_pips'
# - color: 'on_color_land_count'
# - size: 'mana_value'
# - allow filtering by 'on_color_land_count' and 'on_color_pips'
fig = px.scatter(
    df,
    x='average_cast_turn',
    y='on_color_pips',
    color='on_color_land_count',
    hover_data=['on_color_land_count', 'on_color_pips'],
    title='Mana Value Simulation Results, Off Color Pips = 0',
    labels={
        'average_cast_turn': 'Average Cast Turn',
        'on_color_pips': 'On Color Pips',
        'on_color_land_count': 'On Color Land Count',
        'mana_value': 'Mana Value'
    },
    size='mana_value',
)

land_buttons = []
land_buttons.append(
    dict(
        label='All',
        method='update',
        args=[{'visible': [True] * len(df)}]
    )
)
for land_count in df['on_color_land_count'].unique():
    land_buttons.append(
        dict(
            label=str(land_count),
            method='update',
            args=[{'visible': [land_count == lc for lc in df['on_color_land_count']]}]
        )
    )

mana_buttons = []
mana_buttons.append(
    dict(
        label='All',
        method='update',
        args=[{'visible': [True] * len(df)}]
    )
)
for mana_value in df['mana_value'].unique():
    mana_buttons.append(
        dict(
            label=str(mana_value),
            method='update',
            args=[{'visible': [mana_value == mv for mv in df['mana_value']]}]
        )
    )

fig.update_layout(
    xaxis_title='Average Cast Turn',
    yaxis_title='On Color Pips',
    legend_title='On Color Land Count',
    # updatemenus=[
    #     dict(
    #         type='dropdown',
    #         buttons=land_buttons,
    #         direction='down',
    #         xanchor='left',
    #         y=1.00,
    #         yanchor='top',
    #     ),
    #     dict(
    #         type='dropdown',
    #         buttons=mana_buttons,
    #         direction='down',
    #         showactive=True,
    #         xanchor='left',
    #         y=0.9,
    #         yanchor='top',
    #     )
    # ]
)
fig.show()