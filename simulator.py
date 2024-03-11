import streamlit as st
import pandas as pd
import numpy as np
from radcad import Model, Simulation, Experiment
from radcad.engine import Engine, Backend
import plotly.express as px
import plotly.graph_objects as go

from model.classes import Entity
from model.logic import generate_dynamic_functions

from model.system_parameters import system_parameters as default_system_parameters
from model.initial_variables import setup_initial_state

st.set_page_config(layout="wide")

# Display an image with the center class applied
col00, col01, col02 = st.columns([1, 2, 1])

with col00:
    st.write("")  

with col01:
    
    st.image("images/logo2.svg", use_column_width=True)

with col02:
    st.write("")


def simulate(system_parameters,
             timesteps=5 * 52,
             runs=1,
             start_date=pd.to_datetime("2024-01-01"),
             total_supply=1_000_000_000,
             multiplier=0.4):
    entity_names = [
        entity['name'] for entity in system_parameters['entities'][0]
    ]
    policy_functions, state_update_functions = generate_dynamic_functions(
        entity_names)

    state_update_blocks = [{
        'policies': policy_functions,
        'variables': state_update_functions,
    }]

    initial_state = setup_initial_state(system_params=system_parameters)
    model = Model(initial_state=initial_state,
                  state_update_blocks=state_update_blocks,
                  params=system_parameters)
    simulation = Simulation(model=model, timesteps=timesteps, runs=runs)
    experiment = Experiment([simulation])
    experiment.engine = Engine(backend=Backend.PATHOS)

    results = experiment.run()
    df = pd.DataFrame(results)
    df['date'] = [
        starting_date + pd.Timedelta(weeks=wk) for wk in range(timesteps + 1)
    ]

    # Check if a staking rate has been generated and stored in the session state
    if 'staking_rate' in st.session_state:
        df['percentage_staked'] = st.session_state['staking_rate'][:len(df)]

    else:
        st.error(
            "Staking rate not generated. Please generate the staking rate first."
        )
        return pd.DataFrame(
        )  # Return an empty DataFrame to prevent further execution

    # Apply the balance to each entity column
    for entity in system_parameters['entities'][0]:
        df[entity['name']] = df[entity['name']].apply(lambda x: x.balance)

    df['unlocked_supply'] = df[[
        entity['name'] for entity in system_parameters['entities'][0]
    ]].sum(axis=1)
    df['total_supply'] = total_supply  # Assuming total_supply is defined elsewhere
    df['weekly_inflation'] = 0

    df['weekly_inflation_percent'] = 0
    df['annual_inflation_percent'] = 0

    df['total_staked'] = (df['percentage_staked'] /
                          100) * df['unlocked_supply']

    for t in range(len(df)):
        staking_rate = df.loc[t, 'percentage_staked']  # S in the formula
        annual_minting_rate = multiplier * np.sqrt(
            staking_rate)  # M = 0.4 * sqrt(S)
        weekly_minting_rate = annual_minting_rate / 52  # Convert annual minting rate to weekly

        df.loc[t, 'annual_inflation_percent'] = annual_minting_rate
        df.loc[t, 'weekly_inflation_percent'] = weekly_minting_rate

        weekly_inflation_tokens = (weekly_minting_rate /
                                   100) * df.loc[t, 'total_supply']
        df.loc[t, 'weekly_inflation'] = weekly_inflation_tokens

        if t > 0:
            df.loc[t, 'total_supply'] = df.loc[t - 1, 'total_supply'] + df.loc[
                t, 'weekly_inflation']
        else:
            
            df.loc[t, 'total_supply'] += weekly_inflation_tokens

    return df


st.sidebar.header('System Parameter Configuration')
suppl_expander = st.sidebar.expander('Total Supply & TGE Date', expanded=False)
with suppl_expander:
    starting_date = st.date_input("Starting Date",
                                  value=pd.to_datetime("2024-01-01"))
total_supply = suppl_expander.number_input('Total Supply (to be vested)',
                                           value=1_000_000_000,
                                           step=1000)

staking_expander = st.sidebar.expander('Staking Parameters', expanded=False)
base_mu = staking_expander.number_input('Base Mu (Mean of Drift)',
                                        value=0.05,
                                        step=0.01)
sigma = staking_expander.number_input(
    'Sigma (Standard Deviation of Drift)',
    value=0.5,
    step=0.10,
    help=
    "Sigma controls the volatility of the staking percentage's change over time. A higher sigma results in more significant fluctuations."
)
lower_bound = staking_expander.number_input(
    'Lower Bound of Target Percentage Staked', value=20.0, step=1.0)
upper_bound = staking_expander.number_input(
    'Upper Bound of Target Percentage Staked', value=60.0, step=1.0)


inflation_parameter = st.sidebar.expander('Inflation Coefficient',
                                          expanded=False)
with inflation_parameter:
    # Input for inflation coefficient
    multiplier = st.number_input('Multiplier for Annual Minting Rate',
                                 value=0.4,
                                 format="%.2f")

    # Explanation about the inflation coefficient
    st.markdown("""
    ### How the Inflation Coefficient Works
    The inflation coefficient (multiplier) directly influences the annual minting rate of new tokens. It is applied as a multiplier in the formula used to calculate the annual inflation rate, which is based on the square root of the staking rate (`S`):
    
    ```
    Annual Minting Rate = Multiplier * sqrt(Staking Rate)
    ```
    
    - **Higher Multiplier:** Increases the annual minting rate, leading to higher inflation. 
    - **Lower Multiplier:** Decreases the annual minting rate, leading to lower inflation.
    
                
    Adjusting the multiplier allows you to simulate different inflation scenarios and their impact on the token ecosystem.
    """)

    # Validate bounds
if lower_bound >= upper_bound:
    st.sidebar.error(
        'Error: The upper bound must be higher than the lower bound.')

entities_expander = st.sidebar.expander('Allocation Configuration',
                                        expanded=False)
entities_data = []
total_percentage = 0

# Default names and percentages for entities
default_entities = {
    'Early_Contributors': 20.04,
    'Investors': 18.17,
    'Starkware': 10.76,
    'Grants': 12.93,
    'Community_Provisions': 9.00,
    'Community_Rebates': 9.00,
    'Strategic_reserves': 10.00,
    'Treasury': 8.10,
    'Donations': 2.00,
}

# Dynamically adjust the number of entities based on default entities
entities_count = len(default_entities)

for i, (name, default_percentage) in enumerate(default_entities.items()):
    col1, col2 = entities_expander.columns([3, 2])
    with col1:
        # Use default names for entities
        entity_name = st.text_input(f'Name {i+1}', value=name)
    with col2:
        percentage = st.number_input(
            f'Percentage of Total Supply going to: {entity_name}',
            min_value=0.0,
            max_value=100.0,
            value=default_percentage,
            step=0.05,
            key=f'percentage_{i}')

    tge_unlock_percentage = entities_expander.number_input(
        f'TGE Unlock % for {entity_name}',
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.01,
        key=f'tge_{i}')  # TGE unlock percentage inpu
    vesting_months = entities_expander.slider(
        f'Vesting Length (Months) of {entity_name}',
        min_value=0,
        max_value=60,
        value=48,
        key=f'vesting_{i}')
    cliff_weeks = entities_expander.slider(
        f'Cliff Length (Weeks) of {entity_name}',
        min_value=0,
        max_value=260,
        value=0,
        key=f'cliff_{i}',
        help="1 month =  4.34524 weeks")
    frequency = entities_expander.slider(
        f'Frequency of unlocks of  {entity_name}',
        min_value=1,
        max_value=260,
        value=1,
        key=f'frequency_{i}',
        help="1 means everyweek, 2 every second week and so on. ")
    delay = entities_expander.slider(f'Delay (weeks) of {entity_name}',
                                     min_value=0,
                                     max_value=260,
                                     value=0,
                                     key=f'delay_{i}',
                                     help="1 month =  4.34524 weeks")

    entities_data.append({
        'name': entity_name,
        'tge_unlock_percentage': tge_unlock_percentage,
        'percentage_of_total_supply': percentage,
        'vesting_length_months': vesting_months,
        'cliff_length_weeks': cliff_weeks,
        'frequency': frequency,
        'delay': delay,
    })

    total_percentage += percentage

if total_percentage != 100.0:
    st.sidebar.error(
        'Error: The total percentage of all entities must add up to 100%.')

# Update system parameters with entities data
system_parameters = {
    'total_supply': [total_supply],
    'entities': [entities_data]
}


st.markdown("<h1 style='text-align: center;'>Starknet Vesting Simulator</h1>",
            unsafe_allow_html=True)

if st.button('Generate Staking Rate'):
    time_steps = 5 * 52 + 1 
    base_line = np.linspace(lower_bound, upper_bound,
                            num=time_steps)  # Base linear progression
    random_fluctuations = np.random.normal(
        loc=0, scale=sigma, size=time_steps)  # Generate random fluctuations

    # Apply dynamic adjustments to the base line to simulate realistic staking percentage fluctuations
    staking_rate_with_fluctuations = []
    for t in range(time_steps):
        if t == 0:
            staking_rate_with_fluctuations.append(base_line[t])
        else:
            adjusted_mu = base_mu
            if base_line[t - 1] < lower_bound:
                adjusted_mu = base_mu * (
                    1 + (lower_bound - base_line[t - 1]) / lower_bound)
            elif base_line[t - 1] > upper_bound:
                adjusted_mu = -base_mu * (base_line[t - 1] -
                                          upper_bound) / upper_bound

            new_value = base_line[t] + adjusted_mu + random_fluctuations[t]
            new_value = max(0,
                            min(100,
                                new_value)) 
            staking_rate_with_fluctuations.append(new_value)

   
    st.session_state['staking_rate'] = staking_rate_with_fluctuations
    st.success('Staking rate generated!')

if st.button('Run Simulation'):
    if total_percentage != 100.0:
        st.error(
            'Error: The total percentage of all entities must add up to 100%.')
    elif upper_bound <= lower_bound:
        st.error('Error: The upper bound must be higher than the lower bound.')
    else:
       
        df = simulate(system_parameters,
                      total_supply=total_supply,
                      multiplier=multiplier)

        col1, col2 = st.columns(2)

        with col1:
            
            df['inflation_tokens'] = df['weekly_inflation'].cumsum()
            df['unlocked_supply'] = df['unlocked_supply'] + df[
                'inflation_tokens']

            fig_area = px.area(
                df,
                x='date',
                y=[
                    entity['name']
                    for entity in system_parameters['entities'][0]
                ] +
                ['inflation_tokens'
                 ],  
                title='Tokenomics Supply Over Time')
            fig_area.update_layout(
                title={
                    'text': 'Tokenomics Supply Over Time',
                    'x': 0.5,  
                    'xanchor': 'center'
                })

            st.plotly_chart(fig_area, use_container_width=True)

        with col2:

            pie_data = pd.DataFrame(system_parameters['entities'][0])
            fig_pie = px.pie(pie_data,
                             values='percentage_of_total_supply',
                             names='name',
                             title='Percentage Distribution of Total Supply')
            fig_pie.update_layout(
                title={
                    'text': 'Percentage Distribution of Total Supply',
                    'x': 0.5,  
                    'xanchor': 'center'
                })

            st.plotly_chart(fig_pie, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_line1 = px.line(
                df,
                x='date',
                y=['unlocked_supply', 'total_staked'],
                )
            fig_line1.update_layout(
                title={
                    'text': 'STRK Supply Over Time - Unlocks + Inflation',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                width = 600,
                height = 450)
            st.plotly_chart(fig_line1)

        with col4:
            fig_line2 = px.line(df,
                                x='date',
                                y='percentage_staked',
                                title='Staked over time')
            fig_line2.update_layout(title={
                'text': ' % Staked  over time',
                'x': 0.5,
                'xanchor': 'center'
            })
            st.plotly_chart(fig_line2)

        col5, _ = st.columns(2)

        with col5:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['annual_inflation_percent'],
                    mode='lines',
                    name='Annual Inflation Percent',
                    hoverinfo='text+name',
                    text=df.apply(lambda row:
                                  f"Staking Rate: {row['percentage_staked']}%",
                                  axis=1)))

            
            fig.update_layout(
                title={
                    'text': 'Annual Inflation Percent Over Time',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Timestep',
                yaxis_title='Annual Inflation %',
                hovermode='closest'
            )  

            
            st.plotly_chart(fig)

        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_file = convert_df_to_csv(df)
        st.download_button(
            label="Download data as CSV",
            data=csv_file,
            file_name="simulation_data.csv",
            mime="text/csv",
        )
