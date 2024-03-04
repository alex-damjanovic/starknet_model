system_parameters = {
    'total_supply': [1_000_000_000],  # Total supply of tokens in the system
    'entities': [[  # default structure
        {
            'name': 'Treasury',
            'percentage_of_total_supply': 50,
            'vesting_length_months': 48,
            'cliff_length_weeks': 52,
            'frequency': 2,
            'delay': 5
        }, {
            'name': 'Community',
            'percentage_of_total_supply': 50,
            'vesting_length_months': 36,
            'cliff_length_weeks': 0,
            'frequency': 1,
            'delay': 10
        }
    ]]
}
