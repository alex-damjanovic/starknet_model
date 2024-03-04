# initial_variables.py
from .types import STRK
from .classes import Entity
from datetime import datetime, timedelta


def setup_initial_state(system_params: dict) -> dict:
    total_supply = system_params['total_supply'][0]
    entities_params = system_params['entities']
    initial_state = {}

    for entity_params in entities_params[0]:
        entity = Entity(
            name=entity_params['name'],
            tge_unlock_percentage=entity_params['tge_unlock_percentage'])
        allocation = STRK(
            (entity_params['percentage_of_total_supply'] / 100) * total_supply)
        entity.update_delay(entity_params['delay'])
        entity.update_frequency(entity_params['frequency'])
        entity.update_total_allocation(allocation)
        entity.update_vesting_length_months(
            entity_params['vesting_length_months'])
        entity.update_cliff_length_weeks(entity_params['cliff_length_weeks'])
        entity.start_vesting()
        initial_state[entity.name] = entity

    initial_state['unlocked_supply'] = 0

    return initial_state
