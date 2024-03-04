# logic.py
from datetime import timedelta


def generate_dynamic_functions(entity_names):
    policy_functions = {}
    state_update_functions = {}

    for name in entity_names:
        # Policy function
        def policy_function(params,
                            substep,
                            state_history,
                            previous_state,
                            name=name):
            entity = previous_state[name]
            entity.simulate_week_passage()
            return {f'update_{name.replace(" ", "_").lower()}': entity}

        policy_functions[
            f'p_update_{name.replace(" ", "_").lower()}'] = policy_function

        # State update function
        def state_update_function(params,
                                  substep,
                                  state_history,
                                  previous_state,
                                  policy_input,
                                  name=name):
            return (name,
                    policy_input[f'update_{name.replace(" ", "_").lower()}'])

        state_update_functions[name] = state_update_function

    return policy_functions, state_update_functions


def update_tge_date(params, substep, state_history, previous_state,
                    policy_input):
    new_date = previous_state['tge_date'] + timedelta(weeks=1)
    return ('tge_date', new_date)
