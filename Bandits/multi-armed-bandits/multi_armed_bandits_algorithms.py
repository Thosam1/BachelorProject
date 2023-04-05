import numpy as np


def explore_then_exploit_naive(n_steps, n_arms, explore_fraction, p):
    if explore_fraction > 1.0:
        raise Exception("Explore fraction shouldn't be more than 1 !")

    explore_per_arm = int(np.floor(n_steps * explore_fraction / n_arms))
    n_steps_explore = explore_per_arm * n_arms
    n_steps_exploit = n_steps - n_steps_explore

    # Arm chosen and reward corresponding
    actions = np.zeros(n_steps)
    rewards = np.zeros(n_steps)
    # List of tuples (step, arm)
    changes = []

    results_all_arms_end_of_explore = np.zeros((n_arms, explore_per_arm))

    # Explore
    for arm_index in range(n_arms):
        results_for_arm = np.random.choice([0, 1], size=(1, explore_per_arm),
                                           p=[1 - float(p[arm_index]), float(p[arm_index])])
        results_all_arms_end_of_explore[arm_index, :] = results_for_arm
        start = explore_per_arm * arm_index
        end = explore_per_arm * (1 + arm_index)
        rewards[start:end] = results_for_arm
        actions[start:end] = np.ones(explore_per_arm) * arm_index
        changes.append((start, arm_index))

    end_of_explore_results = np.sum(results_all_arms_end_of_explore, axis=1)

    # Exploit
    best_arm = np.argmax(end_of_explore_results)
    result_for_exploited_arm = np.random.choice([0, 1], size=(1, n_steps_exploit),
                                                p=[1 - float(p[best_arm]), float(p[best_arm])])
    rewards[n_steps_explore:] = result_for_exploited_arm
    actions[n_steps_explore:] = np.ones(n_steps_exploit) * best_arm
    changes.append((n_steps_explore, best_arm))

    return explore_per_arm, results_all_arms_end_of_explore, end_of_explore_results, actions, rewards, changes


def total_explore_arm_pulls_for_elimination_algo(n_arms, n_rounds, round_steps_per_arm):
    total = 0
    for round_i in range(n_rounds):
        total = total + n_arms - round_i
    return total * round_steps_per_arm


def explore_then_exploit_elimination(n_steps, n_arms, n_rounds, round_steps_per_arm, p):
    n_steps_explore = total_explore_arm_pulls_for_elimination_algo(n_arms, n_rounds, round_steps_per_arm)
    n_steps_exploit = n_steps - n_steps_explore

    if n_rounds >= n_arms:
        raise Exception("Number of rounds should be strictly lesser than the number of arms")

    if n_steps < n_steps_explore:
        raise Exception(
            "Increase number of trials, or decrease the number of rounds or the number of steps per arm for each round")

    end_of_explore_total = np.zeros(n_arms)
    poor_performing_arms = []

    # Contains (arm, step, total nb of pulls) for elimination history
    eliminations = []

    # Contains (arm, step)
    changes = []

    actions = np.zeros(n_steps)
    rewards = np.zeros(n_steps)

    results_all_arms_end_of_explore = np.zeros((n_arms, 0))
    results_all_arms = np.zeros((n_arms, 0))

    # Stores the total nb of arms visited
    counter_arm = 0

    # Explore
    for i in range(n_rounds):

        results_all_arms = np.zeros((n_arms, round_steps_per_arm))

        for arm_index in range(n_arms):
            # doesn't exist in the poor performing list
            if poor_performing_arms.count(arm_index) == 0:
                results_all_arms[arm_index, :] = np.random.choice([0, 1], size=(1, round_steps_per_arm),
                                                                  p=[1 - float(p[arm_index]), float(p[arm_index])])
                start = round_steps_per_arm * counter_arm
                end = round_steps_per_arm * (1 + counter_arm)
                actions[start:end] = np.ones(round_steps_per_arm) * arm_index
                rewards[start:end] = results_all_arms[arm_index, :]
                changes.append((start, arm_index))
                counter_arm = counter_arm + 1

        end_of_round_results = np.sum(results_all_arms, axis=1)
        end_of_explore_total = end_of_explore_total + end_of_round_results

        index_with_least_wins = np.argsort(end_of_explore_total, axis=0)[i]

        poor_performing_arms.append(index_with_least_wins)
        eliminations.append((index_with_least_wins, round_steps_per_arm * counter_arm,
                             results_all_arms_end_of_explore.shape[1] + round_steps_per_arm))

        results_all_arms_end_of_explore = np.concatenate((results_all_arms_end_of_explore, results_all_arms), axis=1)

        if i == n_rounds - 1:
            results_all_arms = np.concatenate((results_all_arms, results_all_arms_end_of_explore), axis=1)

    # Exploit
    best_arm = np.argmax(end_of_explore_total)
    result_for_exploited_arm = np.random.choice([0, 1], size=(1, n_steps_exploit),
                                                p=[1 - float(p[best_arm]), float(p[best_arm])])
    rewards[n_steps_explore:] = result_for_exploited_arm
    actions[n_steps_explore:] = np.ones(n_steps_exploit) * best_arm
    changes.append((n_steps_explore, best_arm))

    # reshape
    return results_all_arms_end_of_explore, end_of_explore_total, results_all_arms, eliminations, actions, rewards, changes


def epsilon_greedy(n_trials, n_arms, epsilon, p):
    Q = np.zeros(n_arms)  # Action-value estimates
    N = np.zeros(n_arms)  # Number of times each action is taken

    actions = np.zeros(n_trials)  # Arm picked
    rewards = np.zeros(n_trials)  # Result of action

    for t in range(n_trials):

        # Pick an action
        if np.random.rand() < epsilon:
            action = np.random.choice(n_arms)
        else:
            action = np.argmax(Q)

        reward = np.random.binomial(1, p[action])

        # Saving
        actions[t] = action
        rewards[t] = reward

        N[action] += 1
        Q[action] += (reward - Q[action]) * (1 / N[action])

    return actions, rewards, N, Q


def elimination_confidence_bound_algorithm(n_arms, n_steps, c, p):
    # Initialize variables
    wins = np.zeros(n_arms)
    losses = np.zeros(n_arms)
    upper_bounds = np.ones(n_arms) * float('inf')
    lower_bounds = np.ones(n_arms) * float('-inf')
    p_hat = np.zeros(n_arms)
    selected_arms = np.zeros(n_steps)
    rewards = np.zeros(n_steps)
    eliminated_arms = []

    # contains (arm, round, step) for elimination history
    eliminations = []

    counter = 0

    for step in range(n_steps):
        # Calculate upper confidence bounds for each arm
        for i in range(n_arms):
            if eliminated_arms.count(i) == 0:
                if wins[i] + losses[i] > 0:
                    p_hat[i] = wins[i] / (wins[i] + losses[i])
                    upper_bounds[i] = p_hat[i] + c / np.sqrt(wins[i] + losses[i])
                    lower_bounds[i] = p_hat[i] - c / np.sqrt(wins[i] + losses[i])

        for selected_arm in range(n_arms):

            if eliminated_arms.count(selected_arm) == 0:

                selected_arms[counter] = selected_arm

                # Simulate play for each arm and update wins and losses
                if np.random.rand() < p[selected_arm]:  # simulate win with probability p_i
                    wins[selected_arm] += 1
                    rewards[counter] = 1
                else:
                    losses[selected_arm] += 1
                    rewards[counter] = 0

                counter = counter + 1

            if (counter == n_steps):
                return selected_arms, rewards, wins, losses, eliminations

        highest_p_hat = np.argmax(p_hat)

        # Check if there is only one arm left
        if np.sum(upper_bounds != float('-inf')) >= 1:
            # Check to eliminate arm
            for i in range(n_arms):
                if eliminated_arms.count(i) == 0:
                    if upper_bounds[i] < lower_bounds[highest_p_hat]:
                        eliminated_arms.append(i)
                        eliminations.append((i, step, counter))

    # Return selected arms at each step
    return selected_arms, rewards, wins, losses, eliminations


def ucb_algorithm(n_arms, n_steps, c, p):
    # Initialize variables
    wins = np.zeros(n_arms)
    losses = np.zeros(n_arms)
    upper_bounds = np.ones(n_arms) * float('inf')
    lower_bounds = np.ones(n_arms) * float('-inf')
    selected_arms = np.zeros(n_steps)
    rewards = np.zeros(n_steps)

    counter = 0
    for step in range(n_steps):
        # Calculate upper confidence bounds for each arm
        for i in range(n_arms):
            if wins[i] + losses[i] > 0:
                p_i = wins[i] / (wins[i] + losses[i])
                upper_bounds[i] = p_i + c / np.sqrt(wins[i] + losses[i])
                lower_bounds[i] = p_i - c / np.sqrt(wins[i] + losses[i])

        # Select arm with highest upper confidence bound
        selected_arm = np.argmax(upper_bounds)
        selected_arms[counter] = selected_arm

        # Simulate play and update wins and losses
        if np.random.rand() < p[selected_arm]:  # simulate win with probability p_i
            wins[selected_arm] += 1
            rewards[counter] = 1
        else:
            losses[selected_arm] += 1
            rewards[counter] = 0

        counter += 1

    # Return selected arms at each step
    return selected_arms, rewards, wins, losses
