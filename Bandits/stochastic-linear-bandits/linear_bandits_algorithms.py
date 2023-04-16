import numpy as np
import matplotlib.pyplot as plt


class LinUCB:
    def __init__(self, n_arms, n_features, item_features, n_rounds, lambda_param):
        self.n_arms = n_arms
        self.n_features = n_features
        self.item_features = item_features
        self.n_rounds = n_rounds
        self.lambda_param = lambda_param

        # Initializing variables
        self.V_t = lambda_param * np.eye(n_features)
        self.sum_A_s_X_s = np.zeros(n_features)
        self.theta_hat = np.zeros((n_features, n_rounds + 1))
        self.theta_hat[:, 0] = np.random.uniform(low=-1, high=1, size=(n_features, 1)).reshape((n_features,))

        self.beta_param_t = 1.0
        self.actions = np.zeros(n_rounds + 1, dtype=int)
        self.rewards = np.zeros(n_rounds + 1)

    # Choose the best action based on the last theta_hat
    def choose_action(self, curr_round):
        max_value = -np.Inf
        max_index = -1
        for i in range(self.n_arms):
            # Compute the UCB for each arm using the given formula
            estimated_value = self.theta_hat[:, curr_round - 1].T @ self.item_features[:, i]
            penalty_value = np.sqrt(self.beta_param_t) * np.sqrt(
                self.item_features[:, i].T @ (np.linalg.inv(self.V_t) @ self.item_features[:, i]))
            value = estimated_value + penalty_value
            if value >= max_value:
                max_value = value
                max_index = i

        # Storing it
        self.actions[curr_round] = max_index
        return max_index

    # Updating variables after receiving the reward from an action
    def update(self, curr_round, reward):
        self.rewards[curr_round] = reward

        # Update the feature matrix V_t by adding the outer product of the chosen action's feature vector with itself.
        self.V_t += np.outer(self.item_features[:, self.actions[curr_round]],
                             self.item_features[:, self.actions[curr_round]])

        # Compute the inverse of the updated feature matrix V_t.
        V_t_inv = np.linalg.inv(self.V_t)

        # Update the label vector sum_A_s_X_s by adding the outer product of the chosen action's feature vector with the observed reward.
        self.sum_A_s_X_s += self.item_features[:, self.actions[curr_round]] * self.rewards[curr_round]

        # Compute the new estimate of the true_theta vector using the updated feature matrix and label vector.
        # This estimate represents the center of the ellipsoid in the feature space.
        self.theta_hat[:, curr_round] = V_t_inv @ self.sum_A_s_X_s


class EnvironmentLinUCB:
    def __init__(self, n_arms, n_features, item_features, n_rounds, true_theta, noise):
        self.n_arms = n_arms
        self.n_features = n_features
        self.item_features = item_features
        self.n_rounds = n_rounds
        self.true_theta = true_theta
        self.noise = noise

        # Initializing variables
        self.optimal_reward = np.max(item_features.T @ true_theta)
        self.rewards = np.zeros(n_rounds + 1)  # rewards with noise
        self.regrets = np.zeros(n_rounds + 1)

    # Observe the reward of the chosen action and add noise.
    def observe_reward(self, curr_round, action):
        expected_reward = self.true_theta.T @ self.item_features[:, action]
        self.rewards[curr_round] = expected_reward + np.random.normal(
            scale=self.noise)  # this might be the reason why we get negative regret
        return expected_reward, self.rewards[curr_round]

    def calculate_regret(self, curr_round, expected_reward):
        # Compute the regret for this round.
        regret = self.optimal_reward - expected_reward
        self.regrets[curr_round] = self.regrets[curr_round - 1] + regret

    def get_regrets(self):
        return self.regrets
