import numpy as np
import matplotlib.pyplot as plt

class Walker:

    def __init__(self,current_i, current_j, sigma_i, sigma_j, size, context_map):
        self.current_i = current_i
        self.current_j = current_j
        self.sigma_i = sigma_i
        self.sigma_j =  sigma_j
        self.size = size
        self.context_map = context_map

    # Methods 
    def sample_next_step(self, random_state=np.random):
        """ Sample a new position for the walker. """

        # Combine the next-step proposal with the context map to get a next-step
        # probability map
        self.size = self.context_map.shape[0]
        self.next_step_map = self.next_step_proposal()
        self.next_step_probability = self.compute_next_step_probability()

        # Draw a new position from the next-step probability map
        r = random_state.rand()
        cumulative_map = np.cumsum(self.next_step_probability)
        cumulative_map = cumulative_map.reshape(self.next_step_probability.shape)
        self.i_next, self.j_next = np.argwhere(cumulative_map >= r)[0]

        return self.i_next, self.j_next


    def next_step_proposal(self):
        """ Create the 2D proposal map for the next step of the walker. """
        # 2D Gaussian distribution , centered at current position,
        # and with different standard deviations for i and j
        grid_ii, grid_jj = np.mgrid[0:self.size, 0:self.size]
        rad = (
            (((grid_ii - self.current_i) ** 2) / (self.sigma_i ** 2))
            + (((grid_jj - self.current_j) ** 2) / (self.sigma_j ** 2))
        )
        p_next_step = np.exp(-(rad / 2.0)) / (2.0 * np.pi * self.sigma_i * self.sigma_j)
        return p_next_step / p_next_step.sum()


    def compute_next_step_probability(self):
        """ Compute the next step probability map from next step proposal and
        context map. """
        self.next_step_probability = self.next_step_map * self.context_map
        self.next_step_probability /= self.next_step_probability.sum()
        return self.next_step_probability


    def create_context_map(self, map_type='flat'):
        """ Create a fixed context map. """
        if map_type == 'flat':
            self.context_map = np.ones((self.size, self.size))
        elif map_type == 'hills':
            grid_ii, grid_jj = np.mgrid[0:self.size, 0:self.size]
            i_waves = np.sin(grid_ii / 130) + np.sin(grid_ii / 10)
            i_waves /= i_waves.max()
            j_waves = np.sin(grid_jj / 100) + np.sin(grid_jj / 50) + \
                np.sin(grid_jj / 10)
            j_waves /= j_waves.max()
            self.context_map = j_waves + i_waves
        elif map_type == 'labyrinth':
            self.context_map = np.ones((self.size, self.size))
            self.context_map[50:100, 50:60] = 0
            self.context_map[20:89, 80:90] = 0
            self.context_map[90:120, 0:10] = 0
            self.context_map[120:self.size, 30:40] = 0
            self.context_map[180:190, 50:60] = 0

            self.context_map[50:60, 50:200] = 0
            self.context_map[179:189, 80:130] = 0
            self.context_map[110:120, 0:190] = 0
            self.context_map[120:self.size, 30:40] = 0
            self.context_map[180:190, 50:60] = 0
            self.context_map /= self.context_map.sum()
        return self.context_map


    def plot_trajectory(self, trajectory):
        """ Plot a trajectory over a context map. """
        trajectory = np.asarray(trajectory)
        plt.matshow(self.context_map)
        plt.plot(trajectory[:, 1], trajectory[:, 0], color='r')
        plt.show()
