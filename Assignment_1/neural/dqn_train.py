import gym
from dqn_agent import DQNAgent
#from dqn_agent_target_network import DQNAgentTargetNetwork
#from dqn_agent_copy import DQNAgentCopy
import numpy as np
from helper_functions import get_session_id, plot_chart, plot_chart_multi, plot_chart_boxplot
from helper_functions import make_chart_title, create_filename, save_result, readable_setting
from settings import defaults
from time import time
from datetime import datetime


class DQNTrain():
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.agent = None
        self.initial_epsilon = None
        self.epsilon = None
        self.epsilon_decay_model = None
        self.epsilon_straight_decay_floor = 0
        self.epsilon_straight_decay_step = 0
        self.epsilon_exponential_decay_floor = 0
        self.epsilon_decay_reward_threshold_step = 0
        self.epsilon_decay_reward_threshold = 0
        self.epsilon_decay_pulse_step_frequency = 0
        self.epsilon_decay_pulse_magnitude = 0

    def adjust_epsilon(self, step=None, episode=None, reward=None):
        if self.epsilon_decay_model in ["STRAIGHT_STEP", "STRAIGHT_EPISODE"]:
            self.epsilon = self.epsilon - self.epsilon_straight_decay_step if self.epsilon - self.epsilon_straight_decay_step > self.epsilon_straight_decay_floor else self.epsilon_straight_decay_floor
        elif self.epsilon_decay_model == "STEP":
            self.epsilon = self.initial_epsilon / step if self.initial_epsilon / step > self.epsilon_exponential_decay_floor else self.epsilon_exponential_decay_floor
        elif self.epsilon_decay_model == "EPISODE":
            self.epsilon = self.initial_epsilon / (episode + 1) if self.initial_epsilon / (episode + 1) > self.epsilon_exponential_decay_floor else self.epsilon_exponential_decay_floor
        elif self.epsilon_decay_model[:5] == "FIXED":
            pass
        elif self.epsilon_decay_model == "REWARD":
            if reward > self.epsilon_decay_reward_threshold and self.epsilon > self.epsilon_straight_decay_floor:
                self.epsilon = self.epsilon * self.epsilon_decay_reward_decay_percent if self.epsilon * self.epsilon_decay_reward_decay_percent > self.epsilon_straight_decay_floor else self.epsilon_straight_decay_floor
                self.epsilon_decay_reward_threshold = self.epsilon_decay_reward_threshold + self.epsilon_decay_reward_threshold_step
        elif self.epsilon_decay_model == "PULSE":
            if self.epsilon > self.epsilon_straight_decay_floor:
                self.epsilon = self.epsilon - self.epsilon_straight_decay_step if self.epsilon - self.epsilon_straight_decay_step > self.epsilon_straight_decay_floor else self.epsilon_straight_decay_floor
            elif step % self.epsilon_decay_pulse_step_frequency == 0:
                self.epsilon = self.epsilon_decay_pulse_magnitude
        else:
            raise Exception(f"UNKNOWN EPSILON DECAY MODEL {self.epsilon_decay_model}")
        self.agent.epsilon = self.epsilon
        return self.epsilon

    def load_agent(self, **kwargs):
        #Todo Change Target Network and Copy to right call signature
        agent_type = kwargs['agent_type']
        if agent_type == "DEFAULT":
            return DQNAgent(**kwargs)
        elif agent_type == "TARGET":
            raise Exception(f"UNKNOWN AGENT TYPE OF {agent_type} NOT IMPLEMENTED")
            #return DQNAgentTargetNetwork(episodes, epsilon=epsilon, gamma=gamma, learning_rate=learning_rate,
            #                                   layer_size_1=layer_size_1,
            #                                  layer_size_2=layer_size_2)
        elif agent_type == "COPY":
            raise Exception(f"GENT TYPE OF {agent_type} NOT IMPLEMENTED")
            #return DQNAgentCopy(episodes, epsilon=epsilon, gamma=gamma, learning_rate=learning_rate,
            #                          layer_size_1=layer_size_1,
            #                         layer_size_2=layer_size_2)
        else:
            raise Exception(f"UNKNOWN AGENT TYPE OF {agent_type}")

    def training_header(self, **kwargs):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"STARTING TRAINING  AGENT:{kwargs['agent_type']}  Episodes:{kwargs['episodes']} Training Iterations:{kwargs['training_iterations']}  Gamma:{kwargs['gamma']} ")
        if kwargs['sweep_value_key']:
            print(f"Sweeping:{kwargs['sweep_value_key']} in {kwargs['sweep_value_list']}")
        print(f"Epsilon Decay Model:{kwargs['epsilon_decay_model']} Starting Epsilon:{kwargs['epsilon']} Floor:{kwargs['epsilon_straight_decay_floor']} Step:{kwargs['epsilon_straight_decay_step']}")
        print(f"Learning Rate:{kwargs['learning_rate']} Layers:({kwargs['layer_size_1'], kwargs['layer_size_2']}) ")
        print(f"Memory Training Batch Size:{kwargs['memory_batch_size']} Memory Max:{kwargs['memory_max']}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def nan_mean(self, mean_array):
        nd_array = np.array(mean_array)
        transpose_nd_array = nd_array.transpose()
        return_array = np.zeros(len(transpose_nd_array))
        for index, column in enumerate(transpose_nd_array):
            if np.isnan(column).all():
                return_array[index] = np.nan
            else:
                return_array[index] = np.nanmean(column)
        return return_array

    def train(self, **kwargs):
        # Unpack Settings
        settings = {**defaults, **kwargs}
        episodes = settings['episodes']
        filter_length = settings['filter_length']
        show_render = settings['show_render']
        show_per_episode_output = settings['show_per_episode_output']

        # Epsilon decay model
        self.initial_epsilon = settings['epsilon']
        self.epsilon = settings['epsilon']
        self.epsilon_decay_model = settings['epsilon_decay_model']
        self.epsilon_straight_decay_floor = settings['epsilon_straight_decay_floor']
        self.epsilon_straight_decay_step = settings['epsilon_straight_decay_step']
        self.epsilon_exponential_decay_floor = settings['epsilon_exponential_decay_floor']
        self.epsilon_decay_reward_threshold_step = settings['epsilon_decay_reward_threshold_step']
        self.epsilon_decay_reward_decay_percent = settings['epsilon_decay_reward_decay_percent']
        self.epsilon_decay_reward_threshold = settings['epsilon_decay_reward_initial_threshold']
        self.epsilon_decay_pulse_step_frequency = settings['epsilon_decay_pulse_step_frequency']
        self.epsilon_decay_pulse_magnitude = settings['epsilon_decay_pulse_magnitude']

        if self.epsilon_decay_model[:5] == "FIXED":
            # A bit of a hack, but allows to pass in this for hypersweeps
            e = float(self.epsilon_decay_model[6:])
            if not e > 0:
                raise Exception(f"FIXED EPSILON NOT A NUMBER epsilon_decay_model:{self.epsilon_decay_model} e:{e}")
            self.initial_epsilon = e
            self.epsilon = e
            # Need to set epsilon because FIXED agent may be used in sweep
            settings['epsilon'] = self.epsilon
        step_decay = False
        episode_decay = False
        if self.epsilon_decay_model in ["STRAIGHT_STEP", "STEP", "PULSE"]:
            step_decay = True
        elif self.epsilon_decay_model in ["EPISODE", "STRAIGHT_EPISODE", "REWARD"]:
            episode_decay = True
        elif self.epsilon_decay_model[:5] in ["FIXED"]:
            pass
        else:
            raise Exception(f"Unsupported Epsilon Decay Model of {self.epsilon_decay_model}")

        self.agent = self.load_agent(**settings)

        # Settings for terminating training on reaching threshold
        solve_reward_avg_threshold = kwargs['solve_reward_avg_threshold']
        end_training_on_reward_avg = kwargs['end_training_on_reward_avg']

        session_id = get_session_id()
        start_time = time()
        rewards = np.zeros((episodes))
        epsilon_values = np.zeros((episodes))
        total_steps = 1
        time_to_solve = episodes # BY setting this, it shows time to solve as high. Could double it for punishment?
        render_flag = False

        solved_threshold = False
        for i in range( episodes ):
            if show_render:
                render_flag = (i + 1) % 25 == 0
            score = 0
            is_done = False
            state = self.env.reset()
            step = 0
            while not is_done:
                action = self.agent.choose_next_action(state)
                next_state, reward, is_done, info = self.env.step(action)
                if render_flag:
                    self.env.render()
                score += reward
                self.agent.update(state, action, reward, next_state, is_done)
                self.agent.train()
                state = next_state
                step += 1
                total_steps += 1
                if step_decay:
                    self.adjust_epsilon(step=total_steps, episode=i)

            if episode_decay:
                self.adjust_epsilon(step=total_steps, episode=i, reward=score)
            rewards[i] = score
            epsilon_values[i] = self.agent.epsilon

            avg_score = np.mean(rewards[max(0, i-100):i+1])
            if avg_score > solve_reward_avg_threshold and not solved_threshold:
                time_to_solve = i
                solved_threshold = True
                if end_training_on_reward_avg:
                    # Save last full value through to the end, can be cut off in charting
                    rewards[rewards==0] = solve_reward_avg_threshold
                    epsilon_values[epsilon_values==0] = epsilon_values[[i]]
                    break
            if show_per_episode_output:
                print(f"episode:{i:3} score:{score:5.0f} avg100:{avg_score:5.0f} e:{self.epsilon:7.6f} steps:{step:4} total_steps:{total_steps:7}")
            else:
                print(f"\repisode:{i:3} score:{score:5.0f} avg100:{avg_score:5.0f} e:{self.epsilon:7.6f} steps:{step:4} total_steps:{total_steps:7}", end='', flush=True)

        end_time = time()
        total_time = end_time - start_time
        time_per_episode = total_time / episodes
        time_per_step = 1000 * (total_time / total_steps)
        print("")
        print(f"ENDED TRAINING Total Time:{total_time/60:4.1f}  Final 100 Avg:{avg_score:3.0f} Time per Step:{time_per_step:6.2f} "
              f"per Episode:{time_per_episode:4.3f}  Thresh {solve_reward_avg_threshold}: {solved_threshold} "
              f"(Date Time:{datetime.now().strftime('%m/%d %H:%M')})")
        print("")

        title = make_chart_title(title="", **settings)
        filename = create_filename(title=title, **kwargs)
        # Chart Results
        if settings['chart_individual_training']:
            plot_chart(rewards, title, y_label="Reward", filter_length=filter_length, solved_line=True, filename=filename)
            if settings['chart_epsilon']:
                plot_chart(epsilon_values, title, y_label="Epsilon", filter=False, filename=filename)

        return [rewards, epsilon_values, time_per_step, time_to_solve]

    def train_iterations(self, **kwargs):
        #Unpack needed settings
        settings = {**defaults, **kwargs}
        training_iterations = settings['training_iterations']
        filter_length = settings['filter_length']

        all_rewards = []
        all_epsilon_values = []
        time_per_step_values = []
        time_to_solve_values = []
        settings = {**settings, **{'chart_individual_training': False}}
        for i in range(training_iterations):
            print(f"ITERATION {i}")
            reward, epsilon_value, time_per_step, time_to_solve = self.train(**settings)
            all_rewards.append(reward)
            all_epsilon_values.append(epsilon_value)
            time_per_step_values.append(time_per_step)
            time_to_solve_values.append(time_to_solve)

        # need to use nanmean - nan values are put in after solve
        rewards = self.nan_mean(all_rewards)
        epsilon_values = self.nan_mean(all_epsilon_values)

        # Prep title - used for charting and saving results
        settings['values_to_show'] = []
        title = ""
        if settings['sweep_value_key']:
            settings['values_to_show'].append('sweep_value_key')
            title += str(settings[settings['sweep_value_key']]) + ' '
        if training_iterations > 1:
            settings['values_to_show'].append('training_iterations')

        title = make_chart_title(title=title, **settings)
        # Chart Results
        if settings['chart_iterations_training']:
            filename = create_filename(title='Reward ' + title, **settings)
            plot_chart(rewards, title, y_label="Reward",  filter_length=filter_length, solved_line=True, filename=filename)
            #plot_chart_multi_individual(all_rewards, title, y_label=None, x_label="Episode", filter=False,
            #                           filter_length=100,
            #                            solved_line=False, filename=filename)
            if settings['chart_epsilon']:
                filename = create_filename(title='Epsilon ' + title, **settings)
                plot_chart(epsilon_values, title, y_label="Epsilon",  filter=False, filename=filename)
            filename = create_filename(title='Step Time ' + title, **settings)
            plot_chart_boxplot('Time per Step ' + title, [time_per_step_values], ["-"], 'Time (000s)', filename=filename)
            filename = create_filename(title='Episodes To Solve ' + title, **settings)
            plot_chart_boxplot('Episodes to Solve ' + title, [time_to_solve_values], ["-"], 'Episodes', filename=filename)

        # Save Results
        if settings['save_iteration_results']:
            sweep_value_key = settings['sweep_value_key']
            if sweep_value_key:
                if sweep_value_key == "epsilon_decay_model":
                    x_label = readable_setting("epsilon_decay_model", **kwargs)
                else:
                    x_label = settings[settings['sweep_value_key']]
            else:
                x_label = '*'
            group_order_id = settings['group_order_id']
            save_result(filename, {'rewards': rewards, 'epsilon_values': epsilon_values, 'time_per_step_values': time_per_step_values,
                                   'time_to_solve_values': time_to_solve_values, 'x_label': x_label,
                                    'group_id': settings['group_id'], 'group_order_id': group_order_id})

        return [rewards, epsilon_values, time_per_step_values, time_to_solve_values, x_label]

    def train_sweep(self, **kwargs):
        # Unpack settings
        settings = {**defaults, **kwargs}
        training_iterations = settings['training_iterations']
        sweep_value_key = settings['sweep_value_key']
        sweep_value_list = settings['sweep_value_list']

        #Create Group ID and Label
        if settings['group_id'] == 0:
            group_id = get_session_id()
            settings['group_id'] = group_id
        if settings['group_label'] == 'DEFAULT':
            settings['group_label'] = sweep_value_key

        rewards = []
        epsilon_values = []
        time_per_step_values = []
        time_to_solve_values = []
        x_labels = []

        self.training_header(**settings)

        start_time = time()
        print(f"Sweep of {sweep_value_key} in {sweep_value_list}")
        for group_order_id, sweep_value in enumerate(sweep_value_list):
            print(f"STARTING {sweep_value} in {sweep_value_list}")
            # Update Settings
            update_settings = {sweep_value_key: sweep_value, 'group_order_id': group_order_id}
            settings = {**settings, **update_settings}

            result, result_epsilon, time_per_step_value, time_to_solve_value, x_label = self.train_iterations(**settings)
            rewards.append([f'{sweep_value_key} {sweep_value}', result])
            epsilon_values.append([f'{sweep_value_key} {sweep_value}', result_epsilon])
            x_labels.append(x_label)
            time_per_step_values.append(time_per_step_value)
            time_to_solve_values.append(time_to_solve_value)
            end_time = time()
            print(f"Ended {sweep_value} Elapsed Time:{end_time - start_time:5.0f}")

        if training_iterations > 1:
            settings['values_to_show'] = ['training_iterations']
        title = make_chart_title(title=f"{settings['sweep_value_key'].title().replace('_', ' ')} ", **settings)

        filename = create_filename(title='Reward ' + title, **settings)
        plot_chart_multi(rewards, title, y_label="Reward", filter_length=settings['filter_length'], solved_line=True,
                         filter=True, filename=filename)


        if settings['chart_epsilon']:
            filename = create_filename(title='Epsilon ' + title, **settings)
            plot_chart_multi(epsilon_values, title, y_label="Epsilon", filter=False, solved_line=False, filename=filename)
        filename = create_filename('Time per Step ' + title, **settings)
        plot_chart_boxplot('Time per Step ' + title, time_per_step_values, x_labels, 'Time (000s)', filename=filename)
        filename = create_filename('Episodes to Solve ' + title, **settings)
        plot_chart_boxplot('Episodes to Solve ' + title, time_to_solve_values, x_labels, 'Episodes', filename=filename)

        end_time = time()

        # Save Results
        #if kwargs['save_sweep_results']:
         #   save_result(filename, {'rewards':rewards, 'epsilon_values':epsilon_values, 'time_per_step_values':time_per_step_values,
         #                          'time_to_solve_values':time_to_solve_values})
        print(f"FINISHED SWEEP Elapsed Time:{end_time - start_time:5.0f}")


if __name__ == '__main__':
    dqn_trainer = DQNTrain()
    dqn_trainer.train_iterations(training_iterations=2, episodes=7, filter_length=3, save_iteration_results=True,
                                 result_file_x_label='Quit on Threshold',
                                 end_training_on_reward_avg=False, solve_reward_avg_threshold=-150)
    #dqn_trainer.train_iterations(training_iterations=5, episodes=5, filter_length=5)
    #dqn_trainer.train_sweep(filter_length=5, episodes=8, sweep_value_key='layer_size_1', sweep_value_list=[32, 64], training_iterations=5, show_per_episode_output=True)
    #figure_sweep(filter_length=10, episodes=15, sweep_value_key='learning_rate', sweep_value_list=[0.01, 0.001], training_iterations=3)
    #dqn_trainer = DQNTrain()
    #dqn_trainer.train_iterations(training_iterations=3, episodes=25)
    #dqn_trainer.train(episodes=25)
    #figure_learning_rates(episodes=2000, filter_length=100)
    #figure_epsilon_models(episodes=2000, filter_length=100)
    #figure_epsilon_straight_decay_floors(episodes=2000, filter_length=100)
    #figure_episode_epsilon_step_decays(episodes=2000, filter_length=100)
    #figure_step_epsilon_step_decays(episodes=2000, filter_length=100)
    #figure_epsilon_models(episodes=500)
    #dqn_trainer = DQNTrain()
    #dqn_trainer.train(50, show_render=False, epsilon_decay_model="EPISODE")


