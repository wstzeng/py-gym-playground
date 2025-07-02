import matplotlib.pyplot as plt
from collections import deque
import logging
from typing import Union, List
import os

from .plot_config import set_plot_style

class TrainingMonitor:
    """
    Handles live plotting, CLI logging, and optionally saving logs to file
    during training of RL agents.

    Modes:
      - 'cli': print logs to console (default)
      - 'live': show live updating plot
      - 'file': save logs to `log/` directory

    Modes can be combined by passing a list, e.g., ['cli', 'live', 'file'].
    """
    def __init__(
        self,
        env_name: str,
        total_iterations: int,
        window_size: int = 100,
        mode: Union[str, List[str]] = 'cli',
        log_dir: str = 'log',
    ):
        self.modes = [mode] if isinstance(mode, str) else mode
        self.total_iterations = total_iterations
        self.window_size = window_size
        self.episode_counts = []
        self.avg_rewards_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.env_name = env_name

        # Setup logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
            )

        # Prepare log file if 'file' mode
        self.log_dir = log_dir
        self.log_file_path = None
        if 'file' in self.modes:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file_path = os.path.join(self.log_dir, f'{env_name}_training.log')
            self._log_file = open(self.log_file_path, 'w')
            header = "Iteration,AverageReward,Loss\n"
            self._log_file.write(header)
            self._log_file.flush()

        # Prepare live plot if 'live' mode
        if 'live' in self.modes:
            styles = set_plot_style()
            plt.ion()
            self.fig, self.ax1 = plt.subplots(1, 1)
            self.fig.suptitle(f'RL Training on {env_name}', **styles['suptitle'])

            self.ax2 = self.ax1.twinx()

            colors = styles.get('colors', {})
            reward_color = colors.get('reward_line', 'tab:blue')
            loss_color = colors.get('loss_line', 'tab:orange')

            self.line1, = self.ax1.plot(
                [], [], label=f'Average Reward (last {self.window_size})',
                color=reward_color
            )
            self.line2, = self.ax2.plot(
                [], [], label=f'Policy Loss (last {self.window_size})',
                color=loss_color, linestyle='--')

            self.ax1.set_xlabel('Training Iteration')
            self.ax1.set_ylabel('Average Reward', color=reward_color)
            self.ax2.set_ylabel('Policy Loss', color=loss_color)

            self.ax1.tick_params(axis='y', labelcolor=reward_color)
            self.ax2.tick_params(axis='y', labelcolor=loss_color)

            lines = [self.line1, self.line2]
            labels = [l.get_label() for l in lines]
            self.ax1.legend(
                lines, labels,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.02),
                borderaxespad=0.,
                frameon=False
            )

            self.fig.tight_layout(rect=[0, 0, 1, 0.98])

    def update(self, iteration: int, avg_reward: float, loss: float):
        self.episode_counts.append(iteration)
        self.avg_rewards_history.append(avg_reward)
        self.loss_history.append(loss)

        log_msg = (
            f"[{iteration:3d}/{self.total_iterations}] "
            + f"Avg Reward: {avg_reward:8.2f}, Loss: {loss:12.4f}"
        )

        if 'cli' in self.modes:
            self.logger.info(log_msg)

        if 'file' in self.modes and self._log_file:
            self._log_file.write(f"{iteration},{avg_reward},{loss}\n")
            self._log_file.flush()

        if 'live' in self.modes:
            self.line1.set_xdata(self.episode_counts[-len(self.avg_rewards_history):])
            self.line1.set_ydata(list(self.avg_rewards_history))

            self.line2.set_xdata(self.episode_counts[-len(self.loss_history):])
            self.line2.set_ydata(list(self.loss_history))

            self.ax1.set_xlim(
                max(0, iteration - len(self.avg_rewards_history)),
                iteration + 1
            )
            self.ax2.set_xlim(
                max(0, iteration - len(self.loss_history)),
                iteration + 1
            )

            if self.avg_rewards_history:
                min_r, max_r = min(self.avg_rewards_history), max(self.avg_rewards_history)
                self.ax1.set_ylim(min_r - 0.1 * abs(min_r),
                                  max_r + 0.1 * abs(max_r))

            if self.loss_history:
                min_l, max_l = min(self.loss_history), max(self.loss_history)
                self.ax2.set_ylim(min_l - 0.1 * abs(min_l),
                                  max_l + 0.1 * abs(max_l))

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if 'live' in self.modes:
            plt.ioff()
            plt.close()

        if 'file' in self.modes and self._log_file:
            self._log_file.close()
            self._log_file = None
            self.logger.info(f"Training log saved to: {self.log_file_path}")
