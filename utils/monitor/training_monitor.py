import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import deque
from typing import Union, List
import os
import numpy as np
from .plot_config import set_plot_style

# Import rich components
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

class TrainingMonitor:
    """
    Handles live plotting with XKCD support, CLI logging, and file logging.
    Optimized for Reward-on-top visibility and trend envelopes.
    """
    def __init__(
        self,
        env_name: str,
        agent_name: str,
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

        self.full_reward_history = []
        self.full_loss_history = []
        self.agent_name = agent_name

        # --- Rich CLI Setup ---
        self.console = Console()
        self.progress = None
        if 'cli' in self.modes:
            self.progress = Progress(
                TextColumn("[bold blue]{task.fields[env]:>16.16}", justify="left"),
                BarColumn(bar_width=20), 
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TextColumn("[bold green]Avg R: {task.fields[avg_r]:>8.2f}"),
                "•",
                TextColumn("[bold magenta]Loss: {task.fields[loss]:>10.4f}"),
                "•",
                TimeElapsedColumn(),
                "/",
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            self.task = self.progress.add_task(
                "Training", 
                total=total_iterations, 
                env=env_name, 
                avg_r=0.0, 
                loss=0.0
            )

        # --- File Logging Setup ---
        if 'file' in self.modes:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, f'{env_name}_training.csv')
            self._log_file = open(self.log_file_path, 'w')
            self._log_file.write("Iteration,AverageReward,Loss\n")
            self._log_file.flush()

        # --- Live Plot Setup ---
        if 'live' in self.modes:
            styles = set_plot_style()
            plt.ion()
            self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
            self.fig.subplots_adjust(top=0.88)
            self.ax2 = self.ax1.twinx()

            self.ax1.set_xlabel('Iteration', fontweight='bold')
            self.ax1.set_ylabel('Reward', fontweight='bold', color='tab:blue')
            self.ax2.set_ylabel('Loss', fontweight='bold', color='tab:orange')

            self.ax1.tick_params(
                axis='y', color='tab:blue', labelcolor='tab:blue')
            self.ax2.tick_params(
                axis='y', color='tab:orange', labelcolor='tab:orange')
            
            # Layering: Reward (ax1) on top of Loss (ax2)
            self.ax1.set_zorder(self.ax2.get_zorder() + 1)
            self.ax1.patch.set_visible(False)

            # 1. Raw Reward Line (Bold, Light Purple)
            self.line_raw, = self.ax1.plot([], [], color='tab:purple', alpha=0.3, linewidth=2, label='Raw Reward', zorder=5)
            # 2. Trend Line (Bold, Darker Blue)
            self.line_trend, = self.ax1.plot([], [], color='tab:blue', linewidth=2.5, label='Trend (SMA)', zorder=7)
            # 3. Loss Line (Dashed Orange)
            self.line_loss, = self.ax2.plot([], [], color='tab:orange', linestyle='--', alpha=0.6, label='Loss', zorder=3)
            
            self.fill_area = None
            self.fig.suptitle(f'{agent_name}: {env_name}', **styles['suptitle'])

            # Legend Layout
            lines = [self.line_trend, self.line_raw, self.line_loss]
            labels = [l.get_label() for l in lines]

    def update(self, iteration: int, avg_reward: float, loss: float):
        self.episode_counts.append(iteration)
        self.avg_rewards_history.append(avg_reward)
        self.loss_history.append(loss)
        
        # Keep full history for final plotting
        self.full_reward_history.append(avg_reward)
        self.full_loss_history.append(loss)

        if self.progress:
            self.progress.update(
                self.task, completed=iteration, avg_r=avg_reward, loss=loss)

        if 'file' in self.modes and self._log_file:
            self._log_file.write(f"{iteration},{avg_reward},{loss}\n")
            self._log_file.flush()

        if 'live' in self.modes:
            self._update_plot()

    def _update_plot(self):
        # 使用 deque 內的資料進行 live 繪製
        x_data = list(self.episode_counts)[-len(self.avg_rewards_history):]
        y_reward = np.array(self.avg_rewards_history)
        y_loss = list(self.loss_history)
        if len(y_reward) == 0: return

        # 1: Centered Window SMA (處理邊界效應)
        window = 20
        half_w = window // 2
        smoothed, std_up, std_lo = [], [], []
        
        for i in range(len(y_reward)):
            # 取對稱窗口 [i - half_w, i + half_w]
            start = max(0, i - half_w)
            end = min(len(y_reward), i + half_w + 1)
            view = y_reward[start:end]
            
            mu, std = np.mean(view), np.std(view)
            smoothed.append(mu)
            std_up.append(mu + 1.28 * std)
            std_lo.append(mu - 1.28 * std)

        # 2. Update Lines
        self.line_raw.set_data(x_data, y_reward)
        self.line_trend.set_data(x_data, smoothed)
        self.line_loss.set_data(x_data, y_loss)

        # 3. Update Shaded Area
        if len(y_reward) > 1:
            if self.fill_area: self.fill_area.remove()
            self.fill_area = self.ax1.fill_between(
                x_data, std_lo, std_up, color='tab:blue', alpha=0.12, zorder=4, linewidth=0
            )

        # 4. Refresh
        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if self.progress:
            self.progress.stop()
            self.console.print(f"[bold green]Training on {self.env_name} completed![/bold green]")

        if 'file' in self.modes:
            self._save_final_plot()

        if 'live' in self.modes:
            plt.ioff()

        if 'file' in self.modes and hasattr(self, '_log_file'):
            self._log_file.close()

    def _save_final_plot(self):
        """Generates and saves a high-res summary plot of the entire training."""
        plt.ioff()
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        
        x = self.episode_counts
        y_r = np.array(self.full_reward_history)
        y_l = self.full_loss_history

        window = max(20, len(x) // 50)
        half_w = window // 2
        smoothed = [np.mean(y_r[max(0, i-half_w):min(len(y_r), i+half_w+1)]) for i in range(len(y_r))]

        ax1.plot(x, y_r, color='tab:purple', alpha=0.2, label='Raw Reward')
        ax1.plot(x, smoothed, color='tab:blue', linewidth=2, label='Trend (SMA)')
        ax2.plot(x, y_l, color='tab:orange', linestyle='--', alpha=0.5, label='Loss')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Reward', color='tab:blue')
        ax2.set_ylabel('Loss', color='tab:orange')
        ax1.tick_params(
            axis='y', color='tab:blue', labelcolor='tab:blue')
        ax2.tick_params(
            axis='y', color='tab:orange', labelcolor='tab:orange')
        
        plt.title(f'Final Training Summary: {self.env_name}')
        save_path = self.log_file_path.replace('.csv', '.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        self.console.print(f"[*] Full history plot saved to {save_path}")
        plt.close(fig)
