import gymnasium as gym
from state_encoders.grid_encoder import SimpleCNNArchitecture, GridEncoder

def main(env_name='CarRacing-v3', T=200, N=10):
    env = gym.make(env_name, continuous=False)
    state_dim = env.observation_space.shape
    env.close()

    cnn_architecture = SimpleCNNArchitecture(
        in_channels=state_dim[2],
        height=state_dim[0],
        width=state_dim[1]
    )
    grid_encoder = GridEncoder(state_dim, cnn_architecture)
    print(f"Grid Encoder Output Dimension: {grid_encoder.get_output_dim()}")

if __name__ == '__main__':
    main(T=500)
