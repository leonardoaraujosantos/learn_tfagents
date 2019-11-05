import numpy as np
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
import pyvirtualdisplay

# Embed video on notebook
def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


# Calculate the average of rewards by number of episodes
def compute_avg_return(environment, policy, num_episodes=10):    
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def plot_avg_returns_by_step(steps, returns):
    plt.plot(steps, returns)
    plt.ylabel('Average Return');
    plt.xlabel('Step');
    plt.title('Average Returns vs Step');
    plt.show()
    
def make_video(video_filename, eval_env, eval_py_env, tf_agent, num_episodes=3):
    num_episodes = 3
    video_filename = 'imageio.mp4'
    with imageio.get_writer(video_filename, fps=60) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = tf_agent.policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())