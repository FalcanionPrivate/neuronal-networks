import tensorflow as tf
import keras
import pong
from keras.models import Sequential
import numpy as np
from typing import Callable
import random
import pandas as pd

n_inputs = 5  # == env.observation_space.shape[0]

model = Sequential(
    [
        keras.layers.Dense(20, activation="sigmoid", input_shape=[n_inputs]),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(20, activation="relu"),
        # keras.layers.Dense(10, activation="relu"),
        # evtl mehr versteckte Schichten
        keras.layers.Dense(
            3,
            activation="linear",  # bei mehr als einem Output Softmax-Aktivierungsfunktion
        ),
    ]
)

n_iterrations = 150
n_episodes_per_update = 50
n_max_steps = 200
discount_factor = 0.95
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
history = []
progress = 10

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
)


def play_one_step(
    game: pong.Game,
    obs: np.ndarray,
    model: keras.Model,
    loss_fn: Callable,
    zufaelligkeit: float,
):
    with tf.GradientTape() as tape:
        probabilities = model(np.reshape(obs, (1, 5)))
        if random.random() > zufaelligkeit:
            action = np.argmax(probabilities.numpy()[0], axis=0)
        else:
            action = random.randint(0, 2)
        y_target = tf.cast(
            np.array([1 if i == action else 0 for i in range(3)], ndmin=2), tf.float32
        )
        loss = tf.reduce_mean(loss_fn(y_target, probabilities))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done = game.pong_step(action)
    return obs, reward, done, action


def play_multible_episodes(
    game: pong.Game,
    n_episodes: int,
    n_max_steps: int,
    model: keras.Model,
    loss_fn: Callable,
    zufaelligkeit: float,
    progress: int,
):
    all_rewards = []
    all_grads = []
    winkel = int(5 + (progress) / 10 * 40)
    for episode in range(n_episodes):
        current_rewards = []
        # current_grads = []
        obs = game.reset(winkel=winkel)
        current_obs = []
        next_obs = []
        current_actions = []
        while True:
            current_obs.append(obs)
            obs, reward, done, action = play_one_step(
                game,
                obs,
                model,
                loss_fn,
                zufaelligkeit,
            )
            next_obs.append(obs)
            current_actions.append(action)
            current_rewards.append(reward)
            # current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        # all_grads.append(current_grads)

        # all_final_rewards = discount_rewards(current_rewards, discount_factor)
        # all_mean_grads = []
        # for var_index in range(len(model.trainable_variables)):
        #     mean_grads = tf.reduce_mean(
        #         [
        #             final_reward * current_grads[step][var_index]
        #             for step, final_reward in enumerate(all_final_rewards)
        #         ],
        #         axis=0,
        #     )
        #     all_mean_grads.append(mean_grads)

        targets = reward + discount_factor * (
            np.amax(model.predict(np.array(next_obs)))
        )

        target_vector = model.predict(np.array(current_obs))
        indexes = np.array([i for i in range(target_vector.shape[0])])
        actions = np.array(current_actions)
        target_vector[[indexes], [actions]] = targets
        model.fit(np.array(current_obs), target_vector, epochs=1, verbose=0)

        # optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
        print(
            game.ball_y,
            game.schlaeger_y,
            zufaelligkeit,
            reward,
            # sum(all_final_rewards),
            progress,
        )
        history.append(
            (
                game.ball_y,
                game.schlaeger_y,
                zufaelligkeit,
                reward,
                # sum(all_final_rewards),
            )
        )
    return all_rewards, all_grads


def discount_rewards(rewards: list, discount_factor: float):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted


def discount_and_normalize_rewards(all_rewards: list, discount_factor: float):
    all_discounted_rewards = [
        discount_rewards(rewards, discount_factor) for rewards in all_rewards
    ]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()  # durchschnitt des reward arrays
    reward_std = flat_rewards.std()  # standartabweichung des reward arrays
    return [
        (discount_rewards - reward_mean)
        / reward_std  # herabgesetzte rewards - durchschnitt / standartabweichung
        for discount_rewards in all_discounted_rewards  # für jeden reward im array aller rewards
    ]


if __name__ == "__main__":
    game = pong.Game(mit_grafik=True)
    for rounds in range(progress):
        for interation in range(n_iterrations):
            zufaelligkeit = 0.5 * (n_iterrations - interation) / n_iterrations
            all_rewards, all_grads = play_multible_episodes(
                game,
                n_episodes_per_update,
                n_max_steps,
                model,
                loss_fn,
                zufaelligkeit,
                rounds,
            )
            # all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
            # all_mean_grads = []
            # for var_index in range(len(model.trainable_variables)):
            #     mean_grads = tf.reduce_mean(
            #         [
            #             final_reward * all_grads[episode_index][step][var_index]
            #             for episode_index, final_rewards in enumerate(all_final_rewards)
            #             for step, final_reward in enumerate(final_rewards)
            #         ],
            #         axis=0,
            #     )
            #     all_mean_grads.append(mean_grads)
            # optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
        # pd.DataFrame.from_records(
        #     history, columns=[("ball_y", "schlaeger_y", "zufaelligkeit", "reward")]
        # ).to_csv("history.csv")
    game.self.quit_grafik()
