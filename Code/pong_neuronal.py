import pong

import numpy as np
from typing import Callable
import random
import pandas as pd
from math import tan
import torch
from torch import nn
from collections import deque

n_inputs = 5  # == env.observation_space.shape[0]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


# model = Sequential(
#     [
#         keras.layers.Dense(30, activation="sigmoid", input_shape=[n_inputs]),
#         keras.layers.Dense(30, activation="relu"),
#         keras.layers.Dense(30, activation="relu"),
#         # keras.layers.Dense(10, activation="relu"),
#         # evtl mehr versteckte Schichten
#         keras.layers.Dense(
#             3,
#             activation="softmax",  # bei mehr als einem Output Softmax-Aktivierungsfunktion
#         ),
#     ]
# )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Softmax(),
            nn.Linear(256, 128),
            nn.Softmax(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

n_iterrations = 150
n_episodes_per_update = 100
n_max_steps = 200
discount_factor = 0.99
# optimizer = keras.optimizers.Adam()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
loss_fn = torch.nn.HuberLoss()
batch_size = 32


history = []
progress = 10

# model.compile(
#     optimizer=optimizer,
#     loss=loss_fn,
# )


def train_one_epoch(
    model: NeuralNetwork,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.HuberLoss,
):

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(inputs)

    # Compute the loss and its gradients
    loss = loss_fn(outputs, labels)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    # Gather data and report
    return loss.item()


def play_one_step(
    game: pong.Game,
    obs: np.ndarray,
    model: NeuralNetwork,
    loss_fn: Callable,
    zufaelligkeit: float,
):

    if random.random() > zufaelligkeit:
        probabilities: torch.Tensor = model(
            torch.from_numpy(np.reshape(obs.astype(np.float32), (1, 5))).to(
                device=device
            )
        )
        with torch.no_grad():
            np_probs = probabilities.to(torch.device("cpu")).numpy()[0]
            # print(np_probs)
            action = np.argmax(np_probs, axis=0)
    else:
        action = random.randint(0, 2)

    obs, reward, done = game.pong_step(action)
    return obs, reward, done, action


def play_multible_episodes(
    game: pong.Game,
    n_episodes: int,
    n_max_steps: int,
    model: NeuralNetwork,
    loss_fn: Callable,
    zufaelligkeit: float,
    progress: int,
):
    all_rewards = []
    all_grads = []
    # winkel = 0
    winkel = int(5 + (progress) / 10 * 40)
    softmax = torch.nn.Softmax(dim=1)
    for episode in range(n_episodes):
        current_rewards = []
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
        indices = np.random.randint(len(current_obs), size=batch_size)

        with torch.no_grad():
            targets = np.sum(current_rewards) / 1000 + discount_factor * (
                np.amax(
                    model(
                        torch.from_numpy(np.array(next_obs, dtype=np.float32)).to(
                            device=device,
                        )
                    )
                    .to(torch.device("cpu"))
                    .numpy()
                )
            )

            target_vector: np.ndarray = (
                softmax(
                    model(
                        torch.from_numpy(np.array(current_obs, dtype=np.float32)).to(
                            device=device,
                        )
                    )
                )
                .to(torch.device("cpu"))
                .numpy()
            )
        model.train()
        indexes = np.array([i for i in range(target_vector.shape[0])])
        actions = np.array(current_actions)
        target_vector[[indexes], [actions]] = targets
        loss = train_one_epoch(
            model=model,
            inputs=torch.from_numpy(np.array(current_obs, dtype=np.float32)).to(
                device=device
            ),
            labels=torch.from_numpy(target_vector.astype(dtype=np.float32)).to(
                device=device
            ),
            loss_fn=loss_fn,
        )
        model.eval()
        # model.fit(np.array(current_obs), target_vector, epochs=1, verbose=0)

        # optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
        print(
            f"rand {zufaelligkeit:3.0%}, reward={np.sum(current_rewards):10.3f}, progress={progress:3}, loss={loss:20.4f}"
        )
        history.append(
            (
                zufaelligkeit,
                np.sum(current_rewards),
                progress,
                loss,
            )
        )

    pd.DataFrame.from_records(
        history, columns=["random", "reward", "progress", "loss"]
    ).to_csv("history.csv")


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
            play_multible_episodes(
                game,
                n_episodes_per_update,
                n_max_steps,
                model,
                loss_fn,
                zufaelligkeit,
                rounds,
            )

    game.self.quit_grafik()
