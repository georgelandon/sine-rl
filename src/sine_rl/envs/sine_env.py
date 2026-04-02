from __future__ import annotations
import os
import numpy as np
import gymnasium as gym

class SineEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        episode_length: int = 600,
        training: bool = False,
        render_mode: str | None = None,
        width: int = 800,
        height: int = 600,
    ):
        super().__init__()

        self.episode_length = episode_length
        self.training = training
        self.render_mode = render_mode
        self.width = width
        self.height = height

        self.action_space = gym.spaces.Box(low=-500.0, high=500.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Pygame state (lazy)
        self._pygame_inited = False
        self._screen = None
        self._surface = None
        self._clock = None

        self.reset()

    def _ensure_pygame(self) -> None:
        if self._pygame_inited:
            return

        # Headless safe (e.g. remote linux). Only matters if render is enabled.
        if os.environ.get("DISPLAY", "") == "":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame
        pygame.init()
        self._clock = pygame.time.Clock()

        # Off-screen surface (always available for rgb_array)
        self._surface = pygame.Surface((self.width, self.height))

        # On-screen window only for "human"
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("SineEnv")

        self._pygame_inited = True
        self._clear()

    def _clear(self) -> None:
        import pygame
        self._surface.fill((255, 255, 255))
        if self._screen is not None:
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.amplitude = np.random.uniform(1, 100)
        self.frequency = np.random.uniform(0.01, 0.1)
        self.phase = np.random.uniform(0, 2 * np.pi)

        self.time = 0
        self.y_pred = np.random.uniform(-150, 150)
        self.prev_y_true = self.y_pred

        if (not self.training) and (self.render_mode in ("human", "rgb_array")):
            self._ensure_pygame()

        obs = np.array([self.time, 0.0, 0.0, self.frequency], dtype=np.float32)
        return obs, {}

    def step(self, action):
        y_true = self.amplitude * np.sin(self.frequency * self.time + self.phase)
        self.y_pred += float(action[0])

        # time increment
        self.time += 1

        if self.time > 1:
            slope = (y_true - self.prev_y_true) / 1.0
        else:
            slope = 0.0

        # Reward: distance error scaled by a slope-sensitive factor (more sensitive near extrema)
        sensitivity_factor = float(np.exp(-abs(slope)))
        reward = -abs(y_true - self.y_pred) * sensitivity_factor

        self.prev_y_true = y_true

        terminated = False
        truncated = self.time >= self.episode_length

        if (not self.training) and (self.render_mode in ("human", "rgb_array")):
            self._ensure_pygame()
            self._draw(y_true=y_true, y_pred=self.y_pred, sensitivity_factor=sensitivity_factor)
            self._present()

        obs = np.array([self.time, y_true, self.y_pred, slope], dtype=np.float32)
        return obs, float(reward), terminated, truncated, {}

    def _draw(self, y_true: float, y_pred: float, sensitivity_factor: float) -> None:
        import pygame

        x = int(self.time % self.width)
        y_center = self.height // 2

        y_true_px = int(y_center - y_true)
        y_pred_px = int(y_center - y_pred)

        g = max(0, min(255, int(sensitivity_factor * 255)))
        pygame.draw.circle(self._surface, (0, g, 255), (x, y_true_px), 1)
        pygame.draw.circle(self._surface, (255, 0, 0), (x, y_pred_px), 1)

        # Handle quit events only for human rendering
        if self._screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Let caller close; we just stop updating window
                    self.render_mode = None

    def _present(self) -> None:
        import pygame
        if self._screen is not None:
            pygame.event.pump()
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.render_mode is None:
            return None

        self._ensure_pygame()

        if self.render_mode == "rgb_array":
            import pygame
            arr = pygame.surfarray.array3d(self._surface)  # (W,H,3)
            return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

        # "human" draws in _present()
        return None

    def close(self):
        if not self._pygame_inited:
            return
        import pygame
        pygame.quit()
        self._pygame_inited = False
        self._screen = None
        self._surface = None
        self._clock = None
