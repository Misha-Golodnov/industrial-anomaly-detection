"""Небольшой NeuTraL AD для нормальных окон; PyTorch — необязательная зависимость.

Самостоятельная реализация DCL по Qiu et al., ICML 2021, с MLP для коротких
окон TEP. Не воспроизводит архитектуру LNT и не использует метки отказов.
"""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.base import NoveltyDetector, check_windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Установите PyTorch: uv pip install --python .venv/Scripts/python.exe "
            "-r requirements-neural.txt"
        ) from exc
    return torch


def deterministic_contrastive_loss(embeddings, temperature: float = 0.1):
    """DCL на один объект: original — положительная пара каждой трансформации.

    Знаменатель включает original и остальные преобразования, кроме самой
    anchor. logsumexp исключает переполнение при малой температуре.
    """
    torch = _torch()
    if temperature <= 0 or embeddings.ndim != 3 or embeddings.shape[1] < 3:
        raise ValueError("Нужны original, >= 2 трансформаций и temperature > 0.")
    unit = torch.nn.functional.normalize(embeddings, dim=-1)
    similarities = unit @ unit.transpose(1, 2) / temperature
    count = embeddings.shape[1]
    diagonal = torch.eye(count, dtype=torch.bool, device=embeddings.device)[None]
    logits = similarities.masked_fill(diagonal, -torch.inf)
    terms = torch.logsumexp(logits[:, 1:], dim=-1) - similarities[:, 1:, 0]
    transforms = count - 1
    return terms.sum(dim=1) / (transforms * np.log(transforms))


def _network(dimension: int, hidden: int, latent: int, transforms: int):
    torch = _torch()
    nn = torch.nn

    class TransformationNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, latent)
            )
            self.transforms = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dimension, 32), nn.ReLU(), nn.Linear(32, dimension)
                    )
                    for _ in range(transforms)
                ]
            )

        def forward(self, x):
            views = torch.stack(
                [x, *[x + transform(x) for transform in self.transforms]], dim=1
            )
            return self.encoder(views)

    return TransformationNetwork()


class NeuralTransformationIndex(NoveltyDetector):
    """DCL-оценка четырёх обучаемых остаточных преобразований развёрнутого окна.

    Число эпох фиксируется до теста. Масштабирование по каналам выполняется
    общим ChannelScaler; нормализации отдельного окна нет, уровень сохранён.
    learn_transformations=False — абляция с фиксированными случайными преобразованиями.
    """

    def __init__(
        self,
        *,
        epochs: int = 20,
        batch_size: int = 256,
        hidden: int = 128,
        latent: int = 32,
        transforms: int = 4,
        learning_rate: float = 1e-3,
        seed: int = 0,
        learn_transformations: bool = True,
    ) -> None:
        if min(epochs, batch_size, hidden, latent) < 1 or transforms < 2:
            raise ValueError("Неверная размерность или число эпох/трансформаций.")
        if learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным.")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.transforms = int(transforms)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.learn_transformations = bool(learn_transformations)

    def fit_windows(self, X: np.ndarray) -> None:
        torch = _torch()
        values = self._values(X)
        self.shape_ = values.shape[1:]
        dimension = int(np.prod(self.shape_))
        data = torch.from_numpy(values.reshape(len(values), dimension))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network_ = _network(
                dimension, self.hidden, self.latent, self.transforms
            )
            if not self.learn_transformations:
                self.network_.transforms.requires_grad_(False)
            optimizer = torch.optim.Adam(
                [p for p in self.network_.parameters() if p.requires_grad],
                lr=self.learning_rate,
                weight_decay=1e-5,
            )
            generator = torch.Generator().manual_seed(self.seed)
            self.train_history_ = []
            self.network_.train()
            for _ in range(self.epochs):
                permutation = torch.randperm(len(data), generator=generator)
                total = 0.0
                for indices in permutation.split(self.batch_size):
                    loss = deterministic_contrastive_loss(
                        self.network_(data[indices])
                    ).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Нейросетевой loss перестал быть конечным.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 5.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(indices)
                self.train_history_.append(total / len(data))
        self.network_.eval()
        self.n_parameters_ = sum(p.numel() for p in self.network_.parameters())

    @staticmethod
    def _values(X: np.ndarray) -> np.ndarray:
        values = check_windows(X, allow_empty=True)
        if not np.isfinite(values).all():
            raise ValueError("Окна должны быть конечными.")
        return np.asarray(values, dtype=np.float32, order="C")

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("Сначала вызовите fit.")
        torch = _torch()
        values = self._values(X)
        if values.shape[1:] != self.shape_:
            raise ValueError("Форма окна отличается от обучающей.")
        result = np.empty(len(values), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                chunk = values[start : start + self.batch_size]
                data = torch.from_numpy(chunk.reshape(len(chunk), -1))
                result[start : start + len(chunk)] = deterministic_contrastive_loss(
                    self.network_(data)
                ).numpy()
        return result
