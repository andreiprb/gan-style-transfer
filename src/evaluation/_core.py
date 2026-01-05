import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg


@dataclass
class EvaluationResult:
    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    fid_score: float | None = None
    ssim_score: float | None = None
    training_loss_history: list[float] = field(default_factory=list)
    convergence_epoch: int | None = None
    total_params: int | None = None
    training_time_seconds: float | None = None
    notes: str = ''

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'fid_score': self.fid_score,
            'ssim_score': self.ssim_score,
            'training_loss_history': self.training_loss_history,
            'convergence_epoch': self.convergence_epoch,
            'total_params': self.total_params,
            'training_time_seconds': self.training_time_seconds,
            'notes': self.notes
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'EvaluationResult':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class InceptionV3Features(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self._load_inception()

    def _load_inception(self):
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        inception.eval()
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ).to(self.device)
        for param in self.blocks.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        features = self.blocks(x)
        features = features.view(features.size(0), -1)
        return features


class FIDCalculator:
    def __init__(self, device: torch.device, num_features: int = 1000):
        self.device = device
        self.num_features = num_features
        self.inception = InceptionV3Features(device)
        self.real_mu: np.ndarray | None = None
        self.real_sigma: np.ndarray | None = None

    @torch.no_grad()
    def _extract_features(
        self,
        images: torch.Tensor | DataLoader,
        max_samples: int | None = None
    ) -> np.ndarray:
        self.inception.eval()
        features_list = []
        n_samples = 0
        max_samples = max_samples or self.num_features
        if isinstance(images, DataLoader):
            for batch in images:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                features = self.inception(batch)
                features_list.append(features.cpu().numpy())
                n_samples += batch.size(0)
                if n_samples >= max_samples:
                    break
        else:
            images = images.to(self.device)
            features = self.inception(images)
            features_list.append(features.cpu().numpy())
        features = np.concatenate(features_list, axis=0)[:max_samples]
        return features

    def compute_real_statistics(self, real_loader: DataLoader):
        print(f'Computing statistics for {self.num_features} real images...')
        features = self._extract_features(real_loader, self.num_features)
        self.real_mu = np.mean(features, axis=0)
        self.real_sigma = np.cov(features, rowvar=False)
        print('Real image statistics computed.')

    def calculate_fid(self, generated_images: torch.Tensor | DataLoader) -> float:
        if self.real_mu is None or self.real_sigma is None:
            raise RuntimeError('Must call compute_real_statistics() before calculate_fid()')
        gen_features = self._extract_features(generated_images, self.num_features)
        gen_mu = np.mean(gen_features, axis=0)
        gen_sigma = np.cov(gen_features, rowvar=False)
        fid = self._calculate_frechet_distance(self.real_mu, self.real_sigma, gen_mu, gen_sigma)
        return float(fid)

    @staticmethod
    def _calculate_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-06
    ) -> float:
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.001):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return fid


class SSIMCalculator:
    def __init__(self, window_size: int = 11, channel: int = 3):
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        def gaussian(window_size: int, sigma: float) -> torch.Tensor:
            gauss = torch.tensor(
                [np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)]
            )
            return gauss / gauss.sum()

        _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        device = img1.device
        window = self.window.to(device)
        channel = img1.size(1)
        if channel != self.channel:
            window = self._create_window(self.window_size, channel).to(device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())

    def calculate_batch(self, batch1: torch.Tensor, batch2: torch.Tensor) -> list[float]:
        scores = []
        for i in range(batch1.size(0)):
            score = self.calculate(batch1[i], batch2[i])
            scores.append(score)
        return scores


class TrainingMonitor:
    def __init__(self, model_name: str, log_dir: str = 'outputs/logs'):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.losses: dict[str, list[float]] = {}
        self.gradients: dict[str, list[float]] = {}
        self.learning_rates: list[float] = []
        self.epoch_times: list[float] = []
        self.start_time: datetime | None = None
        self.current_epoch: int = 0

    def start_training(self):
        self.start_time = datetime.now()

    def log_loss(self, name: str, value: float):
        if name not in self.losses:
            self.losses[name] = []
        self.losses[name].append(value)

    def log_gradient_norm(self, name: str, model: nn.Module):
        if name not in self.gradients:
            self.gradients[name] = []
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradients[name].append(total_norm)

    def log_learning_rate(self, lr: float):
        self.learning_rates.append(lr)

    def log_epoch_time(self, seconds: float):
        self.epoch_times.append(seconds)
        self.current_epoch += 1

    def get_training_time(self) -> float | None:
        if self.start_time is None:
            return None
        return (datetime.now() - self.start_time).total_seconds()

    def detect_convergence(
        self,
        loss_name: str = 'g_loss',
        window: int = 10,
        threshold: float = 0.01
    ) -> int | None:
        if loss_name not in self.losses:
            return None
        losses = self.losses[loss_name]
        if len(losses) < window * 2:
            return None
        for i in range(window, len(losses)):
            window_losses = losses[i - window:i]
            variance = np.var(window_losses)
            if variance < threshold:
                return i - window
        return None

    def save(self):
        data = {
            'model_name': self.model_name,
            'losses': self.losses,
            'gradients': self.gradients,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'total_epochs': self.current_epoch,
            'total_time_seconds': self.get_training_time(),
            'convergence_epoch': self.detect_convergence()
        }
        path = self.log_dir / f'{self.model_name}_training_log.json'
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Training log saved to {path}')


class Evaluator:
    def __init__(self, model_name: str, device: torch.device, output_dir: str = 'outputs/reports'):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fid_calc = FIDCalculator(device)
        self.ssim_calc = SSIMCalculator()

    def set_real_data(self, real_loader: DataLoader):
        self.fid_calc.compute_real_statistics(real_loader)

    @torch.no_grad()
    def evaluate(
        self,
        generator: nn.Module,
        content_loader: DataLoader,
        training_monitor: TrainingMonitor | None = None,
        num_samples: int = 1000
    ) -> EvaluationResult:
        generator.eval()
        result = EvaluationResult(model_name=self.model_name)
        result.total_params = sum(p.numel() for p in generator.parameters())
        print(f'Generating {num_samples} images for FID calculation...')
        generated_images = []
        content_images = []
        n_generated = 0
        for batch in content_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            with torch.no_grad():
                fake = generator(batch)
            generated_images.append(fake.cpu())
            content_images.append(batch.cpu())
            n_generated += batch.size(0)
            if n_generated >= num_samples:
                break
        generated_images = torch.cat(generated_images, dim=0)[:num_samples]
        content_images = torch.cat(content_images, dim=0)[:num_samples]
        print('Calculating FID...')
        result.fid_score = self.fid_calc.calculate_fid(generated_images)
        print(f'FID Score: {result.fid_score:.2f}')
        print('Calculating SSIM...')
        ssim_scores = self.ssim_calc.calculate_batch(content_images, generated_images)
        result.ssim_score = float(np.mean(ssim_scores))
        print(f'SSIM Score: {result.ssim_score:.4f}')
        if training_monitor is not None:
            if 'g_loss' in training_monitor.losses:
                result.training_loss_history = training_monitor.losses['g_loss']
            result.convergence_epoch = training_monitor.detect_convergence()
            result.training_time_seconds = training_monitor.get_training_time()
        return result

    def compare_models(self, results: list[EvaluationResult]) -> dict[str, dict]:
        comparison = {
            'fid_ranking': sorted(
                [(r.model_name, r.fid_score) for r in results if r.fid_score],
                key=lambda x: x[1]
            ),
            'ssim_ranking': sorted(
                [(r.model_name, r.ssim_score) for r in results if r.ssim_score],
                key=lambda x: x[1],
                reverse=True
            ),
            'params_ranking': sorted(
                [(r.model_name, r.total_params) for r in results if r.total_params],
                key=lambda x: x[1]
            ),
            'convergence_ranking': sorted(
                [(r.model_name, r.convergence_epoch) for r in results if r.convergence_epoch],
                key=lambda x: x[1]
            )
        }
        return comparison