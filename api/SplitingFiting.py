import torch
import torch.optim as optim
import numpy as np
import scipy as sp
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from BaselineRemoval import BaselineRemoval as br
from scipy.interpolate import interp1d
from tqdm import tqdm


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    # return data / np.max(data)


def baseline(x):
    obj = br(x)
    return obj.ZhangFit(lambda_=0.5)


# 定义插值函数
def interplotation(x, dim=3000):
    f = interp1d(np.arange(len(x)), x, kind='cubic')
    return f(np.linspace(0, len(x) - 1, dim))


def gaussian_cauchy(x, mu, sigma, amp, weight):
    gaussian_component = weight * amp * np.exp(-((x - mu) / sigma) ** 2 / 2)
    cauchy_component = (1 - weight) * amp / (np.pi * sigma * (1 + ((x - mu) / sigma) ** 2))
    return gaussian_component + cauchy_component


class GaussianCauchyModel(torch.nn.Module):
    def __init__(self, num_peaks):
        super(GaussianCauchyModel, self).__init__()
        self.num_peaks = num_peaks
        self.mu = torch.nn.Parameter(torch.rand(num_peaks))
        self.sigma = torch.nn.Parameter(torch.rand(num_peaks))
        self.amp = torch.nn.Parameter(torch.rand(num_peaks))
        self.weight = torch.nn.Parameter(torch.rand(num_peaks))

    def forward(self, x, mu=None):
        if mu is not None:
            mu = torch.tensor(mu, dtype=torch.float32).clone().to(self.mu.device)
            self.mu.data = mu
        else:
            self.mu.data = torch.abs(self.mu.data)
        self.mu.data = self.mu.data.clamp(min=0, max=len(x) - 1)
        self.sigma.data = torch.abs(self.sigma.data).clamp(min=1e-5)
        self.amp.data = torch.abs(self.amp.data).clamp(min=0, max=1)
        self.weight.data = self.weight.data.clamp(min=0, max=1)

        gaussian_component = torch.sum(
            self.weight * self.amp * torch.exp(-((x.unsqueeze(1) - self.mu) / self.sigma) ** 2 / 2), dim=1)
        cauchy_component = torch.sum(
            (1 - self.weight) * self.amp / (np.pi * self.sigma * (1 + ((x.unsqueeze(1) - self.mu) / self.sigma) ** 2)),
            dim=1)
        return gaussian_component + cauchy_component


class PeakParsing:
    def __init__(self, spectrum, epochs=1000, lr=0.1, device='cpu'):
        self.spectrum = normalization(spectrum)
        self.spectrum = interplotation(self.spectrum)
        self.spectrum = sp.signal.savgol_filter(self.spectrum, 7, 2)
        self.spectrum = torch.tensor(self.spectrum, dtype=torch.float32)
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.num_peaks = 0
        self.raw_peaks = None
        self._init_params()
        self._fit_data()

    def peak_loss(self):
        model_output = self.model(torch.arange(len(self.spectrum)).to(self.device), mu=self.model.mu)
        loss_weight = self.spectrum
        loss = torch.sum(loss_weight * (self.spectrum - model_output) ** 2)
        return loss

    def _init_params(self):
        peaks, _ = sp_signal.find_peaks(self.spectrum, distance=5)
        widths = sp_signal.peak_widths(self.spectrum, peaks, rel_height=0.5)[0]
        mu_list = peaks
        sigma_list = widths
        self.num_peaks = len(mu_list)
        amp_list = self.spectrum[mu_list]
        weight_list = np.ones(len(mu_list)) * 0.5
        self.raw_peaks = mu_list
        self.model = GaussianCauchyModel(self.num_peaks)

        mu_data = torch.tensor(mu_list, dtype=torch.float32).clone().detach().requires_grad_(True).to(self.device)
        sigma_data = torch.tensor(sigma_list, dtype=torch.float32).clone().detach().requires_grad_(True).to(self.device)
        amp_data = torch.tensor(amp_list, dtype=torch.float32).clone().detach().requires_grad_(True).to(self.device)
        weight_data = torch.tensor(weight_list, dtype=torch.float32).clone().detach().requires_grad_(True).to(
            self.device)
        self.model.mu.data = mu_data
        self.model.sigma.data = sigma_data
        self.model.amp.data = amp_data
        self.model.weight.data = weight_data
        self.spectrum = self.spectrum.to(self.device)

        self.init_params = {'mu': mu_list, 'sigma': sigma_list, 'amp': amp_list, 'weight': weight_list}

    def _fit_data(self):
        # Adam优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 训练过程
        fit_bar = tqdm(range(self.epochs))
        for i in fit_bar:
            optimizer.zero_grad()
            loss = self.peak_loss()
            loss.backward()
            optimizer.step()

            fit_bar.set_description(f'EPOCH: {i + 1}, LOSS: {loss.item():.4f}')

    def predict_spectrum(self):
        x = torch.arange(len(self.spectrum), dtype=torch.float32).to(self.device)
        return self.model(x, mu=self.model.mu).cpu().detach().numpy()

    def get_params(self, return_init_params=False):
        if return_init_params:
            return self.init_params
        else:
            self.mu_list = self.model.mu.data.cpu().numpy()
            self.sigma_list = self.model.sigma.data.cpu().numpy()
            self.amp_list = self.model.amp.data.cpu().numpy()
            self.weight_list = self.model.weight.data.cpu().numpy()

            optim_params = {'mu': self.mu_list, 'sigma': self.sigma_list, 'amp': self.amp_list,
                            'weight': self.weight_list}
            return optim_params


    def get_peaks(self):
        return self.raw_peaks

    def gaussian_cauchy(x, mu, sigma, amp, weight):
        gaussian_component = weight * amp * np.exp(-((x - mu) / sigma) ** 2 / 2)
        cauchy_component = (1 - weight) * amp / (np.pi * sigma * (1 + ((x - mu) / sigma) ** 2))
        return gaussian_component + cauchy_component


