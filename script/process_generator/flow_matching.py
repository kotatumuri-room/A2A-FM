import torch
from script.process_generator.base_process import ProcessGenerator
from hydra.utils import instantiate
from script.process_generator.optimal_transport import OptimalTransportCalculator


class OptimalTransportFlowMatcher(ProcessGenerator):
    """
    Implementaion of Tong et al. (2023), Improving and generalizing flow-based generative models with minibatch optimal transport
        this class can be used for most of the OT based flow matching methods by altering the ot_calculator
    """

    def __init__(self, ot_calculator: OptimalTransportCalculator, **kwargs):
        super().__init__(**kwargs)
        self.ot_calculator = instantiate(ot_calculator, data_handler=self.data_handler)

    def set_batch(self, z0: torch.Tensor, z1: torch.Tensor):
        self.z0, self.z1 = self.ot_calculator.conduct_sort(z0, z1)

    def construct_target(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        v = self.data_handler.convert_target(self.z1 - self.z0)
        return v

    def get_model_input(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        z0_hat, z1_hat = self.data_handler.model_input_terminals(self.z0, self.z1)
        return z0_hat * (1 - t) + z1_hat * t

    def sample_t(self) -> torch.Tensor:
        batch_size = len(self.z0)
        return torch.rand(batch_size, 1, device=self.z0.device)


class ClassiferFreeGuidanceFlowMatcher(ProcessGenerator):
    """Implementation of Zheng et al. (2023), Guided Flows for Generative Modeling and Decision Making"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_batch(self, z0: torch.Tensor, z1: torch.Tensor):
        self.z0, self.z1 = z0, z1

    def construct_target(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        return self.data_handler.convert_target(
            self.z0 * self.data_handler.sigma_dot(t)
            + self.z1 * self.data_handler.alpha_dot(t)
        )

    def get_model_input(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        z0_hat, z1_hat = self.data_handler.model_input_terminals(self.z0, self.z1)
        zt = self.data_handler.alpha(t) * z1_hat + self.data_handler.sigma(t) * z0_hat
        zt[:, self.data_handler.x_dim :] = z0_hat[:, self.data_handler.x_dim :]
        zt[:, self.data_handler.x_dim + self.data_handler.c_dim - 1] = t.squeeze(1)
        return zt

    def sample_t(self) -> torch.Tensor:
        batch_size = len(self.z0)
        return torch.rand(batch_size, 1, device=self.z0.device) * (1 - 1e-5) + 1e-5
