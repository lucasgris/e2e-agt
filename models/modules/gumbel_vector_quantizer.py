import torch
from torch import nn


class GumbelVectorQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_groups = cfg.num_code_vector_groups
        self.num_vectors = cfg.num_code_vectors_per_group

        self.linear = nn.Linear(
            cfg.extracted_feature_size,
            self.num_groups * self.num_vectors
        )
        self.code_book = nn.Parameter(
            torch.randn(1, self.num_groups, self.num_vectors, cfg.code_vector_size // self.num_groups)
        )

        self.temperature = cfg.gumbel_init_temperature

    @staticmethod
    # @torch.jit.script
    def _compute_perplexity(probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs (torch.Tensor): with shape `(B, L, G, V)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            torch.Tensor with shape `(G, V)`
        """
        where_calculate_probs = torch.arange(probs.size(1), device=probs.device).unsqueeze(0) < lengths.unsqueeze(-1)
        probs = probs[where_calculate_probs == 1]

        # perplexity = torch.zeros(probs.size(-2), probs.size(-1), device=probs.device)
        # for i in range(probs.size(-2)):
        #     for j in range(probs.size(-1)):
        #         perplexity[i, j] = -torch.sum(probs[:, i, j] * torch.log(probs[:, i, j] + 1e-10))

        num_values = probs.size(0)
        perplexity = probs.sum(0) / num_values

        return perplexity

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D1)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            code_vectors (torch.Tensor): with shape `(B, L, D2)`
            perplexity (torch.Tensor): with shape `(G, V)`
            )
        """

        assert torch.isnan(hidden_states).sum() == 0, "NaN in hidden_states"
        assert torch.isnan(self.code_book).sum() == 0, "NaN in code_book"

        batch_size, length, _ = hidden_states.shape

        hidden_states = self.linear(hidden_states)
        # `(B, L, G * V)` -> `(B * L * G, V)`
        hidden_states = hidden_states.view(batch_size * length * self.num_groups, -1)

        code_vector_probs = nn.functional.gumbel_softmax(
            hidden_states.float(), tau=self.temperature, hard=True
        ).type_as(hidden_states)
        assert torch.isnan(code_vector_probs).sum() == 0, "NaN in code_vector_probs"
        code_vector_soft_dist = torch.softmax(
            hidden_states.view(batch_size, length, self.num_groups, -1).float(), dim=-1
        )
        if lengths is None:
            lengths = torch.tensor([code_vector_soft_dist.size(1)] * code_vector_soft_dist.size(0), device=code_vector_soft_dist.device)
        perplexity = self._compute_perplexity(code_vector_soft_dist, lengths)

        code_vector_probs = code_vector_probs.view(batch_size * length, self.num_groups, -1).unsqueeze(-1)
        assert torch.isnan(code_vector_probs).sum() == 0, "NaN in code_vector_probs"
        code_vectors = code_vector_probs * self.code_book

        assert torch.isnan(code_vectors).sum() == 0, "NaN in code_vectors (before sum)"

        # `(B * L, G, V, D)` -> `(B, L, G * D)`
        code_vectors = code_vectors.sum(-2).view(batch_size, length, -1)
        assert torch.isnan(code_vectors).sum() == 0, "NaN in code_vectors (after sum)"
        
        return code_vectors, perplexity

if __name__ == "__main__":
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({ 
        "num_code_vector_groups": 6, # Number of code vector divisions (Default: 2)
        "num_code_vectors_per_group": 72, # Number of code vectors (Default: 320)
        "extracted_feature_size": 768, # Output dimension of feature extractor
        "code_vector_size": 768, # Dimension of quantized code vector (Default: 768)
        "gumbel_init_temperature": 2.0
    })

    model = GumbelVectorQuantizer(cfg)
    # B x C x T
    hidden_states = torch.randn(2, 100, 768)
    lengths = torch.tensor([100, 100])

    code_vectors, perplexity = model(hidden_states, lengths)
    print(code_vectors.shape, perplexity.shape, perplexity)
