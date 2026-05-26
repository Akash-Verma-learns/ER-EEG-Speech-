"""
Ordered Neurons LSTM (ON-LSTM).

Reference: Shen et al., "Ordered Neurons: Integrating Tree Structures into
Recurrent Neural Networks", ICLR 2019.

Key idea: a cumax (cumulative-max via cumsum of softmax) gate enforces a
monotone constraint so that neurons representing long-range dependencies
are updated less frequently than those representing short-range patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def cumax(x: torch.Tensor) -> torch.Tensor:
    """Cumulative-maximum approximation via cumsum of softmax."""
    return torch.cumsum(F.softmax(x, dim=-1), dim=-1)


class ONLSTMCell(nn.Module):
    """Single ON-LSTM step.

    Parameters
    ----------
    input_size  : dimensionality of input x_t
    hidden_size : dimensionality of hidden state h_t  (must be divisible by chunk_size)
    chunk_size  : granularity of the ordered-neuron groups
    """

    def __init__(self, input_size: int, hidden_size: int, chunk_size: int = 16):
        super().__init__()
        if hidden_size % chunk_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by chunk_size ({chunk_size})"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunks = hidden_size // chunk_size

        # Master gates — operate at chunk resolution
        self.W_master = nn.Linear(input_size + hidden_size, 2 * self.n_chunks)

        # Standard LSTM gates — operate at unit resolution
        self.W_lstm = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        self._init_weights()

    def _init_weights(self):
        # Forget-gate bias initialised to 1 (standard LSTM trick)
        nn.init.orthogonal_(self.W_lstm.weight)
        nn.init.zeros_(self.W_lstm.bias)
        bias = self.W_lstm.bias.data
        # Bias layout: [i | f | g | o], each hidden_size wide
        bias[self.hidden_size: 2 * self.hidden_size].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x     : (batch, input_size)
        state : (h_{t-1}, c_{t-1}), each (batch, hidden_size); zeros if None

        Returns
        -------
        h_t   : (batch, hidden_size)
        (h_t, c_t)
        """
        batch = x.size(0)
        device = x.device

        if state is None:
            h = x.new_zeros(batch, self.hidden_size)
            c = x.new_zeros(batch, self.hidden_size)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=-1)  # (batch, input+hidden)

        # --- Master gates (chunk-level ordered constraints) ---
        master = self.W_master(combined)                        # (batch, 2·n_chunks)
        mf_raw, mi_raw = master.chunk(2, dim=-1)               # each (batch, n_chunks)

        # d̃_t: master forget — high-index chunks forget *less* (long memory)
        d_tilde = cumax(mf_raw)                                 # (batch, n_chunks)
        # ĩ_t: master input — high-index chunks receive *less* new input
        i_tilde = 1.0 - cumax(mi_raw)                          # (batch, n_chunks)

        # Expand chunk gates to full hidden size
        # (batch, n_chunks, 1) → (batch, n_chunks, chunk_size) → (batch, hidden)
        d_tilde_full = d_tilde.unsqueeze(-1).expand(
            -1, -1, self.chunk_size
        ).reshape(batch, self.hidden_size)
        i_tilde_full = i_tilde.unsqueeze(-1).expand(
            -1, -1, self.chunk_size
        ).reshape(batch, self.hidden_size)

        # ω_t = d̃_t ⊙ ĩ_t  — region of units that accept new information
        omega = d_tilde_full * i_tilde_full                     # (batch, hidden)

        # --- Standard LSTM gates (unit-level) ---
        gates = self.W_lstm(combined)                           # (batch, 4·hidden)
        g_i, g_f, g_g, g_o = gates.chunk(4, dim=-1)

        gate_i = torch.sigmoid(g_i)
        gate_f = torch.sigmoid(g_f)
        gate_g = torch.tanh(g_g)
        gate_o = torch.sigmoid(g_o)

        # --- ON-LSTM cell update ---
        # c_t = ω_t ⊙ (f_t ⊙ c_{t-1} + i_t ⊙ g_t) + (d̃_t − ω_t) ⊙ c_{t-1}
        c_new = (
            omega * (gate_f * c + gate_i * gate_g)
            + (d_tilde_full - omega) * c
        )
        h_new = gate_o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class ONLSTM(nn.Module):
    """Multi-layer ON-LSTM operating on a sequence.

    Parameters
    ----------
    input_size  : feature dimension of each time step
    hidden_size : hidden dimension (shared across layers)
    num_layers  : number of stacked ON-LSTM layers
    chunk_size  : ordered-neuron chunk size
    dropout     : inter-layer dropout (ignored for single layer)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        chunk_size: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        cells = []
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            cells.append(ONLSTMCell(in_dim, hidden_size, chunk_size))
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout) if (num_layers > 1 and dropout > 0) else None

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        outputs      : (batch, seq_len, hidden_size)
        (h_n, c_n)   : final states, each (batch, hidden_size)
        """
        batch, seq_len, _ = x.shape
        device = x.device

        states = [(None,) for _ in self.cells]

        all_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_t, state_t = cell(x_t, states[layer_idx] if states[layer_idx][0] is not None else None)
                states[layer_idx] = state_t
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout(h_t)
                else:
                    x_t = h_t
            all_outputs.append(x_t.unsqueeze(1))  # (batch, 1, hidden)

        outputs = torch.cat(all_outputs, dim=1)    # (batch, seq_len, hidden)
        h_n, c_n = states[-1]
        return outputs, (h_n, c_n)
