"""
ADTWBiLSTM implementation for ICL4RUL (NASA C-MAPSS FD001).

Architecture follows Eqs. (4)-(7) and Figs. 1-2 of the supplied paper,
with explicit implementation assumptions documented at the end of this file.
"""

import torch
import torch.nn as nn


class ADTW(nn.Module):
    """
    Adaptive Degradation Temporal Weighting.

    Input:  (B, S, F)
    Output: (B, S, F), sigmoid feature weights in (0, 1)
    """

    def __init__(self, num_features=14, reduction=2, kernel_size=3,
                 debug_shapes=False):
        super().__init__()

        if num_features < 1:
            raise ValueError("num_features must be >= 1.")
        if reduction < 1:
            raise ValueError("reduction must be >= 1.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.num_features = num_features
        self.reduction = reduction
        self.reduced_features = max(num_features // reduction, 1)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.debug_shapes = debug_shapes

        # Pool/Conv1d operate along time (length S); sensors are channels.
        # The two real/imaginary components use the SAME pooling and Conv1d
        # modules, applied separately because these PyTorch layers are real-valued.
        self.max_pool = nn.MaxPool1d(
            kernel_size=kernel_size, stride=1, padding=self.padding
        )
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=self.padding,
            count_include_pad=False
        )

        self.max_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=self.reduced_features,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding
        )
        self.avg_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=self.reduced_features,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding
        )

        # Eq. (6): each reduced branch is projected back to F features.
        self.max_linear = nn.Linear(self.reduced_features, num_features)
        self.avg_linear = nn.Linear(self.reduced_features, num_features)

        # Eq. (7): sigmoid of the sum of the two branch projections.
        self.sigmoid = nn.Sigmoid()

    def _assert_shape(self, tensor, expected, name):
        actual = tuple(tensor.shape)
        assert actual == tuple(expected), (
            f"{name} shape mismatch: expected {tuple(expected)}, got {actual}"
        )
        if self.debug_shapes:
            print(f"{name:28s}: {actual}")

    def _pool_and_convolve_component(self, component, pool, conv, name,
                                     batch_size, sequence_length):
        """
        Apply a real-valued branch to one FFT component.
        Input component: (B, S, F); layers expect (B, F, S).
        """
        component = component.transpose(1, 2).contiguous()
        self._assert_shape(
            component, (batch_size, self.num_features, sequence_length),
            f"{name} input to pool/conv (B,F,S)"
        )

        pooled = pool(component)
        self._assert_shape(
            pooled, (batch_size, self.num_features, sequence_length),
            f"{name} pooled"
        )

        convolved = conv(pooled)
        self._assert_shape(
            convolved,
            (batch_size, self.reduced_features, sequence_length),
            f"{name} convolved (B,F',S)"
        )

        # Convert to (B, S, F') so IFFT acts on the original time axis.
        return convolved.transpose(1, 2).contiguous()

    def _process_branch(self, x_fft, pool, conv, name, batch_size,
                        sequence_length):
        # Implementation choice required by the prompt: process the real and
        # imaginary FFT parts separately with shared layer weights, then recombine.
        real_processed = self._pool_and_convolve_component(
            x_fft.real, pool, conv, f"{name} real",
            batch_size, sequence_length
        )
        imag_processed = self._pool_and_convolve_component(
            x_fft.imag, pool, conv, f"{name} imag",
            batch_size, sequence_length
        )

        x_complex = torch.complex(real_processed, imag_processed)
        self._assert_shape(
            x_complex, (batch_size, sequence_length, self.reduced_features),
            f"{name} recombined complex"
        )

        # Eq. (4): inverse FFT along time; retain its real part.
        x_time = torch.fft.ifft(x_complex, dim=1).real
        self._assert_shape(
            x_time, (batch_size, sequence_length, self.reduced_features),
            f"{name} IFFT real"
        )
        return x_time

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"ADTW expects (B,S,F), received shape {tuple(x.shape)}"
            )

        batch_size, sequence_length, num_features = x.shape
        if num_features != self.num_features:
            raise ValueError(
                f"ADTW was configured for F={self.num_features}, "
                f"but received F={num_features}."
            )

        # Eq. (4): FFT along time dimension S.
        x_fft = torch.fft.fft(x, dim=1)
        self._assert_shape(
            x_fft, (batch_size, sequence_length, self.num_features),
            "FFT (B,S,F)"
        )

        # Eq. (5), max branch: FFT -> MaxPool1d -> Conv1d -> IFFT.
        x1 = self._process_branch(
            x_fft, self.max_pool, self.max_conv, "Max branch",
            batch_size, sequence_length
        )

        # Eq. (5), average branch: FFT -> AvgPool1d -> Conv1d -> IFFT.
        x2 = self._process_branch(
            x_fft, self.avg_pool, self.avg_conv, "Avg branch",
            batch_size, sequence_length
        )

        # Eq. (6): linear projection from F' back to F in each branch.
        y1 = self.max_linear(x1)
        self._assert_shape(
            y1, (batch_size, sequence_length, self.num_features),
            "Y1 linear"
        )

        y2 = self.avg_linear(x2)
        self._assert_shape(
            y2, (batch_size, sequence_length, self.num_features),
            "Y2 linear"
        )

        # Eq. (7): combine branch projections and produce weights.
        weights = self.sigmoid(y1 + y2)
        self._assert_shape(
            weights, (batch_size, sequence_length, self.num_features),
            "ADTW sigmoid weights"
        )
        assert torch.all(weights > 0).item(), "Sigmoid weights must be > 0."
        assert torch.all(weights < 1).item(), "Sigmoid weights must be < 1."

        return weights


class ADTWBiLSTM(nn.Module):
    """
    ADTW-weighted BiLSTM block.

    Input:  (B, S, 14) for the FD001 setup
    Output: (sequence_output (B, S, 90), ADTW weights (B, S, 14))
    """

    def __init__(self, input_size=14, hidden_size=45, num_layers=2,
                 reduction=2, kernel_size=3, debug_shapes=False):
        super().__init__()

        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size, and num_layers must be >= 1.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.debug_shapes = debug_shapes

        self.adtw = ADTW(
            num_features=input_size,
            reduction=reduction,
            kernel_size=kernel_size,
            debug_shapes=debug_shapes
        )

        # Paper's two-layer, bidirectional LSTM; preserve all S time steps.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"ADTWBiLSTM expects (B,S,F), received {tuple(x.shape)}"
            )

        batch_size, sequence_length, num_features = x.shape
        if num_features != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {num_features}."
            )

        # Step 1: Eq. (7) generates adaptive weights.
        weights = self.adtw(x)
        assert weights.shape == x.shape, (
            f"ADTW weights {tuple(weights.shape)} must match input {tuple(x.shape)}"
        )
        if self.debug_shapes:
            print(f"{'ADTW weights':28s}: {tuple(weights.shape)}")

        # Step 2: apply the learned weights element-wise to the input.
        x_weighted = x * weights
        self._assert_shape(
            x_weighted, (batch_size, sequence_length, self.input_size),
            "Weighted input X*weights"
        )

        # Step 3: BiLSTM returns the full sequence, not only the final step.
        sequence_output, (h_n, c_n) = self.lstm(x_weighted)
        expected_output = (batch_size, sequence_length, 2 * self.hidden_size)
        self._assert_shape(sequence_output, expected_output, "LSTM full output")
        assert h_n.shape == (
            2 * self.num_layers, batch_size, self.hidden_size
        ), f"Unexpected LSTM h_n shape: {tuple(h_n.shape)}"
        assert c_n.shape == (
            2 * self.num_layers, batch_size, self.hidden_size
        ), f"Unexpected LSTM c_n shape: {tuple(c_n.shape)}"

        return sequence_output, weights

    def _assert_shape(self, tensor, expected, name):
        actual = tuple(tensor.shape)
        assert actual == tuple(expected), (
            f"{name} shape mismatch: expected {tuple(expected)}, got {actual}"
        )
        if self.debug_shapes:
            print(f"{name:28s}: {actual}")


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 32
    sequence_length = 40
    num_features = 14
    expected_reduced_features = max(num_features // 2, 1)

    model = ADTWBiLSTM(
        input_size=num_features,
        hidden_size=45,
        num_layers=2,
        reduction=2,
        kernel_size=3,
        debug_shapes=True
    )

    # Dummy input only; no dataset loading or training loop.
    x = torch.randn(batch_size, sequence_length, num_features)
    print("Input                       :", tuple(x.shape))
    model.zero_grad(set_to_none=True)

    sequence_output, weights = model(x)

    # Required output contracts.
    assert tuple(sequence_output.shape) == (32, 40, 90), (
        f"Expected output (32,40,90), got {tuple(sequence_output.shape)}"
    )
    assert tuple(weights.shape) == (32, 40, 14), (
        f"Expected weights (32,40,14), got {tuple(weights.shape)}"
    )
    assert weights.min().item() > 0.0, "Minimum ADTW weight must be > 0."
    assert weights.max().item() < 1.0, "Maximum ADTW weight must be < 1."
    assert torch.isfinite(sequence_output).all().item()
    assert torch.isfinite(weights).all().item()

    print("Final sequence output         :", tuple(sequence_output.shape))
    print("Final ADTW weights             :", tuple(weights.shape))
    print("ADTW weights min               :", weights.min().item())
    print("ADTW weights max               :", weights.max().item())

    # Gradient check: all ADTW parameter tensors must receive finite,
    # nonzero gradients from a backward pass on output.sum().
    sequence_output.sum().backward()
    adtw_parameters = list(model.adtw.named_parameters())

    missing_gradients = [
        name for name, param in adtw_parameters if param.grad is None
    ]
    zero_gradients = [
        name for name, param in adtw_parameters
        if param.grad is not None and torch.count_nonzero(param.grad).item() == 0
    ]
    nonfinite_gradients = [
        name for name, param in adtw_parameters
        if param.grad is not None and not torch.isfinite(param.grad).all().item()
    ]

    print("ADTW parameter tensors checked :", len(adtw_parameters))
    print("Parameters missing gradients   :", missing_gradients)
    print("Parameters with all-zero grads  :", zero_gradients)
    print("Parameters with nonfinite grads :", nonfinite_gradients)

    assert not missing_gradients, f"Missing ADTW gradients: {missing_gradients}"
    assert not zero_gradients, f"All-zero ADTW gradients: {zero_gradients}"
    assert not nonfinite_gradients, (
        f"Nonfinite ADTW gradients: {nonfinite_gradients}"
    )

    print("Forward shape checks passed.")
    print("ADTW gradient check passed.")
    print("All tests passed.")


"""
ASSUMPTIONS
1. F' = max(F // reduction, 1); default reduction=2.
2. Pooling and Conv1d operate along time (S). Sensor features are channels;
   Conv1d out_channels performs the F -> F' bottleneck.
3. Same padding (kernel_size // 2), stride 1 preserves S for odd kernels.
4. Real and imaginary FFT components are processed separately using the
   same branch layers, then recombined with torch.complex before IFFT.
5. AvgPool1d excludes padded values from its average.
6. Each branch has its own Conv1d and Linear layer, consistent with the
   paper's two branch-specific transformations.
7. The explicit element-wise input weighting X * ADTW(X) follows the task
   prompt's interpretation of the paper's feature weights.
"""
