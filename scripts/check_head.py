"""Q6_K-exact head GEMV vs exact-int8 head GEMV vs float64 reference, same GGUF weights and x."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_fused as F  # noqa: E402
import qwen_gguf as GG  # noqa: E402
from lab.gguf import GGUF, q6_k_blocks  # noqa: E402


def main():
    torch.zeros(1, device="cuda")
    F.check(F.cu.cuInit(0))
    F.check(F.cu.cuCtxSetCurrent(F.check(F.cu.cuDevicePrimaryCtxRetain(F.check(F.cu.cuDeviceGet(0))))))
    sys.argv.append("--q6")  # default path builds the q6 head
    m = GG.GGUFQwen(None)
    x = (torch.randn(m.H, device="cuda") * 0.5).to(torch.bfloat16)
    y6 = torch.zeros(m.V, device="cuda")
    GG._gemv(m, "head", m.head, x, y6, 0, m.V, m.H)
    torch.cuda.synchronize()
    b, t = GGUF(GG.GGUF_PATH).raw("output.weight")
    q6, s6 = q6_k_blocks(b)
    K, N = t.dims
    Wf = (q6.reshape(N, K).astype(np.float64) * np.repeat(s6.reshape(N, K // 16).astype(np.float64), 16, axis=1))
    ref = Wf @ x.float().cpu().double().numpy()
    yq = y6.cpu().double().numpy()
    err = np.max(np.abs(yq - ref)) / np.max(np.abs(ref))
    print(f"q6 head vs float64 reference: normalized max err {err:.2e}; argmax q6 {yq.argmax()} ref {ref.argmax()}")
    sys.exit(0 if err < 1e-4 else 1)


if __name__ == "__main__":
    main()
