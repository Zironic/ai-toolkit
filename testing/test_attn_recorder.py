import torch
from toolkit.attn_recorder import RecordingAttnProcessor


class FakeAttnModule:
    def __init__(self, batch_size=1, heads=2):
        self.batch_size = batch_size
        self.heads = heads
        self.to_q = lambda x: x
        self.to_k = lambda x: x
        # to_v will return a tensor whose second dim represents 'src' positions
        self.to_v = lambda x: x
        # attention scores will be computed based on q/k shapes so tests don't hardcode tgt/src
        def get_att(q, k, a):
            BtimesH = q.shape[0]
            T = q.shape[1]
            S = k.shape[1]
            return torch.ones((BtimesH, T, S)) / float(S)
        self.get_attention_scores = get_att

        def head_to_batch_dim(x):
            # replicate along batch dim for heads
            B = x.shape[0]
            if B == 1:
                return x.repeat(self.heads, 1, 1)
            return x.repeat_interleave(self.heads, dim=0)

        def batch_to_head_dim(x):
            # x: [B*H, T, D] -> [B, T, H*D]
            if x.ndim != 3:
                return x
            BH, T, D = x.shape
            B = self.batch_size
            H = self.heads
            if BH != B * H:
                # best-effort fallback
                return x
            return x.view(B, H, T, D).permute(0, 2, 1, 3).reshape(B, T, H * D)

        # to_out[0] should project last-dim back to channel size (we assume channel==3 for tests)
        self.head_to_batch_dim = head_to_batch_dim
        self.batch_to_head_dim = batch_to_head_dim
        self.to_out = [lambda x: x[..., :3], lambda x: x]
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.spatial_norm = None
        self.group_norm = None


def test_recorder_records_shapes():
    mod = FakeAttnModule()
    rec = RecordingAttnProcessor()
    x = torch.randn((1, 3, 8, 8))
    out = rec(mod, x)
    records = rec.get_records()
    assert len(records) == 1
    attn = records[0]["attn"]
    assert attn.ndim == 4
    B, H, T, S = attn.shape
    assert H == mod.heads


def test_recorder_export_numpy(tmp_path):
    mod = FakeAttnModule()
    rec = RecordingAttnProcessor()
    x = torch.randn((1, 3, 8, 8))
    _ = rec(mod, x)
    out_path = tmp_path / "attn.npz"
    rec.export_numpy(str(out_path))
    assert out_path.exists()
