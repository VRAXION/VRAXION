"""Diagnose dotprod gate alpha values from a checkpoint.

Loads the latest checkpoint, runs a forward pass on random data,
and hooks into the gate to capture alpha values per expert per timestep.
"""
import sys, os, torch, torch.nn.functional as F, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.instnct import INSTNCT
import yaml

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = os.path.join(os.path.dirname(__file__), '..', 'training_output', 'ckpt_latest.pt')
CFG = os.path.join(os.path.dirname(__file__), '..', 'config', 'vraxion_config.yaml')


def main():
    with open(CFG, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    config = raw.get('model', raw)  # nested under 'model:' key

    # load checkpoint
    sd = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    step = sd.get('step', '?')
    print(f"Checkpoint step: {step}")

    # build model
    model = INSTNCT(
        M=config['M'],
        hidden_dim=config.get('hidden_dim', config.get('D', 132)),
        slot_dim=config.get('slot_dim', 64),
        N=config['N'],
        R=config.get('R', 1),
        B=8,
        embed_mode=True,
        kernel_mode=config.get('kernel_mode', 'vshape'),
        checkpoint_chunks=0,
        expert_weighting=False,
        embed_encoding=config.get('embed_encoding', 'learned'),
        output_encoding=config.get('output_encoding', 'learned'),
        pointer_mode=config.get('pointer_mode', 'sequential'),
    ).to(DEVICE)

    model.load_state_dict(sd['model_state'], strict=False)
    model.eval()

    # Collect alpha values by monkey-patching the gate
    alpha_log = []  # will collect (expert_idx, alpha_tensor) tuples

    orig_process = model._process_chunk

    def patched_process(x_chunk, ring_tns, ptr_tns, hidden_tns,
                        S_flt, probs_lst, offsets_long, expert_weights):
        # We need to intercept alpha computation
        # Store S_flt so we know if dotprod mode
        patched_process._S_flt = S_flt
        return orig_process(x_chunk, ring_tns, ptr_tns, hidden_tns,
                           S_flt, probs_lst, offsets_long, expert_weights)

    # Instead of monkey-patching (complex), let's compute alpha manually
    # by doing a forward pass and capturing intermediate values

    B_size = 16
    T = 64
    torch.manual_seed(42)
    x = torch.randint(0, 256, (B_size, T), device=DEVICE)

    # We'll hook into the model to capture alpha
    # The alpha is computed as: sigmoid(input_vec · ring_signal / sqrt(dim))
    # We need input_vec and ring_signal at each step

    # Simpler approach: modify the model temporarily to store alphas
    alphas_per_expert = {i: [] for i in range(model.N)}

    # Save original _process_chunk and create instrumented version
    import types

    original_code = model._process_chunk.__func__

    def instrumented_process(self, x_chunk, ring_tns, ptr_tns, hidden_tns,
                             S_flt, probs_lst, offsets_long, expert_weights):
        M, slot_dim, N = self.M, self.slot_dim, self.N
        ptr_tns = ptr_tns.clone()
        hidden_lst = [hidden_tns[i] for i in range(N)]
        C = x_chunk.shape[1]
        outs_lst = []

        if self._bitlift:
            _bit_shifts = torch.arange(7, -1, -1, device=x_chunk.device)

        for t_idx in range(C):
            if self.embed_mode:
                input_vec_tns = self.input_proj(x_chunk[:, t_idx])
            elif self._bitlift:
                byte_tns = x_chunk[:, t_idx, :]
                bits_tns = ((byte_tns.unsqueeze(-1) >> _bit_shifts) & 1).float()
                input_vec_tns = self.input_proj(bits_tns.reshape(x_chunk.shape[0], -1))
            else:
                input_vec_tns = self.input_proj(x_chunk[:, t_idx])

            for i in range(N):
                center = ptr_tns[i].long().clamp(0, M - 1)
                indices_tns = (center.unsqueeze(1) + offsets_long) % M

                if expert_weights is not None:
                    weights_tns = expert_weights[i].unsqueeze(0).expand(x_chunk.shape[0], -1)
                    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    read_vec_tns = (weights_tns.unsqueeze(-1)
                                    * ring_tns.gather(1, expanded_idx_tns)).sum(1)
                else:
                    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    neighbors_tns = ring_tns.gather(1, expanded_idx_tns)
                    q = self.query_proj[i](hidden_lst[i])
                    scores = (q.unsqueeze(1) * neighbors_tns).sum(-1)
                    scores = scores * (slot_dim ** -0.5)
                    weights_tns = F.softmax(scores, dim=-1)
                    read_vec_tns = (weights_tns.unsqueeze(-1) * neighbors_tns).sum(1)

                theta_tns = (ptr_tns[i] / M) * (2 * math.pi)
                phase_tns = (torch.cos(theta_tns).unsqueeze(-1) * self.phase_cos
                           + torch.sin(theta_tns).unsqueeze(-1) * self.phase_sin)

                ring_signal = self.read_proj[i](read_vec_tns)

                if S_flt == 'dotprod':
                    alpha = torch.sigmoid(
                        (input_vec_tns * ring_signal).sum(-1, keepdim=True)
                        / math.sqrt(ring_signal.shape[-1])
                    )
                    # CAPTURE ALPHA HERE
                    alphas_per_expert[i].append(alpha.detach().cpu())
                    blended_ring = alpha * ring_signal
                else:
                    blended_ring = S_flt * ring_signal

                from model.instnct import _c19_activation, _rho_from_raw, _C_from_raw
                hidden_lst[i] = _c19_activation(
                    input_vec_tns + blended_ring + phase_tns + hidden_lst[i],
                    rho=_rho_from_raw(self.c19_rho_hidden),
                    C=_C_from_raw(self.c19_C_hidden),
                )

                if self.write_proj is not None:
                    write_vec = self.write_proj[i](hidden_lst[i])
                else:
                    write_vec = hidden_lst[i]
                if self._expert_conf is not None:
                    write_vec = self._expert_conf[i].item() * write_vec
                from model.instnct import func_softwrit_tns
                ring_tns = func_softwrit_tns(ring_tns, write_vec, expanded_idx_tns, weights_tns)

                p = probs_lst[min(i, len(probs_lst) - 1)]
                if hasattr(self, 'ptr_dir_head'):
                    dir_logits = self.ptr_dir_head[i](hidden_lst[i])
                    dir_probs = F.softmax(dir_logits, dim=-1)
                    mag_raw = self.ptr_mag_head[i](hidden_lst[i]).squeeze(-1)
                    mag = 1 + (self._ptr_Rmax - 1) * torch.sigmoid(mag_raw)
                    stay = dir_probs[:, 0]
                    fwd = dir_probs[:, 1]
                    step_val = mag * (fwd - dir_probs[:, 2])
                    ptr_tns[i] = (ptr_tns[i] + step_val) % M
                else:
                    ptr_tns[i] = (ptr_tns[i] + 1) % M

            mean_hidden = torch.stack(hidden_lst).mean(0)
            logits = self.output_proj(mean_hidden)
            outs_lst.append(logits)

        hidden_tns = torch.stack(hidden_lst)
        return ring_tns, ptr_tns, hidden_tns, torch.stack(outs_lst, dim=1)

    # Bind the instrumented method
    model._process_chunk = types.MethodType(instrumented_process, model)

    # Run forward
    with torch.no_grad():
        logits, _ = model(x)

    # Analyze alphas
    print(f"\nDevice: {DEVICE}")
    print(f"Input: B={B_size}, T={T}")
    print(f"Experts: N={model.N}")
    print("=" * 60)

    for i in range(model.N):
        all_alpha = torch.cat(alphas_per_expert[i], dim=0).squeeze()  # (T*B,)
        print(f"\nExpert {i}:")
        print(f"  mean  = {all_alpha.mean():.4f}")
        print(f"  std   = {all_alpha.std():.4f}")
        print(f"  min   = {all_alpha.min():.4f}")
        print(f"  max   = {all_alpha.max():.4f}")
        print(f"  <0.1  = {(all_alpha < 0.1).float().mean():.1%}  (gate nearly closed)")
        print(f"  >0.9  = {(all_alpha > 0.9).float().mean():.1%}  (gate wide open)")
        print(f"  0.3-0.7 = {((all_alpha > 0.3) & (all_alpha < 0.7)).float().mean():.1%}  (uncertain)")

        # histogram buckets
        buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts = []
        for lo, hi in zip(buckets[:-1], buckets[1:]):
            c = ((all_alpha >= lo) & (all_alpha < hi)).sum().item()
            counts.append(c)
        total = sum(counts)
        print(f"  histogram:")
        for (lo, hi), c in zip(zip(buckets[:-1], buckets[1:]), counts):
            bar = '#' * int(40 * c / max(total, 1))
            print(f"    [{lo:.1f}-{hi:.1f}) {c:5d} ({100*c/max(total,1):5.1f}%) {bar}")


if __name__ == '__main__':
    main()
