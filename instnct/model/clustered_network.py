"""
Clustered VRAXION: partitioned spiking network across CPU cores.

Architecture:
  - N_CLUSTERS compute clusters, each with own neurons + internal edges
  - 1 IO layer managing inter-cluster routing edges
  - Each cluster evolves independently (local mutations)
  - IO layer evolves the routing between clusters
  - Forward pass is global (all clusters + IO)
  - Crystal is parallel (each cluster + IO independently)

Example: H=8192, 12 cores:
  Core 0-10:  11 clusters x 745 neurons each
  Core 11:    IO layer managing ~5K inter-cluster edges
"""

import numpy as np
import random
import os, sys

_model_dir = os.path.dirname(os.path.abspath(__file__))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)
from quaternary_mask import QuaternaryMask


class Cluster:
    """A local group of neurons with internal edges."""

    def __init__(self, neuron_ids, H_total):
        self.neuron_ids = np.array(neuron_ids, dtype=np.int32)
        self.N = len(neuron_ids)
        self.H_total = H_total
        # Local-to-global and global-to-local mapping
        self.local_to_global = self.neuron_ids.copy()
        self.global_to_local = np.full(H_total, -1, dtype=np.int32)
        for loc, glob in enumerate(self.neuron_ids):
            self.global_to_local[glob] = loc
        # Per-neuron params (local copies)
        self.theta = np.full(self.N, 1.0, dtype=np.float32)
        self.channel = np.ones(self.N, dtype=np.uint8)
        self.polarity = np.ones(self.N, dtype=np.float32)

    def init_params(self, theta_global, channel_global, polarity_global):
        """Copy global params into local."""
        self.theta = theta_global[self.neuron_ids].copy()
        self.channel = channel_global[self.neuron_ids].copy()
        self.polarity = polarity_global[self.neuron_ids].copy()

    def write_params_to_global(self, theta_global, channel_global, polarity_global):
        """Write local params back to global arrays."""
        theta_global[self.neuron_ids] = self.theta
        channel_global[self.neuron_ids] = self.channel
        polarity_global[self.neuron_ids] = self.polarity


class ClusteredNetwork:
    """Partitioned spiking network: clusters + IO layer."""

    def __init__(self, H, n_clusters=11, density=0.01, io_density=0.005, seed=42):
        self.H = H
        self.n_clusters = n_clusters
        self.seed = seed

        # Partition neurons into clusters (roughly equal)
        rng = np.random.RandomState(seed)
        neuron_order = rng.permutation(H)
        cluster_size = H // n_clusters
        remainder = H % n_clusters

        self.clusters = []
        offset = 0
        for i in range(n_clusters):
            size = cluster_size + (1 if i < remainder else 0)
            ids = neuron_order[offset:offset + size]
            self.clusters.append(Cluster(ids, H))
            offset += size

        # Internal edges per cluster (intra-cluster)
        self.cluster_masks = []
        for cl in self.clusters:
            N = cl.N
            # Small local quaternary mask for intra-cluster edges
            local_mask = (rng.rand(N, N) < density).astype(bool)
            np.fill_diagonal(local_mask, False)
            qm = QuaternaryMask.from_bool_mask(local_mask)
            self.cluster_masks.append(qm)

        # IO layer: inter-cluster edges
        # For each cluster pair (i, j) where i != j, allow some edges
        io_edges = []
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    continue
                # Random edges from cluster i neurons to cluster j neurons
                n_io = max(1, int(len(self.clusters[i].neuron_ids) *
                                   len(self.clusters[j].neuron_ids) * io_density))
                for _ in range(n_io):
                    src = int(self.clusters[i].neuron_ids[rng.randint(0, len(self.clusters[i].neuron_ids) - 1)])
                    tgt = int(self.clusters[j].neuron_ids[rng.randint(0, len(self.clusters[j].neuron_ids) - 1)])
                    if src != tgt:
                        io_edges.append((src, tgt))
        self.io_edges = list(set(io_edges))  # deduplicate

        # Global params
        self.theta = np.full(H, 1.0, dtype=np.float32)
        self.channel = rng.randint(1, 9, size=H).astype(np.uint8)
        ref_polarity = np.ones(H, dtype=bool)
        ref_polarity[rng.random(H) < 0.10] = False  # 10% inhibitory
        self.polarity_f32 = np.where(ref_polarity, 1.0, -1.0).astype(np.float32)

        # Init cluster local params
        for cl in self.clusters:
            cl.init_params(self.theta, self.channel, self.polarity_f32)

    def get_global_edges(self):
        """Assemble all cluster + IO edges into global (rows, cols) arrays."""
        all_rows = []
        all_cols = []

        # Intra-cluster edges (local indices -> global)
        for cl, qm in zip(self.clusters, self.cluster_masks):
            local_rows, local_cols = qm.to_directed_edges()
            if len(local_rows) > 0:
                global_rows = cl.local_to_global[local_rows]
                global_cols = cl.local_to_global[local_cols]
                all_rows.append(global_rows)
                all_cols.append(global_cols)

        # IO edges
        if self.io_edges:
            io_r = np.array([e[0] for e in self.io_edges], dtype=np.intp)
            io_c = np.array([e[1] for e in self.io_edges], dtype=np.intp)
            all_rows.append(io_r)
            all_cols.append(io_c)

        if all_rows:
            rows = np.concatenate(all_rows).astype(np.intp)
            cols = np.concatenate(all_cols).astype(np.intp)
        else:
            rows = np.empty(0, dtype=np.intp)
            cols = np.empty(0, dtype=np.intp)
        return rows, cols

    def count_edges(self):
        total = sum(qm.count_edges() for qm in self.cluster_masks)
        total += len(self.io_edges)
        return total

    def mutate_cluster(self, cluster_idx, op, rng):
        """Mutate one cluster's internal edges or params.
        Returns undo log."""
        cl = self.clusters[cluster_idx]
        qm = self.cluster_masks[cluster_idx]
        undo = []

        if op == 'add':
            qm.mutate_add(rng, undo)
        elif op == 'remove':
            qm.mutate_remove(rng, undo)
        elif op == 'reverse':
            qm.mutate_flip(rng, undo)
        elif op == 'mirror':
            qm.mutate_upgrade(rng, undo)
        elif op == 'loop3':
            # Local loop within cluster
            N = cl.N
            if N >= 3:
                nodes = [rng.randint(0, N - 1)]
                for _ in range(2):
                    n = rng.randint(0, N - 1)
                    if n in nodes:
                        return undo
                    nodes.append(n)
                for k in range(3):
                    r, c = nodes[k], nodes[(k + 1) % 3]
                    if qm.get_pair(r, c) != 0:
                        return undo
                for k in range(3):
                    r, c = nodes[k], nodes[(k + 1) % 3]
                    qm.set_pair(r, c, 1)
                    undo.append(('QA', qm._pair_index(r, c), 0))
        elif op == 'theta':
            idx = rng.randint(0, cl.N - 1)
            old = float(cl.theta[idx])
            cl.theta[idx] = float(rng.randint(1, 15))
            self.theta[cl.local_to_global[idx]] = cl.theta[idx]
            undo.append(('CT', cluster_idx, idx, old))
        elif op == 'channel':
            idx = rng.randint(0, cl.N - 1)
            old = int(cl.channel[idx])
            cl.channel[idx] = np.uint8(rng.randint(1, 8))
            self.channel[cl.local_to_global[idx]] = cl.channel[idx]
            undo.append(('CC', cluster_idx, idx, old))
        elif op == 'flip':
            idx = rng.randint(0, cl.N - 1)
            old = float(cl.polarity[idx])
            cl.polarity[idx] *= -1
            self.polarity_f32[cl.local_to_global[idx]] = cl.polarity[idx]
            undo.append(('CF', cluster_idx, idx, old))
        return undo

    def mutate_io(self, op, rng):
        """Mutate the IO layer (inter-cluster edges). Returns undo log."""
        undo = []
        if op == 'add':
            # Add random inter-cluster edge
            ci = rng.randint(0, self.n_clusters - 1)
            cj = rng.randint(0, self.n_clusters - 1)
            if ci == cj:
                return undo
            src = int(self.clusters[ci].neuron_ids[rng.randint(0, len(self.clusters[ci].neuron_ids) - 1)])
            tgt = int(self.clusters[cj].neuron_ids[rng.randint(0, len(self.clusters[cj].neuron_ids) - 1)])
            if src != tgt and (src, tgt) not in set(self.io_edges):
                self.io_edges.append((src, tgt))
                undo.append(('IA', src, tgt))
        elif op == 'remove':
            if self.io_edges:
                idx = rng.randint(0, len(self.io_edges) - 1)
                edge = self.io_edges[idx]
                self.io_edges[idx] = self.io_edges[-1]
                self.io_edges.pop()
                undo.append(('IR', edge[0], edge[1]))
        elif op == 'reverse':
            if self.io_edges:
                idx = rng.randint(0, len(self.io_edges) - 1)
                src, tgt = self.io_edges[idx]
                if (tgt, src) not in set(self.io_edges):
                    self.io_edges[idx] = (tgt, src)
                    undo.append(('IV', src, tgt))
        elif op == 'rewire':
            if self.io_edges:
                idx = rng.randint(0, len(self.io_edges) - 1)
                src, _ = self.io_edges[idx]
                # Find which cluster src belongs to
                src_cl = None
                for i, cl in enumerate(self.clusters):
                    if src in cl.neuron_ids:
                        src_cl = i; break
                # Pick new target from different cluster
                cj = rng.randint(0, self.n_clusters - 1)
                if cj == src_cl:
                    return undo
                new_tgt = int(self.clusters[cj].neuron_ids[rng.randint(0, len(self.clusters[cj].neuron_ids) - 1)])
                if src != new_tgt and (src, new_tgt) not in set(self.io_edges):
                    old_edge = self.io_edges[idx]
                    self.io_edges[idx] = (src, new_tgt)
                    undo.append(('IW', old_edge[0], old_edge[1], src, new_tgt))
        return undo

    def undo_cluster(self, cluster_idx, undo_log):
        """Undo cluster mutations."""
        qm = self.cluster_masks[cluster_idx]
        cl = self.clusters[cluster_idx]
        for entry in reversed(undo_log):
            op = entry[0]
            if op in ('QA', 'QR', 'QF', 'QU', 'QD'):
                _, idx, old_val = entry
                qm.data[idx] = old_val
                if old_val == 0:
                    qm._remove_alive(idx)
                else:
                    qm._add_alive(idx)
            elif op == 'QW':
                _, idx_old, old_val, idx_new = entry
                qm.data[idx_new] = 0; qm._remove_alive(idx_new)
                qm.data[idx_old] = old_val; qm._add_alive(idx_old)
            elif op == 'CT':
                _, ci, idx, old = entry
                cl.theta[idx] = old
                self.theta[cl.local_to_global[idx]] = old
            elif op == 'CC':
                _, ci, idx, old = entry
                cl.channel[idx] = np.uint8(old)
                self.channel[cl.local_to_global[idx]] = np.uint8(old)
            elif op == 'CF':
                _, ci, idx, old = entry
                cl.polarity[idx] = old
                self.polarity_f32[cl.local_to_global[idx]] = old

    def undo_io(self, undo_log):
        """Undo IO mutations."""
        for entry in reversed(undo_log):
            op = entry[0]
            if op == 'IA':
                _, src, tgt = entry
                try:
                    self.io_edges.remove((src, tgt))
                except ValueError:
                    pass
            elif op == 'IR':
                _, src, tgt = entry
                self.io_edges.append((src, tgt))
            elif op == 'IV':
                _, src, tgt = entry
                try:
                    idx = self.io_edges.index((tgt, src))
                    self.io_edges[idx] = (src, tgt)
                except ValueError:
                    pass
            elif op == 'IW':
                _, old_src, old_tgt, new_src, new_tgt = entry
                try:
                    idx = self.io_edges.index((new_src, new_tgt))
                    self.io_edges[idx] = (old_src, old_tgt)
                except ValueError:
                    pass

    def save(self, path):
        """Save full network state."""
        cluster_data = {}
        for i, (cl, qm) in enumerate(zip(self.clusters, self.cluster_masks)):
            cluster_data[f'cl{i}_neurons'] = cl.neuron_ids
            cluster_data[f'cl{i}_qdata'] = qm.data
            cluster_data[f'cl{i}_theta'] = cl.theta
            cluster_data[f'cl{i}_channel'] = cl.channel
            cluster_data[f'cl{i}_polarity'] = cl.polarity
        io_arr = np.array(self.io_edges, dtype=np.int32) if self.io_edges else np.empty((0, 2), dtype=np.int32)
        np.savez_compressed(path,
                            H=self.H, n_clusters=self.n_clusters,
                            theta=self.theta, channel=self.channel,
                            polarity_f32=self.polarity_f32,
                            io_edges=io_arr,
                            **cluster_data)

    def summary(self):
        intra = sum(qm.count_edges() for qm in self.cluster_masks)
        io = len(self.io_edges)
        sizes = [cl.N for cl in self.clusters]
        return (f"ClusteredNetwork H={self.H}, {self.n_clusters} clusters "
                f"(sizes {min(sizes)}-{max(sizes)}), "
                f"intra={intra} + io={io} = {intra+io} edges")
