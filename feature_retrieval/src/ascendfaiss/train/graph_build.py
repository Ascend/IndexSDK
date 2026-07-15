import numpy as np
from tqdm import tqdm
import os
import argparse
from joblib import Parallel, delayed


# -------------------- 数据读取 --------------------
def read_fvecs(filename):
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        raw = np.fromfile(f, dtype=np.float32)
    vec_size = 1 + dim
    n = len(raw) // vec_size
    return raw[: n * vec_size].reshape(n, vec_size)[:, 1:].copy()


# -------------------- 图质量评估（子集） --------------------
def evaluate_graph(graph, dataset, eval_indices, gt, return_recall=False):
    """
    评估图质量，若 return_recall=True 则返回 recall 字典，否则打印详细信息。
    返回字典格式：{'1': val, '10': val, '32': val, '64': val}
    """
    pred_k = graph.shape[1]
    gt_k = gt.shape[1]
    m = len(eval_indices)

    if m == 0:
        recall_dict = {'1': 0.0, '10': 0.0, '32': 0.0, '64': 0.0}
        if return_recall:
            return recall_dict
        else:
            print("  No evaluation points provided, skipping evaluation.")
            return recall_dict

    check_k_list = [1, 10, 32, 64]  # 计算Recall的常见配置
    recall_dict = {}

    if not return_recall:
        print(f"\n{'=' * 50}")
        print("Graph Quality Evaluation (subset)")
        print(f"{'=' * 50}")
        print(f"  Evaluated points       : {m}")
        print(f"  Predicted graph degree : {pred_k}")
        print(f"  Ground truth degree    : {gt_k}")
        print(f"{'=' * 50}")

    for check_k in check_k_list:
        if check_k > gt_k:
            continue
        total_hits = 0
        total_checks = m * min(check_k, pred_k)
        for idx in range(m):
            i = eval_indices[idx]
            pred_neighbors = set(graph[i].tolist())
            true_neighbors = set(gt[idx, :check_k].tolist())
            total_hits += len(pred_neighbors & true_neighbors)
        recall = total_hits / total_checks
        recall_dict[str(check_k)] = recall
        if not return_recall:
            print(
                f"  Recall@{check_k} : {recall:.4f}  "
                f"(avg {total_hits / m:.1f} / {min(check_k, pred_k):.0f} hits per point)"
            )

    if not return_recall:
        print(f"{'=' * 50}\n")
    return recall_dict


# -------------------- GT 生成（仅 L2） --------------------
def generate_gt_numpy(dataset, num_samples=1000, k=64):
    """
    纯 NumPy 生成 GT（L2 距离），对前 num_samples 个点暴力搜索。
    内存友好：分小批计算距离。
    """
    print(f"    Generating GT (NumPy, L2) for first {num_samples} points (k={k})...")
    queries = dataset[:num_samples]
    n, d = dataset.shape
    gt = np.zeros((num_samples, k), dtype=np.int32)

    q_batch = 100
    for start in range(0, num_samples, q_batch):
        end = min(start + q_batch, num_samples)
        q = queries[start:end]
        q_norm = np.sum(q**2, axis=1, keepdims=True)
        d_norm = np.sum(dataset**2, axis=1)
        dot_batch = 4096
        dists = np.empty((end - start, n), dtype=np.float32)
        for i in range(0, n, dot_batch):
            j = min(i + dot_batch, n)
            dots = np.dot(q, dataset[i:j].T)
            dists[:, i:j] = q_norm + d_norm[i:j] - 2.0 * dots
        topk_idx = np.argpartition(dists, k + 1, axis=1)[:, : k + 1]
        topk_dist = np.take_along_axis(dists, topk_idx, axis=1)
        sorted_order = np.argsort(topk_dist, axis=1)
        sorted_idx = np.take_along_axis(topk_idx, sorted_order, axis=1)
        batch_gt = []
        for j in range(end - start):
            row = sorted_idx[j]
            mask = row != (start + j)
            clean = row[mask][:k]
            if len(clean) < k:
                clean = np.concatenate([clean, [clean[-1]] * (k - len(clean))])
            batch_gt.append(clean)
        gt[start:end] = np.array(batch_gt, dtype=np.int32)
    print(f"    GT generated. Shape: {gt.shape}")
    return gt


# -------------------- CAGRA 索引（固定迭代次数，无早停） --------------------
class CagraIndex:
    def __init__(self):
        self.dataset = None
        self.graph = None
        self.degree = None
        self.metric = 'sqeuclidean'

    @classmethod
    def build(
        cls,
        dataset,
        graph_degree=64,
        intermediate_degree=128,
        nn_descent_niter=10,
        metric='sqeuclidean',
        random_seed=42,
        eval_samples=1000,
    ):
        n, d = dataset.shape
        assert intermediate_degree >= graph_degree
        if intermediate_degree % 32 != 0:
            intermediate_degree = ((intermediate_degree + 31) // 32) * 32
            print(f"Intermediate degree adjusted to {intermediate_degree} (multiple of 32)")

        rng = np.random.RandomState(random_seed)

        # ====== 阶段 1：随机初始化 ======
        print("1. Initializing random intermediate graph (batch-wise, fast)...")
        inter_graph = np.zeros((n, intermediate_degree), dtype=np.int32)
        inter_dist = np.zeros((n, intermediate_degree), dtype=np.float32)
        batch_size = 32768
        for start in tqdm(range(0, n, batch_size), desc="   Init batch"):
            end = min(start + batch_size, n)
            B = end - start
            neigh_local = rng.randint(0, n - 1, size=(B, intermediate_degree), dtype=np.int32)
            offset = np.arange(start, end)[:, None]
            mask = neigh_local >= (offset - start)
            neigh_local[mask] += 1
            neigh = neigh_local
            queries = dataset[start:end]
            neigh_vecs = dataset[neigh]
            diff = queries[:, np.newaxis, :] - neigh_vecs
            dists = np.sum(diff**2, axis=2)
            idx = np.argsort(dists, axis=1)
            inter_graph[start:end] = neigh[np.arange(B)[:, None], idx]
            inter_dist[start:end] = dists[np.arange(B)[:, None], idx]
        print("   Initialization done.\n")

        # ====== 阶段 2：提前生成 GT（只一次） ======
        if eval_samples > 0:
            print("2. Generating ground truth for evaluation...")
            k_gt = max(intermediate_degree, graph_degree, 64)
            gt = generate_gt_numpy(dataset, num_samples=eval_samples, k=k_gt)
            eval_indices = np.arange(eval_samples)
            print("    GT ready.\n")
        else:
            gt = None
            eval_indices = np.array([])
            print("2. Skipping ground truth generation (eval_samples=0).\n")

        # ====== 阶段 3：NN‑Descent 迭代（固定次数） ======
        print(f"3. NN-Descent iterations (fixed {nn_descent_niter} iterations)")
        print("   Suggested stopping criterion: all R@1/10/32/64 >= 0.995 for gt of 1000 samples")
        for it in range(nn_descent_niter):
            print(f"   Iteration {it + 1}/{nn_descent_niter}")

            # 3.1 构建反向图
            print("     Building reverse graph...")
            rev_graph = [[] for _ in range(n)]
            for i in tqdm(range(n), desc="     Reverse graph", leave=True):
                for j in inter_graph[i]:
                    if j < n and len(rev_graph[j]) < intermediate_degree:
                        rev_graph[j].append(i)
            rev_graph_np = np.empty(n, dtype=object)
            for i in range(n):
                rev_graph_np[i] = np.array(rev_graph[i], dtype=np.int32)

            # 3.2 并行更新每个点
            def update_single(i, inter_graph, dataset, intermediate_degree, rev_graph_np, rng):
                n1 = inter_graph[i]
                n2 = inter_graph[n1].ravel()
                n3 = rev_graph_np[i]
                merged = np.concatenate([n1, n2, n3])
                candidates = np.unique(merged)
                candidates = candidates[candidates != i]
                if len(candidates) == 0:
                    candidates = rng.choice(
                        len(dataset), size=min(intermediate_degree + 1, len(dataset)), replace=False
                    )
                    candidates = candidates[candidates != i][:intermediate_degree]
                cand_vecs = dataset[candidates]
                query = dataset[i]
                diff = query.astype(np.float32) - cand_vecs
                dists = np.sum(diff * diff, axis=1)
                k_take = min(intermediate_degree, len(candidates))
                if k_take == 0:
                    new_neigh = np.full(intermediate_degree, i, dtype=np.int32)
                    new_dist_ = np.full(intermediate_degree, 1e10, dtype=np.float32)
                    new_dist_[0] = 0.0
                    return new_neigh, new_dist_
                topk_idx = np.argpartition(dists, k_take - 1)[:k_take]
                topk_dists = dists[topk_idx]
                sorted_order = np.argsort(topk_dists)
                topk_idx = topk_idx[sorted_order]
                topk_dists = topk_dists[sorted_order]
                new_neigh = np.empty(intermediate_degree, dtype=np.int32)
                new_dist_ = np.empty(intermediate_degree, dtype=np.float32)
                new_neigh[:k_take] = candidates[topk_idx]
                new_dist_[:k_take] = topk_dists
                if k_take < intermediate_degree:
                    new_dist_[k_take:] = 1e10
                    new_neigh[k_take:] = new_neigh[0]
                return new_neigh, new_dist_

            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(update_single)(i, inter_graph, dataset, intermediate_degree, rev_graph_np, rng)
                for i in tqdm(range(n), desc="     Updating graph")
            )
            new_graph = np.array([r[0] for r in results], dtype=np.int32)
            new_dist = np.array([r[1] for r in results], dtype=np.float32)
            inter_graph, inter_dist = new_graph, new_dist

            # 3.3 评估并打印当前 recall（仅用于观察，不影响停止）
            if eval_samples > 0:
                recalls = evaluate_graph(inter_graph, dataset, eval_indices, gt, return_recall=True)
                print(
                    f"     Current recalls: "
                    f"R@1={recalls.get('1', 0):.4f} R@10={recalls.get('10', 0):.4f} "
                    f"R@32={recalls.get('32', 0):.4f} R@64={recalls.get('64', 0):.4f}\n"
                )

        # ====== 阶段 4：最终评估中间图（详细打印） ======
        if eval_samples > 0:
            print("\n4. [Evaluation] Intermediate graph (final):")
            evaluate_graph(inter_graph, dataset, eval_indices, gt)

        # ====== 阶段 5：CAGRA 优化（绕路剪枝 + 逆边替换） ======
        print("\n5. CAGRA pruning (detour count based) and reverse edge replacement...")
        final_degree = graph_degree
        # 保护比例提升至 3/4，减少替换带来的精度损失
        num_protected = final_degree * 3 // 4

        # 5.1 广播绕路计数（极速版）
        K = inter_graph.shape[1]
        inter_graph = np.clip(inter_graph, 0, n - 1)

        def compute_detour_broadcast(i):
            neighbors = inter_graph[i]  # (K,)
            neigh_of_neigh = inter_graph[neighbors]  # (K, K)
            found = np.any(neigh_of_neigh[:, :, None] == neighbors[None, None, :], axis=1)  # (K, K)
            upper_tri = np.triu(found, k=1)
            detour = upper_tri.sum(axis=0).astype(np.int32)
            return i, detour

        print("  5.1 Computing detour counts (broadcast, parallel) ...")
        print("      Starting detour count computation ...")
        detour_count = np.zeros((n, K), dtype=np.int32)
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(compute_detour_broadcast)(i) for i in tqdm(range(n), desc="  Detour counts")
        )
        for i, det in results:
            detour_count[i] = det

        # 5.2 按绕路数剪枝
        final_graph = np.zeros((n, final_degree), dtype=np.int32)
        print("  5.2 Selecting edges by detour count ...")

        def select_edges(i):
            order = np.argsort(detour_count[i], kind='stable')
            return inter_graph[i, order[:final_degree]]

        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(select_edges)(i) for i in tqdm(range(n), desc="  Selecting edges")
        )
        for i, edges in enumerate(results):
            final_graph[i] = edges

        # 5.3 构建逆向图（向量化）
        print("  5.3 Building reverse graph ...")
        src_ids = np.repeat(np.arange(n), final_degree)
        dst_ids = final_graph.ravel().astype(np.int64)
        valid = dst_ids < n
        src_ids = src_ids[valid]
        dst_ids = dst_ids[valid]

        order = np.argsort(dst_ids)
        src_ids = src_ids[order]
        dst_ids = dst_ids[order]

        unique_dst, indices = np.unique(dst_ids, return_index=True)
        counts = np.diff(np.append(indices, len(dst_ids)))

        max_rev = final_degree
        rev_graph_flat = np.full(n * max_rev, -1, dtype=np.int32)
        rev_offsets = np.zeros(n, dtype=np.int32)

        for idx in range(len(unique_dst)):
            dst = unique_dst[idx]
            start = indices[idx]
            end = start + counts[idx]
            src_list = src_ids[start:end]
            if len(src_list) > max_rev:
                src_list = src_list[:max_rev]
            rev_graph_flat[dst * max_rev : dst * max_rev + len(src_list)] = src_list
            rev_offsets[dst] = len(src_list)

        # 5.4 并行逆向边替换（修复移位逻辑，且保护 3/4 的边）
        print("  5.4 Replacing edges with reverse edges (parallel, fixed shift) ...")

        def replace_for_node(j):
            neighbors = final_graph[j].copy()
            start = j * max_rev
            end = start + rev_offsets[j]
            rev_list = rev_graph_flat[start:end]
            for i in reversed(rev_list):
                pos = np.where(neighbors == i)[0]
                if len(pos) > 0:
                    pos = pos[0]
                else:
                    pos = final_degree

                if pos < num_protected:
                    continue

                if pos == final_degree:  # 不在列表中，插入
                    neighbors[num_protected + 1 :] = neighbors[num_protected:-1]
                else:  # 在列表中且不在保护区，移至保护区位置
                    neighbors[num_protected + 1 : pos + 1] = neighbors[num_protected:pos]
                neighbors[num_protected] = i
            return j, neighbors

        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(replace_for_node)(j) for j in tqdm(range(n), desc="  Replacing edges")
        )
        for j, new_neigh in results:
            final_graph[j] = new_neigh

        # ====== 阶段 6：评估最终图 ======
        if eval_samples > 0:
            print("\n6. [Evaluation] Final graph (after CAGRA pruning + reverse edges):")
            evaluate_graph(final_graph, dataset, eval_indices, gt)

        index = cls()
        index.dataset = dataset
        index.graph = final_graph
        index.degree = final_degree
        index.metric = metric
        return index

    def save(self, output_dir, query_data, visited_map_size=2560000):
        """
        导出 CAGRA 搜索所需文件（全部为必需文件）。
        query_data 必须传入，与 dataset 格式相同的 np.ndarray (float32)。
        """
        os.makedirs(output_dir, exist_ok=True)

        graph_file = os.path.join(output_dir, "knn_graph.bin")
        self.graph.astype(np.uint32).ravel().tofile(graph_file)

        data_file = os.path.join(output_dir, "data_ptr.bin")
        self.dataset.ravel().tofile(data_file)

        visited_file = os.path.join(output_dir, "visited_map.bin")
        visited = np.zeros(visited_map_size, dtype=np.uint32)
        visited.tofile(visited_file)

        query_bin_path = os.path.join(output_dir, "queries.bin")
        query_data.ravel().tofile(query_bin_path)

        graph_size = os.path.getsize(graph_file) / 1024 / 1024
        data_size = os.path.getsize(data_file) / 1024 / 1024
        visited_size = os.path.getsize(visited_file) / 1024 / 1024
        q_size = os.path.getsize(query_bin_path) / 1024 / 1024
        print(
            f"Saved for CAGRA search:\n"
            f"  {graph_file} ({graph_size:.1f} MB)\n"
            f"  {data_file} ({data_size:.1f} MB)\n"
            f"  {visited_file} ({visited_size:.1f} MB)\n"
            f"  {query_bin_path} ({q_size:.1f} MB)"
        )


# ===================== 命令行参数 =====================
def parse_args():
    parser = argparse.ArgumentParser(description='CAGRA Graph Construction (NumPy, enhanced)')
    parser.add_argument(
        '--input_filepath', type=str, required=True, help='Directory containing sift_base.fvecs and sift_query.fvecs'
    )
    parser.add_argument(
        '--output_filepath',
        type=str,
        required=True,
        help='Output directory for knn_graph.bin, data_ptr.bin, visited_map.bin and queries.bin',
    )
    parser.add_argument(
        '--nn_descent_niter',
        type=int,
        default=10,
        help='Number of NN-Descent iterations (default: 10). '
        'It is recommended to set enough iterations so that all R@1/10/32/64 >= 0.995.',
    )
    parser.add_argument(
        '--eval_samples', type=int, default=1000, help='Number of points for quality evaluation (set to 0 to skip)'
    )
    parser.add_argument('--graph_degree', type=int, default=64, help='Final graph out-degree (default: 64)')
    parser.add_argument(
        '--intermediate_degree',
        type=int,
        default=128,
        help='Intermediate graph out-degree, must be >= graph_degree (default: 128)',
    )
    return parser.parse_args()


# ===================== 主流程 =====================
if __name__ == "__main__":
    args = parse_args()

    input_dir = args.input_filepath
    base_file = os.path.join(input_dir, "sift_base.fvecs")
    query_file = os.path.join(input_dir, "sift_query.fvecs")

    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file not found: {base_file}")
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")

    print("Loading base data...")
    base_dataset = read_fvecs(base_file)

    print("Loading query data...")
    query_dataset = read_fvecs(query_file)

    print("Building CAGRA index (NumPy enhanced)...")
    cagra_index = CagraIndex.build(
        base_dataset,
        graph_degree=args.graph_degree,
        intermediate_degree=args.intermediate_degree,
        nn_descent_niter=args.nn_descent_niter,
        eval_samples=args.eval_samples,
    )

    cagra_index.save(args.output_filepath, query_dataset)
