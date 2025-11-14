import argparse
import random
import torch

import tritonblas


def pretty_print_grid(grid_1d: torch.Tensor, num_pid_m: int, num_pid_n: int):
    grid_cpu = grid_1d.cpu().numpy().reshape(num_pid_m, num_pid_n)
    max_width = len(str(int(grid_cpu.max()))) if grid_cpu.size else 1
    sep = " | "
    width = max_width
    print(f"\nGrid ({num_pid_m} x {num_pid_n}):")
    print("=" * (width * num_pid_n + len(sep) * (num_pid_n - 1)))
    for row in grid_cpu:
        entries = sep.join(f"{int(val):{width}d}" for val in row)
        print(entries)
    print("=" * (width * num_pid_n + len(sep) * (num_pid_n - 1)))


def _assert_bijection(out: torch.Tensor, trial_idx: int, a_val: int, c_val: int):
    n = out.numel()
    expected_vals = torch.arange(n, device=out.device, dtype=out.dtype)
    if not torch.equal(torch.sort(out).values, expected_vals):
        bc = torch.bincount(out, minlength=n)
        missing = (bc == 0).nonzero(as_tuple=False).squeeze(1).cpu().tolist()
        dups_idx = (bc > 1).nonzero(as_tuple=False).squeeze(1)
        duplicates = {int(i.item()): int(bc[int(i)].item()) for i in dups_idx}
        raise AssertionError(
            f"Trial {trial_idx}: invalid shuffle (not bijection). "
            f"a={a_val}, c={c_val}, n={n}. Missing: {missing}; Duplicates: {duplicates}"
        )


def cpu_lcg_shuffle(n: int, a: int, c: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.int64)
    transformed = (a * idx + c) % n
    out = torch.empty(n, dtype=torch.int64)
    out[transformed] = idx
    return out


def summarize_counts(counts: torch.Tensor, num_trials: int, num_pid_m: int, num_pid_n: int, plot_pid: int | None):
    n = counts.size(0)
    expected = num_trials / n
    counts_f = counts.to(torch.float64)
    diffs = counts_f - expected
    abs_rel_dev = torch.max(torch.abs(diffs), dim=1).values / expected if expected > 0 else torch.zeros(n)
    chi2_per_pid = torch.sum((diffs * diffs) / expected, dim=1) if expected > 0 else torch.zeros(n)
    worst_dev, worst_pid = torch.max(abs_rel_dev, dim=0)
    avg_dev = float(torch.mean(abs_rel_dev))
    worst_chi2, worst_chi2_pid = torch.max(chi2_per_pid, dim=0)
    avg_chi2 = float(torch.mean(chi2_per_pid))

    print(f"\nTrials: {num_trials}, Grid: {num_pid_m}x{num_pid_n} (n={n})")
    print(f"Expected per-position count per pid: {expected:.4f}")
    print(f"Max relative deviation across pids: {float(worst_dev):.4f} (pid {int(worst_pid)})")
    print(f"Average relative deviation across pids: {avg_dev:.4f}")
    print(f"Max chi^2 across pids: {float(worst_chi2):.4f} (pid {int(worst_chi2_pid)})")
    print(f"Average chi^2 across pids: {avg_chi2:.4f}")

    pid_for_grid = plot_pid if plot_pid is not None else int(worst_pid)
    if pid_for_grid < 0 or pid_for_grid >= n:
        raise ValueError(f"plot_pid must be within [0, {n-1}] if provided")
    row_counts = counts[pid_for_grid].to(torch.int64)
    print("\nSelected pid distribution (counts grid): pid=", pid_for_grid)
    pretty_print_grid(row_counts, num_pid_m, num_pid_n)


def run_shuffle(
    grid_y: int,
    grid_x: int,
    num_trials: int | None,
    seed: int | None,
    summarize: bool,
    plot_pid: int | None,
):
    if grid_y <= 0 or grid_x <= 0:
        raise ValueError("grid dimensions must be positive integers")
    num_trials = num_trials or 1
    n = grid_y * grid_x
    rng = random.Random(seed) if seed is not None else random.Random()

    show_summary = summarize or num_trials > 1
    counts = torch.zeros((n, n), dtype=torch.int64) if show_summary else None
    printed = False
    for t in range(num_trials):
        a, c = tritonblas.choose_lcg_shuffle_params(n, rng=rng)
        output_grid = cpu_lcg_shuffle(n, a, c)
        _assert_bijection(output_grid, t, a, c)
        if not printed:
            print(f"Trial {t}: a={a}, c={c}, n={n} (grid={grid_y}x{grid_x})")
            pretty_print_grid(output_grid, grid_y, grid_x)
            printed = True
        if show_summary:
            rows = output_grid
            cols = torch.arange(n, dtype=torch.int64)
            counts.index_put_((rows, cols), torch.ones(n, dtype=counts.dtype), accumulate=True)

    if show_summary and counts is not None:
        summarize_counts(counts, num_trials, grid_y, grid_x, plot_pid)


def main():
    parser = argparse.ArgumentParser(description="CPU LCG shuffle tester")
    parser.add_argument("grid_y", type=int, help="Grid height (rows)")
    parser.add_argument("grid_x", type=int, help="Grid width (cols)")
    parser.add_argument(
        "--num-trials",
        dest="num_trials",
        type=int,
        default=None,
        help="Optional number of shuffle trials (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNG")
    parser.add_argument(
        "--plot-pid",
        dest="plot_pid",
        type=int,
        default=None,
        help="Pid to display in summary grid (default: worst pid)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Force summary statistics even for single trial",
    )
    args = parser.parse_args()
    run_shuffle(args.grid_y, args.grid_x, args.num_trials, args.seed, args.summarize, args.plot_pid)


if __name__ == "__main__":
    main()
