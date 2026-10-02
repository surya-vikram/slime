"""CPU threads per trainer process: its share of the CPUs this container may use.

torch starts one intra-op thread per two visible cores in every process (120 when 240 cores are visible),
even when a cgroup quota allows far fewer (50 here) and several trainer ranks share them. Oversubscribed
CPU work then stalls: the routing-replay fill took minutes per step until it ran on a few threads.
"""
import os


def available_cpus(cpu_max='/sys/fs/cgroup/cpu.max', cfs_quota='/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                   cfs_period='/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
    """CPUs this process may use: its affinity, capped by a cgroup v2 or v1 quota when one is set."""
    cpus = float(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
    try:
        quota, period = open(cpu_max).read().split()[:2]
        if quota != 'max':
            cpus = min(cpus, int(quota) / int(period))
        return cpus
    except (OSError, ValueError):
        pass
    try:
        quota, period = int(open(cfs_quota).read()), int(open(cfs_period).read())
        if quota > 0 and period > 0:
            cpus = min(cpus, quota / period)
    except (OSError, ValueError):
        pass
    return cpus


def threads_per_rank(ranks_per_node, cpus=None):
    """An even share of the available CPUs for each of `ranks_per_node` processes, at least one."""
    cpus = available_cpus() if cpus is None else cpus
    return max(1, int(cpus // max(1, ranks_per_node)))
