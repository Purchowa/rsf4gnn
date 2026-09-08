"""
Hardware counter analysis and normalized metrics computation.

This module provides utilities to analyze raw hardware counters and derive
normalized metrics such as cache-miss rates, misses per kilo-instruction,
and other performance ratios.
"""

from typing import Dict, Optional, Tuple
import math


class CounterMetricsAnalyzer:
    """
    Analyze raw hardware counters and compute normalized metrics.
    
    This class takes raw counter values and derives performance metrics like:
    - Cache miss rates (L1, LLC)
    - Misses per kilo-instruction (MPKI)
    - Misses per second
    - Branch misprediction rates
    """
    
    def __init__(self, counters: Dict[str, int]):
        """
        Initialize the analyzer with raw counter values.
        
        Args:
            counters: Dictionary of counter_name -> value pairs
        """
        self.counters = counters
        self.derived_metrics = {}
        self._normalize_counter_names()
    
    def _normalize_counter_names(self):
        """
        Normalize counter names to a standard format for easier access.
        
        Perf outputs counters with varying formats; this standardizes them.
        """
        self.normalized = {}
        for key, value in self.counters.items():
            # Normalize key by making lowercase and removing special chars
            normalized_key = key.lower().replace('-', '_').replace(' ', '_')
            self.normalized[normalized_key] = value
    
    def _get_counter_value(self, *possible_names) -> Optional[int]:
        """
        Get counter value by trying multiple possible names.
        
        Args:
            *possible_names: Possible counter names to try
        
        Returns:
            Counter value if found, None otherwise
        """
        for name in possible_names:
            normalized = name.lower().replace('-', '_').replace(' ', '_')
            if normalized in self.normalized:
                return self.normalized[normalized]
        return None
    
    def compute_all_derived_metrics(self) -> Dict[str, float]:
        """
        Compute all available derived metrics from the counters.
        
        Returns:
            Dictionary of derived_metric_name -> value pairs
        """
        metrics = {}
        
        # Cache miss rates
        l1_miss_rate = self.compute_l1_cache_miss_rate()
        if l1_miss_rate is not None:
            metrics['l1_dcache_miss_rate'] = l1_miss_rate
        
        llc_miss_rate = self.compute_llc_cache_miss_rate()
        if llc_miss_rate is not None:
            metrics['llc_miss_rate'] = llc_miss_rate
        
        # MPKI metrics
        l1_mpki = self.compute_l1_misses_per_kilo_instruction()
        if l1_mpki is not None:
            metrics['l1_dcache_mpki'] = l1_mpki
        
        llc_mpki = self.compute_llc_misses_per_kilo_instruction()
        if llc_mpki is not None:
            metrics['llc_mpki'] = llc_mpki
        
        # Branch metrics
        branch_mispredict_rate = self.compute_branch_misprediction_rate()
        if branch_mispredict_rate is not None:
            metrics['branch_misprediction_rate'] = branch_mispredict_rate
        
        # Additional ratios
        cpi = self.compute_cycles_per_instruction()
        if cpi is not None:
            metrics['cycles_per_instruction'] = cpi

        # GPU L2 cache metrics
        gpu_l2_hit_rate = self.compute_gpu_l2_hit_rate()
        if gpu_l2_hit_rate is not None:
            metrics['gpu_l2_hit_rate'] = gpu_l2_hit_rate

        gpu_l2_miss_rate = self.compute_gpu_l2_miss_rate()
        if gpu_l2_miss_rate is not None:
            metrics['gpu_l2_miss_rate'] = gpu_l2_miss_rate
        
        return metrics

    def compute_gpu_l2_hit_rate(self) -> Optional[float]:
        """
        Compute GPU L2 hit rate from NCU-derived counters.

        Returns:
            Hit rate in [0.0, 1.0] or None if required counters unavailable.
        """
        hit_rate = self._get_counter_value('gpu_l2_hit_rate')
        if hit_rate is not None:
            if hit_rate > 1.0:
                return hit_rate / 100.0
            return hit_rate

        hits = self._get_counter_value('gpu_l2_hits')
        misses = self._get_counter_value('gpu_l2_misses')
        if hits is None or misses is None:
            return None
        total = hits + misses
        if total == 0:
            return None
        return hits / total

    def compute_gpu_l2_miss_rate(self) -> Optional[float]:
        """
        Compute GPU L2 miss rate from NCU-derived counters.

        Returns:
            Miss rate in [0.0, 1.0] or None if required counters unavailable.
        """
        hit_rate = self.compute_gpu_l2_hit_rate()
        if hit_rate is None:
            return None
        return 1.0 - hit_rate
    
    def compute_l1_cache_miss_rate(self) -> Optional[float]:
        """
        Compute L1 data cache miss rate.
        
        Formula: L1_misses / L1_loads
        
        Returns:
            Miss rate (0.0-1.0) or None if required counters unavailable
        """
        misses = self._get_counter_value(
            'L1-dcache-load-misses',
            'l1_dcache_load_misses',
            'l1-dcache-load-misses'
        )
        loads = self._get_counter_value(
            'L1-dcache-loads',
            'l1_dcache_loads',
            'l1-dcache-loads'
        )
        
        if misses is None or loads is None or loads == 0:
            return None
        
        return misses / loads
    
    def compute_llc_cache_miss_rate(self) -> Optional[float]:
        """
        Compute Last-Level Cache (LLC) miss rate.
        
        Formula: LLC_misses / LLC_references
        
        Returns:
            Miss rate (0.0-1.0) or None if required counters unavailable
        """
        misses = self._get_counter_value(
            'LLC-load-misses',
            'llc_load_misses',
            'llc-load-misses'
        )
        references = self._get_counter_value(
            'LLC-loads',
            'llc_loads',
            'llc-loads',
            'cache-references'
        )
        
        if misses is None or references is None or references == 0:
            return None
        
        return misses / references
    
    def compute_l1_misses_per_kilo_instruction(self) -> Optional[float]:
        """
        Compute L1 misses per kilo-instruction (MPKI).
        
        Formula: (L1_misses / instructions) * 1000
        
        Returns:
            MPKI value or None if required counters unavailable
        """
        misses = self._get_counter_value(
            'L1-dcache-load-misses',
            'l1_dcache_load_misses',
            'l1-dcache-load-misses'
        )
        instructions = self._get_counter_value(
            'instructions',
            'insn',
            'inst_retired.any'
        )
        
        if misses is None or instructions is None or instructions == 0:
            return None
        
        return (misses / instructions) * 1000.0
    
    def compute_llc_misses_per_kilo_instruction(self) -> Optional[float]:
        """
        Compute LLC misses per kilo-instruction (MPKI).
        
        Formula: (LLC_misses / instructions) * 1000
        
        Returns:
            MPKI value or None if required counters unavailable
        """
        misses = self._get_counter_value(
            'LLC-load-misses',
            'llc_load_misses',
            'llc-load-misses'
        )
        instructions = self._get_counter_value(
            'instructions',
            'insn',
            'inst_retired.any'
        )
        
        if misses is None or instructions is None or instructions == 0:
            return None
        
        return (misses / instructions) * 1000.0
    
    def compute_branch_misprediction_rate(self) -> Optional[float]:
        """
        Compute branch misprediction rate.
        
        Formula: branch_misses / branch_instructions
        
        Returns:
            Misprediction rate (0.0-1.0) or None if required counters unavailable
        """
        misses = self._get_counter_value(
            'branch-misses',
            'branch_misses',
            'br_misp_retired.all_branches'
        )
        branches = self._get_counter_value(
            'branch-instructions',
            'branch_instructions',
            'branches'
        )
        
        if misses is None or branches is None or branches == 0:
            return None
        
        return misses / branches
    
    def compute_cycles_per_instruction(self) -> Optional[float]:
        """
        Compute Cycles Per Instruction (CPI).
        
        Formula: cycles / instructions
        
        Returns:
            CPI value or None if required counters unavailable
        """
        cycles = self._get_counter_value(
            'cycles',
            'cpu-cycles',
            'cpu_cycles'
        )
        instructions = self._get_counter_value(
            'instructions',
            'insn',
            'inst_retired.any'
        )
        
        if cycles is None or instructions is None or instructions == 0:
            return None
        
        return cycles / instructions
    
    def compute_ipc(self) -> Optional[float]:
        """
        Compute Instructions Per Cycle (IPC).
        
        Formula: instructions / cycles
        
        Returns:
            IPC value or None if required counters unavailable
        """
        cpi = self.compute_cycles_per_instruction()
        if cpi is None or cpi == 0:
            return None
        return 1.0 / cpi


def analyze_run_diagnostics(run_log: Dict) -> Dict:
    """
    Analyze diagnostics from a single run log and compute derived metrics.
    
    This function reads the diagnostics block from a training run log and
    computes all available normalized metrics.
    
    Args:
        run_log: Dictionary containing the run log (with 'diagnostics' key)
    
    Returns:
        Dictionary with analysis results including derived metrics
    """
    result = {
        'has_diagnostics': False,
        'status': 'unavailable',
        'derived_metrics': {},
        'raw_counters': {},
    }
    
    # Check if diagnostics exist
    if 'diagnostics' not in run_log:
        return result
    
    diagnostics = run_log['diagnostics']
    result['has_diagnostics'] = True
    result['status'] = diagnostics.get('status', 'unavailable')
    
    # Get raw counters
    if 'counters' in diagnostics:
        result['raw_counters'] = diagnostics['counters'].copy()
        
        # Only compute derived metrics if we have counters
        if diagnostics['counters']:
            try:
                analyzer = CounterMetricsAnalyzer(diagnostics['counters'])
                result['derived_metrics'] = analyzer.compute_all_derived_metrics()
            except Exception as e:
                result['analysis_error'] = str(e)
    
    # Include metadata
    if 'metadata' in diagnostics:
        result['metadata'] = diagnostics['metadata'].copy()
    
    return result


def compare_runs_diagnostics(run_logs: list) -> Dict:
    """
    Compare diagnostics across multiple run logs.
    
    Args:
        run_logs: List of run log dictionaries
    
    Returns:
        Dictionary with comparative analysis
    """
    analyses = [analyze_run_diagnostics(log) for log in run_logs]
    
    # Find common derived metrics
    common_metrics = set(analyses[0]['derived_metrics'].keys()) if analyses else set()
    for analysis in analyses[1:]:
        common_metrics &= set(analysis['derived_metrics'].keys())
    
    return {
        'run_analyses': analyses,
        'common_derived_metrics': list(common_metrics),
        'runs_with_diagnostics': sum(1 for a in analyses if a['has_diagnostics']),
        'runs_with_success': sum(1 for a in analyses if a['status'] == 'success'),
    }
