"""
Diagnostics analysis module for plot_compare_metrics.

Provides utilities for reading and analyzing low-level hardware counter
diagnostics from training run logs.
"""

from typing import Dict, List, Any
import sys
import os

# Add parent directory to path to import fuseGNN
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fuseGNN.utils import analyze_run_diagnostics, compare_runs_diagnostics
except ImportError:
    # Fallback if fuseGNN is not in path
    analyze_run_diagnostics = None
    compare_runs_diagnostics = None


def extract_diagnostics_summary(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract and analyze diagnostics from a list of run logs.
    
    Args:
        logs: List of run log dictionaries
    
    Returns:
        Dictionary with diagnostics analysis
    """
    result = {
        'available': False,
        'runs_analyzed': 0,
        'runs_with_diagnostics': 0,
        'runs_with_success': 0,
        'common_metrics': [],
        'analyses': []
    }
    
    if not logs:
        return result
    
    # Check if any log has diagnostics
    has_any_diagnostics = any('diagnostics' in log for log in logs)
    if not has_any_diagnostics:
        return result
    
    result['available'] = True
    result['runs_analyzed'] = len(logs)
    
    if analyze_run_diagnostics is None:
        # Fallback analysis without fuseGNN
        for log in logs:
            if 'diagnostics' in log:
                result['runs_with_diagnostics'] += 1
                diagnostics = log['diagnostics']
                if diagnostics.get('status') == 'success':
                    result['runs_with_success'] += 1
    else:
        # Use fuseGNN analysis
        try:
            comparison = compare_runs_diagnostics(logs)
            result['runs_with_diagnostics'] = comparison['runs_with_diagnostics']
            result['runs_with_success'] = comparison['runs_with_success']
            result['common_metrics'] = comparison['common_derived_metrics']
            result['analyses'] = comparison['run_analyses']
        except Exception as e:
            result['analysis_error'] = str(e)
    
    return result


def format_diagnostics_report(diagnostics_summary: Dict[str, Any]) -> str:
    """
    Format diagnostics summary as a human-readable report string.
    
    Args:
        diagnostics_summary: Output from extract_diagnostics_summary
    
    Returns:
        Formatted report string
    """
    if not diagnostics_summary.get('available'):
        return "No diagnostics available in logs."
    
    lines = [
        "=" * 60,
        "HARDWARE COUNTER DIAGNOSTICS SUMMARY",
        "=" * 60,
        f"Runs analyzed: {diagnostics_summary['runs_analyzed']}",
        f"Runs with diagnostics: {diagnostics_summary['runs_with_diagnostics']}",
        f"Runs with successful capture: {diagnostics_summary['runs_with_success']}",
        ""
    ]
    
    if diagnostics_summary.get('common_metrics'):
        lines.append("Common derived metrics across all runs:")
        for metric in diagnostics_summary['common_metrics']:
            lines.append(f"  - {metric}")
    else:
        lines.append("No common derived metrics found.")
    
    if 'analysis_error' in diagnostics_summary:
        lines.append("")
        lines.append(f"Analysis error: {diagnostics_summary['analysis_error']}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def get_derived_metrics_from_log(log: Dict[str, Any]) -> Dict[str, float]:
    """
    Get derived metrics from a single run log if available.
    
    Args:
        log: Run log dictionary
    
    Returns:
        Dictionary of derived metric names -> values
    """
    if analyze_run_diagnostics is None or 'diagnostics' not in log:
        return {}
    
    try:
        analysis = analyze_run_diagnostics(log)
        return analysis.get('derived_metrics', {})
    except Exception:
        return {}


def get_raw_counters_from_log(log: Dict[str, Any]) -> Dict[str, int]:
    """
    Get raw hardware counters from a single run log if available.
    
    Args:
        log: Run log dictionary
    
    Returns:
        Dictionary of counter names -> values
    """
    if 'diagnostics' not in log:
        return {}
    
    diagnostics = log['diagnostics']
    return diagnostics.get('counters', {})
