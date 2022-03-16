from __future__ import annotations
from enum import Enum
from typing import Dict, List, Tuple, TYPE_CHECKING

import resources.loader as load
import utils.values as values

if TYPE_CHECKING:
    import resources.call_counts as cc


JsonModeMetrics = List[load.JsonType]
JsonWlMetrics = Dict["OptMode", JsonModeMetrics]
WlMetrics = Dict["OptMode", "ModeMetrics"]


class OptMode(Enum):
    FULL_PROF = 0
    NO_PROF = 1
    SAMP_MISS = 2
    SAMP_HIT = 3


class OverheadMetrics:
    def __init__(self) -> None:
        self.tot_time_overhead: float = 0.0
        self.tot_data_overhead: float = 0.0
        self.data_offset: float = 0.0
        self.data_per_record: float = 0.0
        self.instrumented_hit: float = 0.0
        self.sample_miss: float = 0.0
        self.sample_hit: float = 0.0
    
    def compute_overheads(self, wl_metrics: WlMetrics, call_count: cc.CallCounts) -> None:
        # no_prof   = main + (overhead_per_call / 2)
        #                     ^^^ main time contains approx. half of the per_call overhead
        # full_prof = main + (overhead_per_call / 2) + (total_overhead)
        #               (overhead_per_call * total_non_main_calls) ^^^
        # samp_miss = main + (overhead_per_call / 2) + (total_overhead)
        #        (overhead_per_sample_miss * total_non_main_calls) ^^^
        # samp_hit  = main + (overhead_per_call / 2) + (total_overhead)
        #         (overhead_per_sample_hit * total_non_main_calls) ^^^
        no_prof_time = wl_metrics[OptMode.NO_PROF].main_time
        # Total time / size overheads
        self.tot_time_overhead = wl_metrics[OptMode.FULL_PROF].main_time - no_prof_time
        self.tot_data_overhead = (
            wl_metrics[OptMode.FULL_PROF].data_size - wl_metrics[OptMode.NO_PROF].data_size
        )
        # Per-call overheads
        self.instrumented_hit = self.tot_time_overhead / (call_count.total_calls - 1)
        # TODO: measure the actual values
        self.sample_miss = self.instrumented_hit * 0.9
        # (
        #     (wl_metrics[OptMode.SAMP_MISS].main_time - no_prof_time) / (call_count.total_calls - 1)
        # )
        # TODO: measure the actual values
        self.sample_hit = self.instrumented_hit * 1.05
        # (
        #     (wl_metrics[OptMode.SAMP_HIT].main_time - no_prof_time) / (call_count.total_calls - 1)
        # )
        # Data_per_record = (full_prof_size - no_prof_size) / records_count
        #         (full_prof_records_count - no_prof_records_count) / 2 ^^^
        #    Each collected function record consists of 2 data records (func begin + end)
        self.data_offset = wl_metrics[OptMode.NO_PROF].data_size
        records = (wl_metrics[OptMode.FULL_PROF].records - wl_metrics[OptMode.NO_PROF].records) / 2
        self.data_per_record = (self.tot_data_overhead / records)


class ModeMetrics:
    def __init__(self, mode: OptMode, mode_metrics: JsonModeMetrics) -> None:
        self.mode: OptMode = mode
        self.data_size: float = 0.0
        self.records: float = 0.0
        self.main_time: float = 0.0
        self._init_metrics(mode_metrics)

    def _init_metrics(self, mode_metrics: JsonModeMetrics) -> None:
        for metrics in mode_metrics:
            self.data_size += metrics['data_size']
            self.records += metrics['records_count']
            self.main_time += self._find_main_time(metrics)
        # Compute average from all repeated runs
        self.data_size /= len(mode_metrics)
        self.records /= len(mode_metrics)
        self.main_time /= len(mode_metrics)
    
    @staticmethod
    def _find_main_time(run_metrics: load.JsonType) -> int:
        for pid, levels in run_metrics['cg_level_funcs'].items():
            # Ignore non-bottom processes
            if not run_metrics['process_hierarchy'][pid]['bottom']:
                continue
            # Find the main function record, if available
            for func_data in levels['0']:
                if func_data[0] == 'main':
                    return func_data[4]
        raise RuntimeError("No exclusive time for the 'main' function was found!")


class WorkloadMetrics:
    def __init__(self, workload_name: str, workload_metrics: JsonWlMetrics) -> None:
        self.workload_name: str = workload_name
        self.per_mode: WlMetrics = {
            mode: ModeMetrics(mode, mode_metrics) 
            for mode, mode_metrics in workload_metrics.items()
        }
        self.overhead: OverheadMetrics = OverheadMetrics()
    
    def compute_overheads(self, call_counts: cc.CallCounts) -> WorkloadMetrics:
        self.overhead.compute_overheads(self.per_mode, call_counts)
        return self


class ProjectMetrics:
    def __init__(self, project_name: str) -> None:
        self.project_name: str = project_name
        self.per_workload: Dict[str, WorkloadMetrics] = self._build_workload_metrics(project_name)
    
    def __getitem__(self, workload_name: str) -> WorkloadMetrics:
        return self.per_workload[workload_name]
    
    @classmethod
    def _build_workload_metrics(cls, project_name: str) -> Dict[str, WorkloadMetrics]:
        full_metrics = load.load_json(f'{project_name}_metrics.json', values.METRICS_DIR)
        # Workload name -> optimization mode -> metrics subset (JSON)
        workload_metrics: Dict[str, JsonWlMetrics] = {}
        for run_id, run_metrics in full_metrics.items():
            workload, opt_mode, _ = cls._parse_metrics_id(run_id)
            # Update the metrics subset for the workload
            (workload_metrics
             .setdefault(workload, {})
             .setdefault(OptMode(opt_mode), [])
             .append(run_metrics)
            )
        # Construct the Workload metrics from the Json-formatted file
        return {
            wl_name: WorkloadMetrics(wl_name, wl_metrics) 
            for wl_name, wl_metrics in workload_metrics.items()
        }
    
    @staticmethod
    def _parse_metrics_id(run_id: str) -> Tuple[str, int, int]:
        # init-run.opt-mode.round.workload
        parts = run_id.split('.', maxsplit=3)
        # Workload, opt_mode, round
        return parts[-1], int(parts[1]), int(parts[2])