"""Benchmark interfaces for the PS-named evaluation sets.

Every benchmark here is a real, documented interface with a `run()` method -
but none of them have been executed, because none of the actual benchmark
datasets could be downloaded into this sandbox (see
docs/network_constraints.md). `run()` returns a status dict rather than a
score whenever the dataset isn't present locally; it never fabricates a
number. This satisfies the frozen instruction: "create the benchmark
interfaces now, but label all unexecuted benchmarks as NOT YET EVALUATED."
"""
from __future__ import annotations

import dataclasses
import os
from typing import Callable, Optional

NOT_YET_EVALUATED = "NOT_YET_EVALUATED"


@dataclasses.dataclass
class BenchmarkResult:
    benchmark: str
    task: str
    split: str
    metric: str
    status: str
    score: Optional[float] = None
    notes: str = ""


class BenchmarkTask:
    name: str
    task: str
    split: str
    metric: str

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path

    def _dataset_present(self) -> bool:
        return bool(self.dataset_path) and os.path.exists(self.dataset_path)

    def run(self, specialist_fn: Optional[Callable] = None) -> BenchmarkResult:
        if not self._dataset_present():
            return BenchmarkResult(
                benchmark=self.name,
                task=self.task,
                split=self.split,
                metric=self.metric,
                status=NOT_YET_EVALUATED,
                score=None,
                notes=(
                    f"Dataset not found at '{self.dataset_path}'. This benchmark "
                    "has not been run - see docs/network_constraints.md for why "
                    "it could not be downloaded in the dev sandbox."
                ),
            )
        return self._run_impl(specialist_fn)

    def _run_impl(self, specialist_fn) -> BenchmarkResult:
        raise NotImplementedError(
            f"{self.name}: dataset was found locally but the scoring loop has "
            "not been implemented yet - wire this up once real data is present."
        )


class VRSBenchTask(BenchmarkTask):
    name = "VRSBench"
    task = "single-image captioning / VQA / grounding"
    split = "test"
    metric = "task-appropriate (BLEU/CIDEr for captioning, accuracy for VQA, IoU for grounding)"


class RSVQATask(BenchmarkTask):
    name = "RSVQA"
    task = "single-image visual question answering"
    split = "test"
    metric = "accuracy (per question type)"


class CDVQATask(BenchmarkTask):
    name = "CDVQA"
    task = "bi-temporal change-based visual question answering"
    split = "test"
    metric = "accuracy (per question type)"


class ISROSACEvalTask(BenchmarkTask):
    name = "ISRO/SAC evaluation set"
    task = "Cartosat-2S optical + RISAT SAR paired evaluation (per SIH26167 spec)"
    split = "held-out (annotations withheld from teams)"
    metric = "task-appropriate, normalised across metrics per PS text"


ALL_BENCHMARKS = [VRSBenchTask, RSVQATask, CDVQATask, ISROSACEvalTask]


def run_all(dataset_paths: Optional[dict] = None) -> list[BenchmarkResult]:
    dataset_paths = dataset_paths or {}
    results = []
    for cls in ALL_BENCHMARKS:
        task = cls(dataset_path=dataset_paths.get(cls.name))
        results.append(task.run())
    return results
