#!/usr/bin/env python3
import logging
import math
import time
from typing import List, Tuple, Literal, Dict

from sodium.x.runtime.pool.flow.context import FlowContext
from sodium.x.runtime.pool.flow.process.common import Flow
from sodium.x.runtime.pool.flow.types import FlowSelect, BaseFlow


class FlowBenchmark(BaseFlow):
    """
        New Implemented, AUTO_BENCHMARK, AUTO_BENCHMARK_NoSR.
    """

    logger: logging.Logger

    jobs: int
    processes: int
    chunksize: int
    tune: Literal[FlowSelect.AUTO_BENCHMARK, FlowSelect.AUTO_BENCHMARK_V2]

    def __init__(self,
                 jobs: int,
                 processes: int,
                 chunksize: int,
                 tune: Literal[FlowSelect.AUTO_BENCHMARK, FlowSelect.AUTO_BENCHMARK_V2],
                 logger: logging.Logger | None = None):

        if logger is None:
            self.logger = logging.getLogger(self.name)

        else:
            self.logger = logger

        self.jobs = jobs
        self.processes = processes
        self.chunksize = chunksize

        if tune in (FlowSelect.AUTO_BENCHMARK, FlowSelect.AUTO_BENCHMARK_V2):
            self.tune = tune

        else:
            raise Exception(f"tune is not accepted, only use for main context, "
                            f"except {(FlowSelect.AUTO_BENCHMARK, FlowSelect.AUTO_BENCHMARK_V2)}")

    @staticmethod
    def _test_mul_num(x: int) -> int:
        k = x + 1
        x = 32 - k
        j = (x / 32) * .4
        time.sleep(j + .2)
        return k * 2

    def _benchmark(self) -> Dict[FlowSelect, float]:

        data: Dict[FlowSelect, float] = {}
        params = tuple(map(lambda x: (x,), range(32)))
        outputs_expected: List[int] = [2, 4, 6, 8, 10, 12,
                                       14, 16, 18, 20, 22,
                                       24, 26, 28, 30, 32,
                                       34, 36, 38, 40, 42,
                                       44, 46, 48, 50, 52,
                                       54, 56, 58, 60, 62,
                                       64]

        with FlowContext(self.jobs, self.processes, self.chunksize) as ctx:

            for tune in FlowSelect:
                if self.tune in (FlowSelect.AUTO_BENCHMARK_V2,):
                    if tune in (FlowSelect.SINGLE_RUN_SYNC,):
                        continue

                if tune in (FlowSelect.AUTO_BENCHMARK, FlowSelect.AUTO_BENCHMARK_V2):
                    continue

                outputs: List[int] = []
                start = time.perf_counter()

                for result in ctx.run(self._test_mul_num, params, tune=tune, unpack=True).sync():
                    outputs.append(result)

                end = time.perf_counter()
                elapsed = end - start

                self.logger.info(f"{tune.name} time {elapsed :.2f} second(s)")

                outputs = sorted(outputs)
                if outputs != outputs_expected:
                    raise Exception("Outputs is not expected")

                data[tune] = elapsed

        return data

    def ranked(self) -> Tuple[FlowSelect, ...]:
        """
            Ranked all methods from FlowContext.
        :return:
        """
        self.logger.info("Start Benchmark...")

        benchmark = self._benchmark()

        elapsed_min = math.inf
        elapsed_max = -math.inf

        # scoring
        avg_time = 0.0
        elapsed_sum = 0.0

        # tunes
        # max_tune: FlowSelect
        # min_tune: FlowSelect

        for i, tune in enumerate(benchmark):
            elapsed = benchmark[tune]
            elapsed_min = min(elapsed, elapsed_min)
            elapsed_max = max(elapsed, elapsed_max)
            avg_time = elapsed if i == 0 else (avg_time + elapsed) / 2
            elapsed_sum += elapsed

            # Max Tune.
            # if elapsed_min == elapsed:
            # max_tune = tune

            # Min Tune.
            # if elapsed_max == elapsed:
            # min_tune = tune

        long_dist_min_max = elapsed_min / elapsed_max
        long_dist_max_min = elapsed_max / elapsed_min

        sum_min_max = elapsed_min + elapsed_max
        avg_min_max = elapsed_min / sum_min_max
        avg_max_min = elapsed_max / sum_min_max

        max_score = 1 - (elapsed_max / elapsed_sum)
        min_score = 1 - (elapsed_min / elapsed_sum)
        avg_score = 1 - (avg_time / elapsed_sum)

        self.logger.info(f"Score(Min) {min_score :.2f}")
        self.logger.info(f"Score(Max) {max_score :.2f}")
        self.logger.info(f"Score(Avg) {avg_score :.2f}")
        self.logger.info(f"Average(Min/Max) {avg_min_max :.2f}")
        self.logger.info(f"Average(Max/Min) {avg_max_min :.2f}")
        self.logger.info(f"Long_Distance(Min/Max) {long_dist_min_max :.2f}")
        self.logger.info(f"Long_Distance(Max/Min) {long_dist_max_min :.2f}")

        return tuple(tune
                     for elapsed in sorted(benchmark.values())
                     for tune in benchmark
                     if benchmark[tune] == elapsed)
        # return max_tune, min_tune

    def ctx(self) -> FlowContext:
        """
            Select Best Options. Create New Context.
        :return:
        """

        tune, *_ = self.ranked()
        context = FlowContext(self.jobs, self.processes, self.chunksize, tune=tune)
        self.logger.info(f"{context.name}::Using({tune})")
        return context


def show_benchmark():
    """
        Benchmark Show Up.
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    jobs, processes, chunksize = Flow.usage(0.6) * 16
    benchmark = FlowBenchmark(jobs, processes, chunksize, tune=FlowSelect.AUTO_BENCHMARK_V2)

    # ranked list...
    for i, tune in enumerate(benchmark.ranked()):
        print(f"{tune.name} #{i + 1}")
