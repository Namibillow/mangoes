# -*- coding: utf-8 -*-

import logging
import multiprocessing
import multiprocessing.queues
import traceback

logger = logging.getLogger(__name__)


class _Traceback(object):
    def __init__(self, trace):
        self.trace = trace


class _Sentinel(object):
    pass


class DataParallel:
    """Utility class to parallelize a function to apply on iterable data

    Parameters
    ----------
    task: callable
        function that takes an iterable as first argument and returns a value that can be reduced with the 'reducer'
        parameter.
        Its signature should be : `task_function(jobs_iterator, *args, **kwargs)`

    reducer: callable
        function that takes two objects of the same type (same type as the output of the 'task' function), merge them
        and returns an object of the same type.
        Its signature should be : `reduce_function(current_output, new_output)`
    nb_workers: int
        number of subprocess to use (default=1)
    batch: int, default None
        if provided, yields the result of task and call the reducer after `batch` iterations

    """

    def __init__(self, task, reducer, nb_workers=1, batch=None):
        self.task = task
        self.reducer = reducer

        self.nb_workers = max(1, min(nb_workers, multiprocessing.cpu_count()))
        self.workers = []
        self.job_dispatch_worker = None

        self.batch = batch

        self.input_queue = multiprocessing.Queue(maxsize=2*nb_workers)
        self.output_queue = multiprocessing.Queue()

        self._logger = logging.getLogger("{}.{}".format(__name__, self.__class__.__name__))

    def run(self, data, *args, **kwargs):
        """Run the `task` function on `data` with `nb_workers` processes.

        Parameters
        ----------
        data: iterator
            iterator over data to process
        args
            arguments for `task` function
        kwargs
            keywords arguments for `task` function

        Returns
        -------
        the result of the merging of all the values returned by `task`
        """
        running_workers = self._start_task_workers(max(1, self.nb_workers - 1), *args, **kwargs)
        self._start_dispatching_worker(data)

        try:
            result = self._process_output_queue(running_workers)
        except Exception:
            self._terminate()
            raise

        self.job_dispatching_worker.join()
        logger.debug("'Job dispatcher' worker terminated.")
        return result

    def _process_output_queue(self, running_workers):
        final_result = None
        while running_workers > 0:
            # Listen for regular outputs or exception's traceback
            try:
                output = self.output_queue.get()

                if isinstance(output, _Traceback):
                    raise Exception(output.trace)

                worker_id, part_result = output

                if isinstance(part_result, _Sentinel):
                    self.workers[worker_id].join()
                    running_workers -= 1
                    logger.debug("'Mapper' worker n°{} terminated, {} running.".format(worker_id, running_workers))
                else:
                    if final_result is None:
                        final_result = part_result
                    else:
                        final_result = self.reducer(final_result, part_result)
            except multiprocessing.queues.Empty:
                pass
        return final_result

    def _start_task_workers(self, nb, *args, **kwargs):
        self.workers = [self._create_a_task_worker(worker_id, *args, **kwargs)
                        for worker_id in range(nb)]
        for worker in self.workers:
            worker.start()
        return nb

    def _start_dispatching_worker(self, jobs):
        self.job_dispatching_worker = multiprocessing.Process(target=self._create_queue,
                                                              args=(jobs,))
        self.job_dispatching_worker.daemon = True
        self.job_dispatching_worker.start()

    def _create_queue(self, jobs):
        try:
            for job in jobs:
                self.input_queue.put(job)

            for _ in range(self.nb_workers):
                self.input_queue.put(_Sentinel)
        except Exception:
            tb = traceback.format_exc()
            self.output_queue.put(_Traceback(tb))

    def _worker(self, worker_id, *args, **kwargs):
        try:
            for primary_output in self.task(iter(self.input_queue.get, _Sentinel), *args, **kwargs):
                output = (worker_id, primary_output)
                self.output_queue.put(output)
            self.output_queue.put((worker_id, _Sentinel()))
        except Exception:
            tb = traceback.format_exc()
            self.output_queue.put(_Traceback(tb))

    def _create_a_task_worker(self, worker_id, *args, **kwargs):
        if self.batch:
            kwargs['batch'] = self.batch
        worker = multiprocessing.Process(target=self._worker,
                                         args=[worker_id, *args], kwargs=kwargs)
        worker.daemon = True
        return worker

    def _terminate(self):
        try:
            for worker in self.workers:
                worker.terminate()
            self.job_dispatch_worker.terminate()
        except Exception:
            pass
