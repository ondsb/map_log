from typing import Callable

import torch


# todo: wip


class ProfilerDecorator:
    def __init__(self, profiler_func: Callable) -> None:
        self._enabled = False
        self._profiler_func = profiler_func

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, new_value: bool) -> None:
        if not isinstance(new_value, bool):
            raise ValueError("enabled can only be set to a boolean value")
        self._enabled = new_value

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if self._enabled:
                with self._profiler_func():
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper


def profiler_func():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./bench_log"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
        with_modules=False,
    )


profiler_decorator = ProfilerDecorator(profiler_func)
profiler_decorator.enabled = True


@profiler_decorator
def some_function(x, y):
    return x + y


print(some_function(2, 3))
