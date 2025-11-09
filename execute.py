import io
import argparse
import importlib
import inspect
import sys
import json
import traceback
import torch
import sympy
from dataclasses import dataclass, asdict, field, is_dataclass, fields
import os
import re
from execute_util import Rendering, pop_renderings
from file_util import ensure_directory_exists, relativize


@dataclass(frozen=True)
class StackElement:
    path: str
    """The path to the file containing the code."""

    line_number: int
    """The line number of the code."""

    function_name: str
    """The name of the function that we're in."""

    code: str
    """The source code that is executed."""


@dataclass
class Step:
    stack: list[StackElement]
    """The stack of function calls."""

    env: dict[str, any]
    """The local variables including function arguments(that we're @inspect-ing)."""

    renderings: list[Rendering] = field(default_factory=list)
    """The output of the code (see execute_util.py)."""

    stdout: str | None = None
    """The stdout of the code."""

    stderr: str | None = None
    """The stderr of the code."""


@dataclass(frozen=True)
class Trace:
    files: dict[str, str]
    steps: list[Step]


def to_primitive(value: any) -> any:
    if isinstance(value, (int, float, str, bool)):
        return value
    # Force it to be a primitive
    return str(value)

def to_serializable_value(value: any) -> any:
    """Convert any type to a serializable value."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, sympy.core.numbers.Integer):
        return int(value)
    if isinstance(value, sympy.core.numbers.Float):
        return float(value)
    if isinstance(value, sympy.core.symbol.Symbol):
        return str(value)  # Would be nice to signal that this is not a string
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, list):
        return [to_serializable_value(item) for item in value]
    if isinstance(value, dict):
        return {to_primitive(k): to_serializable_value(v) for k, v in value.items()}
    if is_dataclass(value):
        return {
            field.name: to_serializable_value(getattr(value, field.name))
            for field in fields(value)
        }
    # Force it to be a primitive
    return str(value)

def get_inspect_variables(code: str) -> list[str]:
    """
    If code contains "@inspect <variable>" (as a comment), return those variables.
    Example code:
        x, y = str.split("a,b")  # @inspect x, @inspect y
    We would return ["x", "y"]
    """
    variables = []
    # Find all "@inspect <variable>" occurrences
    matches = re.finditer(r"@inspect\s+(\w+)", code)
    for match in matches:
        variables.append(match.group(1))
    return variables


def execute(module_name: str, inspect_all_variables: bool) -> Trace:
    """
    Execute the module and return a trace of the execution.
    """
    steps: list[Step] = []

    # Capture stdout and stderr
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    #sys.stdout = stdout_buffer
    #sys.stderr = stderr_buffer

    # Figure out which files we're actually tracing
    visible_paths = []

    # Stack of locations that we're stepping over
    stepovers = []

    def get_stack() -> list[StackElement]:
        """Return the last element of `stack`, but skip over items where local_trace_func is active."""
        stack = []
        # stack looks like this:
        #   <module> execute [good stuff to return] local_trace_func trace_func get_stack
        items = traceback.extract_stack()
        assert items[0].name == "<module>"
        assert items[1].name == "execute"
        for item in traceback.extract_stack()[2:]:
            if item.name in ("trace_func", "local_trace_func", "get_stack"):
                continue
            stack.append(StackElement(
                path=relativize(item.filename),
                line_number=item.lineno,
                function_name=item.name,
                code=item.line,
            ))
        return stack
    
    def trace_func(frame, event, arg):
        """
        trace_func and local_trace_func are called on various lines of code when executed.
        - trace_func is called *before* a line of code is executed.
        - local_trace_func is called *after* a line of code has been executed
          and will have the values of the variables.
        We generally keep the local_trace_func version.  However, when you have
        a function call that you're tracing through, you want to keep both
        versions.

        We don't care about all the events, so here are the rules:
        - In local_trace_func, if the previous event was the same line (presumably the trace_func)
        - Remove all trace_func(return)
        """

        # Get the current file path from the frame and skip if not in visible paths
        # to avoid tracing deep into imports (which would be slow and irrelevant)
        current_path = frame.f_code.co_filename
        if current_path not in visible_paths:
            return trace_func

        stack = get_stack()

        if event == "return":
            return trace_func

        # Print the current line of code
        item = stack[-1]
        if "@stepover" in item.code:
            if len(stepovers) > 0 and stepovers[-1] == (item.path, item.line_number):
                stepovers.pop()
            else:
                stepovers.append((item.path, item.line_number))
        
        # Skip everything that is strictly under stepovers
        if any(stepover[0] == item.path and stepover[1] == item.line_number for stepover in stepovers for item in stack[:-1]):
            return trace_func

        print(f"  [{len(steps)} {os.path.basename(item.path)}:{item.line_number}] {item.code}", file=real_stdout)

        open_step = Step(
            stack=stack,
            env={},
            stdout="",
            stderr="",
        )
        if len(steps) == 0 or open_step.stack != steps[-1].stack:  # Only add a step if it's not redundant
            steps.append(open_step)
        open_step_index = len(steps) - 1

        def local_trace_func(frame, event, arg):
            """This is called *after* a line of code has been executed."""
            # If the last step was the same line, then just use the same one
            # Otherwise, create a new step (e.g., returning from a function)
            if open_step_index == len(steps) - 1:
                close_step = steps[-1]
            else:
                print(f"  [{len(steps)} {os.path.basename(item.path)}:{item.line_number}] {item.code}", file=real_stdout)

                close_step = Step(
                    stack=stack,
                    env={},
                    stdout="",
                    stderr="",
                )
                steps.append(close_step)

            # Update the environment with the actual values
            locals = frame.f_locals
            if inspect_all_variables:
                vars = locals.keys()
            else:
                vars = get_inspect_variables(item.code)
            for var in vars:
                if var in locals:
                    close_step.env[var] = to_serializable_value(locals[var])
                else:
                    print(f"WARNING: variable {var} not found in locals")
                print(f"    env: {var} = {close_step.env.get(var)}", file=real_stdout)
        
            # Capture stdout and stderr
            close_step.stdout = stdout_buffer.getvalue()
            close_step.stderr = stderr_buffer.getvalue()
            stdout_buffer.truncate(0)
            stdout_buffer.seek(0)
            stderr_buffer.truncate(0)
            stderr_buffer.seek(0)

            # Capture the renderings of the last line
            close_step.renderings = pop_renderings()

            # Pass control back to the global trace function
            return trace_func(frame, event, arg)

        # Pass control to local_trace_func to update the environment
        return local_trace_func
    
    # Run the module
    module = importlib.import_module(module_name)
    visible_paths.append(inspect.getfile(module))
    sys.settrace(trace_func)
    module.main()
    sys.settrace(None)

    # Restore stdout and stderr
    sys.stdout = real_stdout
    sys.stderr = real_stderr

    files = {relativize(path): open(path).read() for path in visible_paths}
    trace = Trace(steps=steps, files=files)
    return trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", help="List of modules to execute (e.g., lecture_01)", type=str, nargs="+")
    parser.add_argument("-o", "--output_path", help="Path to save the trace", type=str, default="var/traces")
    parser.add_argument("-I", "--inspect-all-variables", help="Inspect all variables (default: only inspect variables mentioned in @inspect comments)", action="store_true")
    args = parser.parse_args()

    ensure_directory_exists(args.output_path)

    for module in args.module:
        module = module.replace(".py", "")  # Just in case
        print(f"Executing {module}...")
        trace = execute(module_name=module, inspect_all_variables=args.inspect_all_variables)
        print(f"{len(trace.steps)} steps")
        output_path = os.path.join(args.output_path, f"{module}.json")
        print(f"Saving trace to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(asdict(trace), f, indent=2)
