"""Equinox Module summary library."""

import dataclasses
import io
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import rich.console
import rich.table
import rich.text
import yaml


class _ValueRepresentation(ABC):
    """A class that represents a value in the summary table."""

    @abstractmethod
    def render(self) -> str:
        ...


@dataclasses.dataclass
class _ArrayRepresentation(_ValueRepresentation):
    shape: Tuple[int, ...]
    dtype: Any

    @classmethod
    def from_array(cls, x) -> '_ArrayRepresentation':
        return cls(jnp.shape(x), jnp.result_type(x))

    def render(self):
        shape_repr = ','.join(str(x) for x in self.shape)
        return f'[dim]{self.dtype}[/dim][{shape_repr}]'


@dataclasses.dataclass
class _ObjectRepresentation(_ValueRepresentation):
    obj: Any

    def render(self):
        return repr(self.obj)


@dataclasses.dataclass
class Row:
    """Contains the information about a single row in the summary table.

    Attributes:
        path: A tuple of strings that represents the path to the module.
        module_name: Name of the module class.
        inputs: inputs to the module.
        outputs: Output of the Module.
        parameters: Dictionary of trainable parameters in the module.
        arrays: Dictionary of non-trainable arrays in the module.
        flops: FLOPs cost of calling the module (if computed).
    """

    path: Tuple[str, ...]
    module_name: str
    inputs: Any
    outputs: Any
    parameters: Dict[str, Any]
    arrays: Dict[str, Any]
    flops: Optional[int] = None

    def __post_init__(self):
        self.inputs = self.inputs
        self.outputs = self.outputs
        self.parameters = self.parameters
        self.arrays = self.arrays

    def size_and_bytes(self) -> Tuple[int, int, int, int]:
        """Returns (param_size, param_bytes, array_size, array_bytes)"""
        param_leaves = jax.tree_util.tree_leaves(self.parameters)
        array_leaves = jax.tree_util.tree_leaves(self.arrays)
        
        param_size = sum(x.size for x in param_leaves if hasattr(x, 'size'))
        param_bytes = sum(
            x.size * x.dtype.itemsize for x in param_leaves if hasattr(x, 'size')
        )
        
        array_size = sum(x.size for x in array_leaves if hasattr(x, 'size'))
        array_bytes = sum(
            x.size * x.dtype.itemsize for x in array_leaves if hasattr(x, 'size')
        )
        
        return param_size, param_bytes, array_size, array_bytes


class Table(List[Row]):
    """A list of Row objects representing module summary."""

    def __init__(self, module: eqx.Module, rows: Sequence[Row]):
        super().__init__(rows)
        self.module = module


def _get_flops(fn, *args, **kwargs):
    """Estimate FLOPs for a function."""
    try:
        e = jax.jit(fn).lower(*args, **kwargs)
        cost = e.cost_analysis()
        if cost is None:
            return 0
        flops = int(cost['flops']) if 'flops' in cost else 0
        return flops
    except:
        return 0


def _get_value_representation(x: Any) -> _ValueRepresentation:
    """Convert a value to its representation for display."""
    if isinstance(x, (int, float, bool, type(None))) or (
        isinstance(x, np.ndarray) and np.isscalar(x)
    ):
        return _ObjectRepresentation(x)
    try:
        return _ArrayRepresentation.from_array(x)
    except:
        return _ObjectRepresentation(x)


def _represent_tree(x):
    """Returns a tree with the same structure as `x` but with each leaf replaced
    by a `_ValueRepresentation` object."""
    return jax.tree_util.tree_map(_get_value_representation, x)


def _maybe_render(x):
    """Render a value representation if it has a render method."""
    return x.render() if hasattr(x, 'render') else repr(x)


def _normalize_structure(obj):
    """Normalize object structure for YAML serialization."""
    if isinstance(obj, _ValueRepresentation):
        return obj
    if isinstance(obj, (tuple, list)):
        return tuple(map(_normalize_structure, obj))
    elif isinstance(obj, dict):
        return {
            _normalize_structure(k): _normalize_structure(v) for k, v in obj.items()
        }
    elif dataclasses.is_dataclass(obj):
        return {
            f.name: _normalize_structure(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    else:
        return obj


def _as_yaml_str(value) -> str:
    """Convert value to YAML string representation."""
    if (hasattr(value, '__len__') and len(value) == 0) or value is None:
        return ''

    file = io.StringIO()
    yaml.safe_dump(
        value,
        file,
        default_flow_style=False,
        indent=2,
        sort_keys=False,
        explicit_end=False,
    )
    return file.getvalue().replace('\n...', '').replace("'", '').strip()


def _bytes_repr(num_bytes):
    """Human-readable bytes representation."""
    count, units = (
        (f'{num_bytes / 1e9 :,.1f}', 'GB')
        if num_bytes > 1e9
        else (f'{num_bytes / 1e6 :,.1f}', 'MB')
        if num_bytes > 1e6
        else (f'{num_bytes / 1e3 :,.1f}', 'KB')
        if num_bytes > 1e3
        else (f'{num_bytes:,}', 'B')
    )
    return f'{count} {units}'


def _size_and_bytes_repr(size: int, num_bytes: int) -> str:
    """Format size and bytes for display."""
    if not size:
        return ''
    bytes_repr = _bytes_repr(num_bytes)
    return f'{size:,} [dim]({bytes_repr})[/dim]'


def _get_rich_repr(obj, console_kwargs):
    """Get rich representation of an object."""
    f = io.StringIO()
    console = rich.console.Console(file=f, **console_kwargs)
    console.print(obj)
    return f.getvalue()


def _summary_tree_map(f, tree, *rest):
    """Tree map that treats None as a leaf."""
    return jax.tree_util.tree_map(f, tree, *rest, is_leaf=lambda x: x is None)


def _process_inputs(args, kwargs) -> Any:
    """Normalize the representation of args and kwargs for the inputs column."""
    if args and kwargs:
        input_values = (*args, kwargs)
    elif args and not kwargs:
        input_values = args[0] if len(args) == 1 else args
    elif kwargs and not args:
        input_values = kwargs
    else:
        input_values = ()
    return input_values


def _analyze_module(
    module: eqx.Module,
    path: Tuple[str, ...] = (),
    depth: Optional[int] = None,
    compute_flops: bool = False,
    _current_depth: int = 0
) -> List[Tuple[Tuple[str, ...], str, eqx.Module]]:
    """Recursively analyze module structure."""
    modules = []
    
    # Add current module
    modules.append((path, type(module).__name__, module))
    
    # If we've reached max depth, don't recurse further
    if depth is not None and _current_depth >= depth:
        return modules
    
    # Recursively analyze submodules
    for name, submodule in jax.tree_util.tree_flatten_with_path(module)[0]:
        if isinstance(submodule, eqx.Module) and len(name) > 0:
            # Convert path to string representation
            subpath = path + (str(name[0].key),) if hasattr(name[0], 'key') else path + (str(name[0]),)
            submodules = _analyze_module(
                submodule, 
                subpath, 
                depth, 
                compute_flops, 
                _current_depth + 1
            )
            modules.extend(submodules)
    
    return modules


def _get_module_table(
    module: eqx.Module,
    depth: Optional[int] = None,
    compute_flops: bool = False,
) -> Callable[..., Table]:
    """Create a function that generates a table for the module."""

    def _get_table_fn(*args, **kwargs):
        # Analyze module structure
        module_info = _analyze_module(module, depth=depth, compute_flops=compute_flops)
        
        rows = []
        
        for path, module_name, mod in module_info:
            # Separate parameters and arrays
            params, arrays = eqx.partition(mod, eqx.is_array)
            
            # Filter to only get trainable parameters and non-trainable arrays
            trainable_params = jax.tree_util.tree_map(
                lambda x: x if isinstance(x, jax.Array) else None, params
            )
            non_trainable_arrays = jax.tree_util.tree_map(
                lambda x: x if isinstance(x, jax.Array) else None, arrays
            )
            
            # Remove None values
            trainable_params = jax.tree_util.tree_map(
                lambda x: x, trainable_params, is_leaf=lambda x: x is None
            )
            non_trainable_arrays = jax.tree_util.tree_map(
                lambda x: x, non_trainable_arrays, is_leaf=lambda x: x is None
            )
            
            # For the root module, compute inputs and outputs
            if path == ():
                inputs = _process_inputs(args, kwargs)
                
                # Compute outputs using eval_shape to avoid actual computation
                def forward(*args, **kwargs):
                    return module(*args, **kwargs)
                
                outputs = jax.eval_shape(forward, *args, **kwargs)
                
                # Compute FLOPs if requested
                flops = None
                if compute_flops:
                    flops = _get_flops(forward, *args, **kwargs)
            else:
                inputs = None
                outputs = None
                flops = None
            
            rows.append(Row(
                path=path,
                module_name=module_name,
                inputs=inputs,
                outputs=outputs,
                parameters=trainable_params,
                arrays=non_trainable_arrays,
                flops=flops
            ))
        
        return Table(module, rows)
    
    return _get_table_fn


def _render_table(
    table: Table,
    console_kwargs: Optional[Dict[str, Any]] = None,
    table_kwargs: Optional[Dict[str, Any]] = None,
    column_kwargs: Optional[Dict[str, Any]] = None,
    compute_flops: bool = False,
) -> str:
    """Render a Table to a string representation using rich."""
    if console_kwargs is None:
        console_kwargs = {'force_terminal': True, 'force_jupyter': False}
    if table_kwargs is None:
        table_kwargs = {}
    if column_kwargs is None:
        column_kwargs = {}

    rich_table = rich.table.Table(
        show_header=True,
        show_lines=True,
        show_footer=True,
        title=f'{table.module.__class__.__name__} Summary',
        **table_kwargs,
    )

    # Add columns
    rich_table.add_column('path', **column_kwargs)
    rich_table.add_column('module', **column_kwargs)
    rich_table.add_column('inputs', **column_kwargs)
    rich_table.add_column('outputs', **column_kwargs)
    
    if compute_flops:
        rich_table.add_column('flops', **column_kwargs)
    
    rich_table.add_column('parameters', **column_kwargs)
    rich_table.add_column('arrays', **column_kwargs)

    # Track totals
    total_param_size, total_param_bytes = 0, 0
    total_array_size, total_array_bytes = 0, 0

    for row in table:
        param_size, param_bytes, array_size, array_bytes = row.size_and_bytes()
        total_param_size += param_size
        total_param_bytes += param_bytes
        total_array_size += array_size
        total_array_bytes += array_bytes

        # Format path
        path_repr = '/'.join(row.path) if row.path else ''
        
        # Format parameters representation
        param_repr = ''
        if param_size > 0:
            param_structure = _represent_tree(row.parameters)
            param_structure = _normalize_structure(param_structure)
            param_yaml = _as_yaml_str(
                _summary_tree_map(_maybe_render, param_structure)
            )
            if param_yaml:
                param_repr += param_yaml + '\n\n'
            param_repr += f'[bold]{_size_and_bytes_repr(param_size, param_bytes)}[/bold]'
        
        # Format arrays representation  
        array_repr = ''
        if array_size > 0:
            array_structure = _represent_tree(row.arrays)
            array_structure = _normalize_structure(array_structure)
            array_yaml = _as_yaml_str(
                _summary_tree_map(_maybe_render, array_structure)
            )
            if array_yaml:
                array_repr += array_yaml + '\n\n'
            array_repr += f'[bold]{_size_and_bytes_repr(array_size, array_bytes)}[/bold]'

        # Format inputs and outputs
        inputs_repr = ''
        outputs_repr = ''
        
        if row.inputs is not None:
            inputs_structure = _represent_tree(row.inputs)
            inputs_structure = _normalize_structure(inputs_structure)
            inputs_repr = _as_yaml_str(
                _summary_tree_map(_maybe_render, inputs_structure)
            )
            
        if row.outputs is not None:
            outputs_structure = _represent_tree(row.outputs)
            outputs_structure = _normalize_structure(outputs_structure)
            outputs_repr = _as_yaml_str(
                _summary_tree_map(_maybe_render, outputs_structure)
            )

        # Build row data
        row_data = [
            path_repr,
            row.module_name,
            inputs_repr,
            outputs_repr,
        ]
        
        if compute_flops:
            flops_repr = str(row.flops) if row.flops is not None else ''
            row_data.append(flops_repr)
            
        row_data.extend([param_repr, array_repr])
        
        rich_table.add_row(*row_data)

    # Add footer with totals
    footer_col_idx = 4 if compute_flops else 3
    rich_table.columns[footer_col_idx].footer = rich.text.Text.from_markup(
        'Total', justify='right'
    )
    
    rich_table.columns[footer_col_idx + 1].footer = _size_and_bytes_repr(
        total_param_size, total_param_bytes
    )
    rich_table.columns[footer_col_idx + 2].footer = _size_and_bytes_repr(
        total_array_size, total_array_bytes
    )

    # Add caption with grand totals
    grand_total_size = total_param_size + total_array_size
    grand_total_bytes = total_param_bytes + total_array_bytes
    
    rich_table.caption_style = 'bold'
    rich_table.caption = (
        f'\nTotal Parameters: {_size_and_bytes_repr(total_param_size, total_param_bytes)}\n'
        f'Total Arrays: {_size_and_bytes_repr(total_array_size, total_array_bytes)}\n'
        f'Grand Total: {_size_and_bytes_repr(grand_total_size, grand_total_bytes)}'
    )

    return '\n' + _get_rich_repr(rich_table, console_kwargs) + '\n'


def tabulate(
    module: eqx.Module,
    depth: Optional[int] = None,
    console_kwargs: Optional[Dict[str, Any]] = None,
    table_kwargs: Optional[Dict[str, Any]] = None,
    column_kwargs: Optional[Dict[str, Any]] = None,
    compute_flops: bool = False,
) -> Callable[..., str]:
    """Returns a function that creates a summary of the Equinox Module as a table.

    This function returns a function of the form `(*args, **kwargs) -> str` where 
    `*args` and `**kwargs` are passed to the module during the forward pass.

    `tabulate` uses `jax.eval_shape` under the hood to run the forward computation
    without consuming any FLOPs or allocating memory.

    Args:
        module: The Equinox module to tabulate.
        depth: Controls how many submodule levels deep the summary can go. 
            By default it's `None` which means no limit.
        console_kwargs: An optional dictionary with additional keyword arguments
            that are passed to `rich.console.Console` when rendering the table.
        table_kwargs: An optional dictionary with additional keyword arguments that
            are passed to `rich.table.Table` constructor.
        column_kwargs: An optional dictionary with additional keyword arguments that
            are passed to `rich.table.Table.add_column` when adding columns.
        compute_flops: Whether to include a `flops` column in the table listing the
            estimated FLOPs cost of the module forward pass.

    Returns:
        A function that accepts the same `*args` and `**kwargs` of the forward pass
        and returns a string with a tabular representation of the Module.

    Example:
        >>> import equinox as eqx
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        
        >>> class MLP(eqx.Module):
        ...     layers: list
        ...     
        ...     def __init__(self, key, input_size=10, hidden_size=20, output_size=2):
        ...         keys = jr.split(key, 2)
        ...         self.layers = [
        ...             eqx.nn.Linear(input_size, hidden_size, key=keys[0]),
        ...             eqx.nn.Linear(hidden_size, output_size, key=keys[1])
        ...         ]
        ...     
        ...     def __call__(self, x):
        ...         for layer in self.layers[:-1]:
        ...             x = jax.nn.relu(layer(x))
        ...         return self.layers[-1](x)
        
        >>> model = MLP(jr.PRNGKey(0))
        >>> x = jnp.ones((32, 10))
        >>> tabulate_fn = tabulate(model, compute_flops=True)
        >>> print(tabulate_fn(x))
    """

    def _tabulate_fn(*args, **kwargs):
        table_fn = _get_module_table(
            module,
            depth=depth,
            compute_flops=compute_flops,
        )

        table = table_fn(*args, **kwargs)

        return _render_table(
            table, 
            console_kwargs, 
            table_kwargs, 
            column_kwargs,
            compute_flops
        )

    return _tabulate_fn


if __name__ == "__main__":
    import ase.build
    from nequix.data import preprocess_graph, dict_to_graphstuple, atomic_numbers_to_indices
    import jraph
    from nequix.model import load_model

    def create_batch(size: int, config: dict) -> jraph.GraphsTuple:
        atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
        atoms = atoms.repeat((size, size, size))
        atomic_indices = atomic_numbers_to_indices(config["atomic_numbers"])
        cutoff = config["cutoff"]

        graph = preprocess_graph(atoms, atomic_indices, cutoff, targets=False)
        graph = dict_to_graphstuple(graph)
        batch = jraph.pad_with_graphs(graph, n_node=graph.n_node + 1, n_edge=graph.n_edge)
        return batch

    model, config = load_model("./models/nequix-mp-1.nqx")
    batch = create_batch(3, config)
    tabulate_fn = tabulate(model, compute_flops=True)
    print(tabulate_fn(batch))