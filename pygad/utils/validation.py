import numpy
import random
import warnings
import inspect
import logging

class Validation:
    
    def _validate_header(self, logger, random_seed, suppress_warnings,
                         mutation_by_replacement, sample_size, allow_duplicate_genes):
        """Set up logging and validate flags, sample size, and the random seed.

        Random generators are created after the remaining parameter checks,
        so rejected configurations do not sample values or run constraints.
        """
        self.valid_parameters = False
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger instance or None.")
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        if logger is None and not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        self.suppress_warnings = self._validate_boolean_parameter(suppress_warnings, 'suppress_warnings')
        self.mutation_by_replacement = self._validate_boolean_parameter(mutation_by_replacement, 'mutation_by_replacement')
        self.allow_duplicate_genes = self._validate_boolean_parameter(allow_duplicate_genes, 'allow_duplicate_genes')
        self.sample_size = self._validate_integer_parameter(sample_size, 'sample_size', minimum=1)
        self.random_seed = (None if random_seed is None else
                            self._validate_integer_parameter(random_seed, 'random_seed', minimum=0, maximum=2**32 - 1))

    def _validate_boolean_parameter(self, value, parameter_name):
        """Return a boolean setting or raise an error naming the parameter."""
        if type(value) is not bool:
            raise TypeError(f"{parameter_name} must be a bool, but {type(value).__name__} found.")
        return value

    def _validate_integer_parameter(self, value, parameter_name, minimum=None, maximum=None):
        """Validate a count and return a Python integer before any arithmetic."""
        if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, (int, numpy.integer)):
            raise TypeError(f"{parameter_name} must be an integer, but {type(value).__name__} found.")
        value = int(value)
        if minimum is not None and value < minimum:
            raise ValueError(f"{parameter_name} must be >= {minimum}, but {value} found.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{parameter_name} must be <= {maximum}, but {value} found.")
        return value

    def _validate_numeric_parameter(self, value, parameter_name, minimum=None, maximum=None):
        """Validate a finite real number and normalize NumPy scalar values."""
        if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, (int, float, numpy.integer, numpy.floating)):
            raise TypeError(f"{parameter_name} must be numeric, but {type(value).__name__} found.")
        if isinstance(value, numpy.integer):
            value = int(value)
        elif isinstance(value, numpy.floating):
            value = float(value)
        if isinstance(value, float) and not numpy.isfinite(value):
            raise ValueError(f"{parameter_name} must be finite, but {value} found.")
        if minimum is not None and value < minimum:
            raise ValueError(f"{parameter_name} must be >= {minimum}, but {value} found.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{parameter_name} must be <= {maximum}, but {value} found.")
        return value

    def _validate_callable_parameter(self, function, parameter_name, num_arguments):
        """Check the positional call PyGAD makes without invoking user code.

        Signature binding supports functions, bound methods, callable objects,
        partial functions, and additional optional parameters. Counting names
        alone would also accept required keyword-only parameters incorrectly.
        """
        if (not callable(function) or inspect.isclass(function) or inspect.iscoroutinefunction(function)
                or inspect.iscoroutinefunction(getattr(function, '__call__', None))):
            raise TypeError(f"{parameter_name} must be a synchronous callable.")
        try:
            signature = inspect.signature(function)
            signature.bind(*([None] * num_arguments))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{parameter_name} must accept {num_arguments} positional arguments: {error}") from error
        return function

    def _resolve_operator(self, operator, parameter_name, built_in_operators, num_arguments, allow_none=False):
        """Resolve a built-in name or validate a user-defined operator once."""
        if operator is None and allow_none:
            return None, None
        if isinstance(operator, str):
            operator = operator.lower()
            if operator not in built_in_operators:
                raise TypeError(f"Unknown {parameter_name} '{operator}'. Supported names are {list(built_in_operators)}.")
            return operator, getattr(self, built_in_operators[operator])
        return operator, self._validate_callable_parameter(operator, parameter_name, num_arguments)

    def _validate_range_parameters(self, lower, upper, lower_name, upper_name):
        """Copy scalar or per-gene bounds, validating their shape and values."""
        sequences = (list, tuple, numpy.ndarray)
        for value, name in [(lower, lower_name), (upper, upper_name)]:
            if isinstance(value, numpy.ndarray) and value.ndim == 0:
                raise ValueError(f'{name} must be a numeric scalar or a 1D sequence, not a 0D array.')
        if isinstance(lower, sequences) and isinstance(upper, sequences):
            result = []
            for values, name in [(lower, lower_name), (upper, upper_name)]:
                if numpy.asarray(values, dtype=object).ndim != 1 or len(values) != self.num_genes:
                    raise ValueError(f"{name} must be a 1D sequence with length equal to num_genes ({self.num_genes}).")
                result.append([self._validate_numeric_parameter(value, name) for value in values])
            return tuple(result)
        if isinstance(lower, sequences) or isinstance(upper, sequences):
            raise TypeError(f"{lower_name} and {upper_name} must both be numeric or both be per-gene sequences.")
        lower = self._validate_numeric_parameter(lower, lower_name)
        upper = self._validate_numeric_parameter(upper, upper_name)
        if lower == upper and not self.suppress_warnings:
            warnings.warn(f"The values of {lower_name} and {upper_name} are equal, so sampling uses a fixed value.")
        return lower, upper

    def _copy_parameter_container(self, value):
        """Copy parameter containers recursively, retaining numeric values and callables."""
        if isinstance(value, dict):
            return {key: self._copy_parameter_container(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, numpy.ndarray)):
            if isinstance(value, numpy.ndarray) and value.ndim == 0:
                raise ValueError('Parameter arrays must be sequences, not 0D arrays.')
            return [self._copy_parameter_container(item) for item in value]
        return value

    def _validate_gene_space(self,
                             gene_space):
        """
        Validate the ``gene_space`` parameter and store it on the GA
        instance. ``gene_space`` may be None, a flat iterable that
        applies to every gene, a per-gene nested iterable, or a dict
        with ``low`` / ``high`` (and optional ``step``) keys that
        describes a continuous range.

        Sets ``self.gene_space`` and the helper flag
        ``self.gene_space_nested`` (True when each gene has its own
        space).

        Parameters
        ----------
        gene_space : None, list, tuple, numpy.ndarray, or dict
            See the constructor documentation for the full grammar.

        Raises
        ------
        TypeError
            If ``gene_space`` is not one of the supported container
            types.
        ValueError
            If a nested gene space has an unsupported element type,
            or if a dict gene space is missing required keys.
        """
        gene_space = self._copy_parameter_container(gene_space)
        # Validate gene_space
        self.gene_space_nested = False
        if type(gene_space) is type(None):
            pass
        elif type(gene_space) is range:
            if self._finite_gene_space_length(gene_space) == 0:
                self.valid_parameters = False
                raise ValueError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
        elif type(gene_space) in [list, tuple, numpy.ndarray]:
            if len(gene_space) == 0:
                self.valid_parameters = False
                raise ValueError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
            else:
                for index, el in enumerate(gene_space):
                    if isinstance(el, range):
                        if self._finite_gene_space_length(el) == 0:
                            raise ValueError(f'The gene_space range at index {index} cannot be empty.')
                        self.gene_space_nested = True
                        continue
                    if type(el) in [numpy.ndarray, list, tuple, range]:
                        if len(el) == 0:
                            self.valid_parameters = False
                            raise ValueError(f"The element indexed {index} of 'gene_space' with type {type(el)} cannot be empty (i.e. its length must be >= 0).")
                        else:
                            for val in el:
                                if not (type(val) in [type(None)] + self.supported_int_float_types):
                                    raise TypeError(f"All values in the sublists inside the 'gene_space' attribute must be numeric of type int/float/None but ({val}) of type {type(val)} found.")
                        self.gene_space_nested = True
                    elif type(el) == type(None):
                        pass
                    elif type(el) is dict:
                        self._validate_gene_space_dictionary(el)
                        self.gene_space_nested = True
                    elif not (type(el) in self.supported_int_float_types):
                        self.valid_parameters = False
                        raise TypeError(f"Unexpected type {type(el)} for the element indexed {index} of 'gene_space'. The accepted types are list/tuple/range/numpy.ndarray of numbers, a single number (int/float), or None.")

        elif type(gene_space) is dict:
            self._validate_gene_space_dictionary(gene_space)

        else:
            self.valid_parameters = False
            raise TypeError(f"The expected type of 'gene_space' is list, range, or numpy.ndarray but {type(gene_space)} found.")

        self.gene_space = gene_space

    def _validate_gene_space_dictionary(self, space):
        """Validate range bounds and an optional step in a gene-space dict."""
        if set(space) not in [{'low', 'high'}, {'low', 'high', 'step'}]:
            self.valid_parameters = False
            raise ValueError("A gene_space dictionary must have 'low' and 'high' keys and may also have 'step'.")
        for name, value in space.items():
            space[name] = self._validate_numeric_parameter(value, f"gene_space '{name}'")
        if 'step' not in space:
            space['low'], space['high'] = sorted([space['low'], space['high']])
        if 'step' in space:
            if (space['step'] == 0
                    or (space['step'] > 0 and space['high'] <= space['low'])
                    or (space['step'] < 0 and space['high'] >= space['low'])):
                self.valid_parameters = False
                raise ValueError("The step in a gene_space dictionary must be non-zero and lead from low towards high so the space is not empty.")

    def _validate_init_range(self, init_range_low, init_range_high, num_genes):
        """Validate initialization bounds using the resolved population dimensions."""
        self.init_range_low, self.init_range_high = self._validate_range_parameters(
            init_range_low, init_range_high, 'init_range_low', 'init_range_high')

    def _validate_gene_type(self, gene_type, num_genes):
        """
        Normalize one type, a [type, precision] pair, or a per-gene
        specification into independent [type, precision] lists. A pair
        such as [float, int] specifies two gene types, while [float, 2]
        specifies one floating-point type with decimal precision.
        """
        if any(gene_type is supported_type for supported_type in self.supported_int_float_types):
            self.gene_type = [gene_type, None]
            self.gene_type_single = True
            return
        if type(gene_type) not in [list, tuple, numpy.ndarray]:
            self.valid_parameters = False
            raise TypeError("gene_type must be a supported numeric type, list, tuple, or NumPy array.")
        if isinstance(gene_type, numpy.ndarray) and gene_type.ndim == 0:
            self.valid_parameters = False
            raise ValueError("gene_type must contain a type or a sequence of types, not a 0D NumPy array.")

        specification = list(gene_type)
        first_is_type = len(specification) > 0 and any(
            specification[0] is supported_type for supported_type in self.supported_int_float_types)
        second_is_type = len(specification) == 2 and any(
            specification[1] is supported_type for supported_type in self.supported_int_float_types)
        second_is_sequence = len(specification) == 2 and type(specification[1]) in [list, tuple, numpy.ndarray]
        if len(specification) == 2 and first_is_type and not second_is_type and not second_is_sequence:
            self.gene_type = self._normalize_gene_type_entry(specification, 'gene_type')
            self.gene_type_single = True
        else:
            if len(specification) != num_genes:
                self.valid_parameters = False
                raise ValueError(f"When gene_type specifies a type for each gene, its length ({len(specification)}) must equal the number of genes ({num_genes}).")
            self.gene_type = [self._normalize_gene_type_entry(entry, f'gene_type at index {gene_index}')
                              for gene_index, entry in enumerate(specification)]
            self.gene_type_single = False

    def _normalize_gene_type_entry(self, specification, parameter_name):
        """Return an independent type/precision pair for one specification."""
        if any(specification is supported_type for supported_type in self.supported_int_float_types):
            return [specification, None]
        if type(specification) not in [list, tuple, numpy.ndarray]:
            self.valid_parameters = False
            raise TypeError(f"{parameter_name} must be a supported numeric type or a [type, precision] pair.")
        if isinstance(specification, numpy.ndarray) and specification.ndim != 1:
            self.valid_parameters = False
            raise ValueError(f"A type/precision pair in {parameter_name} must be a 1D sequence of 2 elements.")
        if len(specification) != 2:
            self.valid_parameters = False
            raise ValueError(f"A type/precision pair in {parameter_name} must have 2 elements.")
        gene_dtype, precision = specification
        if not any(gene_dtype is supported_type for supported_type in self.supported_int_float_types):
            self.valid_parameters = False
            raise TypeError(f"The data type in {parameter_name} must be a supported numeric type.")
        if precision is not None:
            if gene_dtype not in self.supported_float_types:
                self.valid_parameters = False
                raise ValueError(f"Integers cannot have precision in {parameter_name}. Use the integer type directly or pair it with None.")
            if type(precision) not in self.supported_int_types or type(precision) is object:
                self.valid_parameters = False
                raise TypeError(f"The precision in {parameter_name} must be an integer or None.")
            precision = int(precision)
        return [gene_dtype, precision]

    def _validate_initial_population_shape(self, initial_population, sol_per_pop, num_genes):
        """
        Validate the population dimensions before any per-gene settings.
        A supplied population determines both dimensions, regardless of
        the values passed to ``sol_per_pop`` and ``num_genes``. Return an
        independent object array so mixed numeric values remain exact.
        """
        if initial_population is None:
            if sol_per_pop is None or num_genes is None:
                self.valid_parameters = False
                raise TypeError("When initial_population is None, both sol_per_pop and num_genes must be specified.")
            sol_per_pop = self._validate_integer_parameter(sol_per_pop, 'sol_per_pop', minimum=1)
            num_genes = self._validate_integer_parameter(num_genes, 'num_genes', minimum=1)
            population = None
        else:
            if type(initial_population) not in [list, tuple, numpy.ndarray]:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the 'initial_population' parameter is expected to be of type list, tuple, or ndarray but {type(initial_population)} found.")
            try:
                population = numpy.array(initial_population, dtype=object, copy=True)
            except ValueError as error:
                self.valid_parameters = False
                raise ValueError("initial_population must be a rectangular 2D list, tuple, or NumPy array.") from error
            if population.ndim != 2 or 0 in population.shape:
                self.valid_parameters = False
                raise ValueError("initial_population must be a non-empty rectangular 2D list, tuple, or NumPy array.")
            for value in population.flat:
                if type(value) not in self.supported_int_float_types:
                    self.valid_parameters = False
                    raise TypeError(f"The values in the initial population can be integers or floats but the value ({value}) of type {type(value)} found.")
            sol_per_pop, num_genes = population.shape

        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.pop_size = (sol_per_pop, num_genes)
        return population

    def _build_initial_population(self, initial_population):
        """
        Store a generated or supplied population after applying gene types,
        constraints, and duplicate repair. Dimensions and numeric values
        are validated before this method is called. Supplied values are
        preserved even outside the generation range or gene space; only
        replacements use the configured generation settings.
        """
        if initial_population is None:
            self.initialize_population(self.allow_duplicate_genes, self.gene_type, self.gene_constraint)
        else:
            self.population = self.prepare_initial_population(initial_population)
            # Keep separate arrays so evolution cannot modify this snapshot.
            self.initial_population = self.population.copy()

    def _validate_mutation_range(self, random_mutation_min_val, random_mutation_max_val):
        """Validate and copy the scalar or per-gene random mutation bounds."""
        self.random_mutation_min_val, self.random_mutation_max_val = self._validate_range_parameters(
            random_mutation_min_val, random_mutation_max_val,
            'random_mutation_min_val', 'random_mutation_max_val')

    def _validate_gene_constraint(self, gene_constraint):
        """Copy one optional constraint per gene and validate its positional call."""
        if gene_constraint is None:
            self.gene_constraint = None
            return
        if not isinstance(gene_constraint, (list, tuple)):
            raise TypeError("gene_constraint must be a list or tuple, or None.")
        if len(gene_constraint) != self.num_genes:
            raise ValueError(f"The number of constraints ({len(gene_constraint)}) must equal num_genes ({self.num_genes}).")
        self.gene_constraint = [None if constraint is None else
                                self._validate_callable_parameter(constraint, f'gene_constraint at index {index}', 2)
                                for index, constraint in enumerate(gene_constraint)]

    def _validate_crossover(self, crossover_type, crossover_probability, sbx_crossover_eta=30):
        """Resolve crossover and validate its probability and finite distribution index."""
        operators = {'single_point': 'single_point_crossover', 'two_points': 'two_points_crossover',
                     'uniform': 'uniform_crossover', 'scattered': 'scattered_crossover', 'sbx': 'sbx_crossover'}
        self.crossover_type, self.crossover = self._resolve_operator(crossover_type, 'crossover_type', operators, 3, allow_none=True)
        self.sbx_crossover_eta = self._validate_numeric_parameter(sbx_crossover_eta, 'sbx_crossover_eta')
        if self.sbx_crossover_eta <= 0:
            raise ValueError('sbx_crossover_eta must be positive.')
        self.crossover_probability = (None if crossover_probability is None else
                                      self._validate_numeric_parameter(crossover_probability, 'crossover_probability', 0, 1))

    def _validate_mutation(self, mutation_type, mutation_probability, mutation_num_genes,
                           mutation_percent_genes, polynomial_mutation_eta=20):
        """Resolve the active mutation control before validating its values.

        Probability takes precedence over a gene count, which takes precedence
        over a percentage. Permutation and polynomial operators keep their
        historical defaults when none of these controls is explicitly set.
        """
        operators = {'random': 'random_mutation', 'swap': 'swap_mutation', 'inversion': 'inversion_mutation',
                     'scramble': 'scramble_mutation', 'adaptive': 'adaptive_mutation', 'polynomial': 'polynomial_mutation'}
        self.mutation_type, self.mutation = self._resolve_operator(mutation_type, 'mutation_type', operators, 2, allow_none=True)
        self.polynomial_mutation_eta = self._validate_numeric_parameter(polynomial_mutation_eta, 'polynomial_mutation_eta')
        if self.polynomial_mutation_eta <= 0:
            raise ValueError('polynomial_mutation_eta must be positive.')
        self.mutation_probability = None
        self.mutation_control_explicitly_set = (mutation_probability is not None or mutation_num_genes is not None or
                                                not (isinstance(mutation_percent_genes, str) and mutation_percent_genes == 'default'))
        if self.mutation_type is None:
            if self.crossover_type is None and not self.suppress_warnings:
                warnings.warn('Crossover and mutation are disabled, so the initial population cannot evolve.')
            return None, 'default'
        adaptive = self.mutation_type == 'adaptive'
        if mutation_probability is not None:
            self.mutation_probability = self._validate_mutation_control(mutation_probability, 'mutation_probability', adaptive, 0, 1)
            mutation_num_genes, mutation_percent_genes = None, 'default'
        elif mutation_num_genes is not None:
            mutation_num_genes = self._validate_mutation_control(mutation_num_genes, 'mutation_num_genes', adaptive, 1, self.num_genes, integer=True)
            mutation_percent_genes = 'default'
        else:
            if isinstance(mutation_percent_genes, str) and mutation_percent_genes == 'default':
                if adaptive:
                    raise TypeError("Adaptive mutation requires a pair of probabilities, gene counts, or percentages.")
                mutation_percent_genes = 10
            mutation_percent_genes = self._validate_mutation_control(mutation_percent_genes, 'mutation_percent_genes', adaptive, 0, 100)
            percentages = mutation_percent_genes if adaptive else [mutation_percent_genes]
            counts = []
            for percentage in percentages:
                if percentage <= 0:
                    raise ValueError('mutation_percent_genes must be > 0 and <= 100.')
                count = int(percentage * self.num_genes / 100)
                if count == 0:
                    count = 1
                    if not self.suppress_warnings:
                        warnings.warn('mutation_percent_genes selects fewer than one gene. mutation_num_genes is set to 1.')
                counts.append(count)
            mutation_num_genes = counts if adaptive else counts[0]
        if self.mutation_by_replacement and self.mutation_type not in ('random', 'adaptive') and not self.suppress_warnings:
            warnings.warn('mutation_by_replacement applies only to random and adaptive mutation.')
        if self.crossover_type is None and self.mutation_type is None and not self.suppress_warnings:
            warnings.warn('Crossover and mutation are disabled, so the initial population cannot evolve.')
        return mutation_num_genes, mutation_percent_genes

    def _validate_mutation_control(self, value, parameter_name, adaptive, minimum, maximum, integer=False):
        """Validate a scalar setting or the two rates used by adaptive mutation."""
        if adaptive:
            if not isinstance(value, (list, tuple, numpy.ndarray)) or numpy.asarray(value, dtype=object).shape != (2,):
                raise ValueError(f"{parameter_name} must be a 1D sequence of two values for adaptive mutation.")
            values = list(value)
        else:
            values = [value]
        validator = self._validate_integer_parameter if integer else self._validate_numeric_parameter
        values = [validator(item, parameter_name, minimum, maximum) for item in values]
        if adaptive and values[0] < values[1] and not self.suppress_warnings:
            warnings.warn(f'The first {parameter_name} value is smaller than the second, so high-quality solutions mutate more frequently.')
        return values if adaptive else values[0]

    def _validate_nsga3_num_divisions(self, parent_selection_type, nsga3_num_divisions):
        """Validate the division count when an NSGA-III operator uses it."""
        if parent_selection_type in ('nsga3', 'tournament_nsga3'):
            if nsga3_num_divisions is None:
                raise ValueError('NSGA-III requires nsga3_num_divisions to be a positive integer.')
            nsga3_num_divisions = self._validate_integer_parameter(nsga3_num_divisions, 'nsga3_num_divisions', minimum=1)
        self.nsga3_num_divisions = nsga3_num_divisions

    def _validate_parent_selection(self, parent_selection_type, K_tournament,
                                   keep_parents, keep_elitism, nsga3_num_divisions=None):
        """Resolve selection and validate tournament size and retained solutions."""
        operators = {'sss': 'steady_state_selection', 'rws': 'roulette_wheel_selection',
                     'sus': 'stochastic_universal_selection', 'random': 'random_selection',
                     'tournament': 'tournament_selection', 'tournament_nsga2': 'tournament_selection_nsga2',
                     'nsga2': 'nsga2_selection', 'tournament_nsga3': 'tournament_selection_nsga3',
                     'nsga3': 'nsga3_selection', 'rank': 'rank_selection'}
        parent_selection_type, self.select_parents = self._resolve_operator(parent_selection_type, 'parent_selection_type', operators, 3)
        if parent_selection_type in ('tournament', 'tournament_nsga2', 'tournament_nsga3'):
            K_tournament = self._validate_integer_parameter(K_tournament, 'K_tournament', minimum=1)
            if K_tournament > self.sol_per_pop:
                if not self.suppress_warnings:
                    warnings.warn(f'K_tournament is clipped to sol_per_pop ({self.sol_per_pop}).')
                K_tournament = self.sol_per_pop
        self.K_tournament = int(K_tournament) if isinstance(K_tournament, numpy.integer) else K_tournament
        self._validate_nsga3_num_divisions(parent_selection_type, nsga3_num_divisions)
        self.keep_parents_explicitly_set = keep_parents is not None
        self.keep_parents = self._validate_integer_parameter(-1 if keep_parents is None else keep_parents,
                                                            'keep_parents', -1, self.num_parents_mating)
        self.keep_elitism = self._validate_integer_parameter(keep_elitism, 'keep_elitism', 0, self.sol_per_pop)
        if self.keep_parents_explicitly_set and self.keep_elitism > 0 and not self.suppress_warnings:
            warnings.warn(f'keep_elitism (={self.keep_elitism}) takes precedence over keep_parents (={self.keep_parents}). Set keep_elitism=0 to retain parents instead.')
        self._refresh_num_offspring()
        return parent_selection_type

    def _refresh_num_offspring(self):
        """
        Set self.num_offspring from the current values of sol_per_pop,
        keep_elitism, keep_parents, and num_parents_mating. Called from
        the initial validation step and again whenever the population
        size changes after construction (for example, when NSGA-III grows
        sol_per_pop to match the number of reference points).
        """
        if self.keep_elitism == 0:
            if self.keep_parents == -1:
                self.num_offspring = self.sol_per_pop - self.num_parents_mating
            elif self.keep_parents == 0:
                self.num_offspring = self.sol_per_pop
            elif self.keep_parents > 0:
                self.num_offspring = self.sol_per_pop - self.keep_parents
        else:
            self.num_offspring = self.sol_per_pop - self.keep_elitism

    def _validate_fitness_func(self, fitness_func, fitness_batch_size):
        """Validate the fitness call and optional batch size before sampling."""
        self.fitness_func = self._validate_callable_parameter(fitness_func, 'fitness_func', 3)
        self.fitness_batch_size = (None if fitness_batch_size is None else
                                   self._validate_integer_parameter(fitness_batch_size, 'fitness_batch_size', 1, self.sol_per_pop))

    def _validate_callbacks(self, on_start, on_fitness, on_parents, on_crossover,
                            on_mutation, on_generation, on_stop):
        """Validate each lifecycle callback using the arguments PyGAD supplies."""
        callbacks = [('on_start', on_start, 1), ('on_fitness', on_fitness, 2),
                     ('on_parents', on_parents, 2), ('on_crossover', on_crossover, 2),
                     ('on_mutation', on_mutation, 2), ('on_generation', on_generation, 1),
                     ('on_stop', on_stop, 2)]
        for name, function, num_arguments in callbacks:
            setattr(self, name, None if function is None else self._validate_callable_parameter(function, name, num_arguments))

    def _validate_stop_criteria(self, stop_criteria):
        """Parse stopping criteria once, preserving order while removing duplicates."""
        self.supported_stop_words = ['reach', 'saturate', 'time', 'evaluations']
        if stop_criteria is None:
            self.stop_criteria = None
            return
        if isinstance(stop_criteria, str):
            criteria = [stop_criteria]
        elif isinstance(stop_criteria, (list, tuple, numpy.ndarray)):
            if numpy.asarray(stop_criteria, dtype=object).ndim != 1:
                raise ValueError('stop_criteria must be a 1D sequence of strings.')
            criteria = list(stop_criteria)
        else:
            raise TypeError('stop_criteria must be a string, a sequence of strings, or None.')
        self.stop_criteria = []
        seen = set()
        for criterion in criteria:
            if not isinstance(criterion, str):
                raise TypeError('Each stop_criteria entry must be a string.')
            if criterion not in seen:
                self.stop_criteria.append(self._parse_stop_criterion(criterion))
                seen.add(criterion)

    def _parse_stop_criterion(self, criterion):
        """Parse a finite threshold or an exact positive integer count."""
        from decimal import Decimal, InvalidOperation
        parts = criterion.split('_')
        word, numbers = parts[0], parts[1:]
        if word not in self.supported_stop_words:
            raise ValueError(f'Unknown stop criterion {word!r}. Supported words are {self.supported_stop_words}.')
        if not numbers or (word != 'reach' and len(numbers) != 1):
            raise ValueError('A stop criterion has the form word_number; only reach accepts multiple thresholds.')
        result = [word]
        for number in numbers:
            try:
                value = Decimal(number)
            except InvalidOperation:
                raise ValueError(f'The threshold in stop_criteria must be numeric, but {number!r} found.') from None
            if not value.is_finite():
                raise ValueError('Stop criterion thresholds must be finite.')
            if word in ('saturate', 'evaluations'):
                if value <= 0 or value != value.to_integral_value():
                    raise ValueError(f'{word} requires a positive integer count.')
                result.append(int(value))
            else:
                if word == 'time' and value < 0:
                    raise ValueError('time requires a non-negative number of seconds.')
                threshold = float(value)
                if not numpy.isfinite(threshold):
                    raise ValueError('Stop criterion thresholds must fit a finite float.')
                result.append(threshold)
        return result

    def _validate_parallel_processing(self, parallel_processing):
        """Normalize the executor mode and an optional integer worker count."""
        if parallel_processing is None:
            self.parallel_processing = None
            return
        if isinstance(parallel_processing, (list, tuple)):
            if len(parallel_processing) != 2:
                raise ValueError('parallel_processing must contain a mode and a worker count.')
            mode, workers = parallel_processing
            if not isinstance(mode, str) or mode not in ('thread', 'process'):
                raise ValueError("The parallel_processing mode must be 'thread' or 'process'.")
            workers = None if workers is None else self._validate_integer_parameter(workers, 'parallel_processing worker count', minimum=0)
        else:
            mode = 'thread'
            workers = self._validate_integer_parameter(parallel_processing, 'parallel_processing', minimum=0)
        self.parallel_processing = None if workers == 0 else [mode, workers]

    def _validate_footer(self,
                         num_generations,
                         parent_selection_type,
                         mutation_percent_genes,
                         mutation_num_genes,
                         save_best_solutions,
                         save_solutions):
        """
        Validate the last group of parameters and store them on the
        GA instance: ``num_generations``, ``save_best_solutions``,
        and ``save_solutions``. Store the already validated mutation
        controls and initialize the lifecycle state before population creation.

        Parameters
        ----------
        num_generations : int
            Number of generations to evolve.
        parent_selection_type : str or callable
            The selection operator name (used for context-specific
            warnings).
        mutation_percent_genes : numeric or 'default'
            Percentage of genes to mutate, kept for back-compatibility.
        mutation_num_genes : int, list, tuple, or None
            Number of genes to mutate per solution, kept for the
            same reason.
        save_best_solutions : bool
            If True, the best solution of every generation is saved
            in ``self.best_solutions``.
        save_solutions : bool
            If True, every solution of every generation is saved in
            ``self.solutions``.

        Raises
        ------
        TypeError
            If ``num_generations`` is not an integer, or
            ``save_best_solutions`` / ``save_solutions`` is not a
            bool.
        ValueError
            If ``num_generations`` is negative.
        """

        self.num_generations = self._validate_integer_parameter(num_generations, 'num_generations', minimum=0)
        self.save_best_solutions = self._validate_boolean_parameter(save_best_solutions, 'save_best_solutions')
        self.save_solutions = self._validate_boolean_parameter(save_solutions, 'save_solutions')
        if not self.suppress_warnings:
            if save_best_solutions:
                warnings.warn('Use save_best_solutions with caution as saving large histories can cause memory overflow.')
            if save_solutions:
                warnings.warn('Use save_solutions with caution as saving large histories can cause memory overflow.')

        # Set the `run_completed` property to False. It is set to `True` only after the `run()` method is complete.
        self.run_completed = False

        # The number of completed generations.
        self.generations_completed = 0

        # Counts how many times the fitness function was called inside
        # the current run(). Used by the "evaluations_<N>" stop
        # criterion. Reset to 0 at the start of each run() call.
        self.num_fitness_evaluations = 0
        # Time at which the current run() call started. Used by the
        # "time_<seconds>" stop criterion. None outside of run().
        self.run_start_time = None

        # Parameters of the genetic algorithm.
        self.parent_selection_type = parent_selection_type

        # Parameters of the mutation operation.
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_num_genes = mutation_num_genes

        # Even though this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        # A list holding the fitness value of the best solution for each generation.
        self.best_solutions_fitness = []

        # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.
        self.best_solution_generation = -1

        self.save_best_solutions = save_best_solutions
        self.best_solutions = []  # Holds the best solution in each generation.

        self.save_solutions = save_solutions
        self.solutions = []  # Holds the solutions in each generation.
        # Holds the fitness of the solutions in each generation.
        self.solutions_fitness = []

        # A list holding the fitness values of all solutions in the last generation.
        self.last_generation_fitness = None
        # A list holding the parents of the last generation.
        self.last_generation_parents = None
        # A list holding the offspring after applying crossover in the last generation.
        self.last_generation_offspring_crossover = None
        # A list holding the offspring after applying mutation in the last generation.
        self.last_generation_offspring_mutation = None
        # Holds the fitness values of one generation before the fitness values saved in the last_generation_fitness attribute. Added in PyGAD 2.16.2.
        self.previous_generation_fitness = None
        # Added in PyGAD 2.18.0. A NumPy array holding the elitism of the current generation according to the value passed in the 'keep_elitism' parameter. It works only if the 'keep_elitism' parameter has a non-zero value.
        self.last_generation_elitism = None
        # Added in PyGAD 2.19.0. A NumPy array holding the indices of the elitism of the current generation. It works only if the 'keep_elitism' parameter has a non-zero value.
        self.last_generation_elitism_indices = None
        # Supported in PyGAD 3.2.0. It holds the pareto fronts when solving a multi-objective problem.
        self.pareto_fronts = None
    
    def validate_parameters(self,
                            num_generations,
                            num_parents_mating,
                            fitness_func,
                            fitness_batch_size,
                            initial_population,
                            sol_per_pop,
                            num_genes,
                            init_range_low,
                            init_range_high,
                            gene_type,
                            parent_selection_type,
                            keep_parents,
                            keep_elitism,
                            K_tournament,
                            nsga3_num_divisions,
                            crossover_type,
                            crossover_probability,
                            sbx_crossover_eta,
                            mutation_type,
                            mutation_probability,
                            polynomial_mutation_eta,
                            mutation_by_replacement,
                            mutation_percent_genes,
                            mutation_num_genes,
                            random_mutation_min_val,
                            random_mutation_max_val,
                            gene_space,
                            gene_constraint,
                            sample_size,
                            allow_duplicate_genes,
                            on_start,
                            on_fitness,
                            on_parents,
                            on_crossover,
                            on_mutation,
                            on_generation,
                            on_stop,
                            save_best_solutions,
                            save_solutions,
                            suppress_warnings,
                            stop_criteria,
                            parallel_processing,
                            random_seed,
                            logger):
        """
        Validate every parameter passed to ``pygad.GA.__init__`` and
        store the parsed values on the GA instance. This method is
        called from the constructor; users rarely need to call it
        directly.

        Validation is split into a sequence of smaller methods
        (``_validate_header``, ``_validate_gene_space``, etc.); see
        their docstrings for the details of each parameter.

        Sets ``self.valid_parameters = True`` when every check
        passes. When a check fails, the method sets
        ``self.valid_parameters = False`` and raises the appropriate
        exception so the caller never sees a partially-constructed
        instance.

        Raises
        ------
        TypeError, ValueError
            Propagated from the per-group validators when a parameter
            is of the wrong type or out of range.
        """

        self._validate_header(logger,
                              random_seed,
                              suppress_warnings,
                              mutation_by_replacement,
                              sample_size,
                              allow_duplicate_genes)

        # Establish dimensions first, especially when the supplied population
        # overrides sol_per_pop and num_genes.
        initial_population = self._validate_initial_population_shape(
            initial_population, sol_per_pop, num_genes)
        self._validate_gene_space(gene_space)
        self._validate_init_range(init_range_low, init_range_high, self.num_genes)
        self._validate_gene_type(gene_type, self.num_genes)
        self._validate_gene_constraint(gene_constraint)
        if self.gene_space_nested and len(gene_space) != self.num_genes:
            self.valid_parameters = False
            raise ValueError(f"When gene_space is nested, its length ({len(gene_space)}) must equal the number of genes ({self.num_genes}).")

        self._validate_mutation_range(random_mutation_min_val, random_mutation_max_val)
        self.num_parents_mating = self._validate_integer_parameter(
            num_parents_mating, 'num_parents_mating', 1, self.sol_per_pop)

        self._validate_crossover(crossover_type,
                                 crossover_probability,
                                 sbx_crossover_eta=sbx_crossover_eta)

        mutation_num_genes, mutation_percent_genes = self._validate_mutation(mutation_type,
                                                                             mutation_probability,
                                                                             mutation_num_genes,
                                                                             mutation_percent_genes,
                                                                             polynomial_mutation_eta=polynomial_mutation_eta)

        parent_selection_type = self._validate_parent_selection(parent_selection_type,
                                                                K_tournament,
                                                                keep_parents,
                                                                keep_elitism,
                                                                nsga3_num_divisions)

        self._validate_fitness_func(fitness_func,
                                    fitness_batch_size)

        self._validate_callbacks(on_start,
                                 on_fitness,
                                 on_parents,
                                 on_crossover,
                                 on_mutation,
                                 on_generation,
                                 on_stop)

        self._validate_stop_criteria(stop_criteria)

        self._validate_parallel_processing(parallel_processing)

        self._validate_footer(num_generations,
                              parent_selection_type,
                              mutation_percent_genes,
                              mutation_num_genes,
                              save_best_solutions,
                              save_solutions)

        self.numpy_random_generator = numpy.random.RandomState(self.random_seed)
        self.python_random_generator = random.Random(self.random_seed)
        self.gene_space_unpacked = self.unpack_gene_space(
            range_min=self.init_range_low, range_max=self.init_range_high)
        self._build_initial_population(initial_population)
        self.valid_parameters = True

    def validate_multi_stop_criteria(self, stop_word, number):
        """Compatibility helper for parsing multiple reach thresholds."""
        if stop_word != 'reach':
            raise ValueError('Only reach accepts multiple stop thresholds.')
        return self._parse_stop_criterion('_'.join([stop_word] + list(number)))[1:]
