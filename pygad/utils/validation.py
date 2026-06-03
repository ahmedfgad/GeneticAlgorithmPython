import numpy
import random
import warnings
import inspect
import logging

class Validation:
    
    def _validate_header(self,
                         logger,
                         random_seed,
                         suppress_warnings,
                         mutation_by_replacement,
                         sample_size,
                         allow_duplicate_genes):
        """
        Validate the first group of constructor parameters and store
        them on the GA instance. Sets up the logger (creating a
        default console logger when ``logger`` is None), seeds the
        random generators when ``random_seed`` is given, and persists
        the four flag-style parameters on ``self``.

        Parameters
        ----------
        logger : logging.Logger or None
            A logger object. When None, a default console logger is
            created.
        random_seed : int or None
            Seed for the numpy and random random generators. When
            None, the generators are left in their current state.
        suppress_warnings : bool
            If True, ``warnings.warn`` calls inside PyGAD are skipped.
        mutation_by_replacement : bool
            If True, the random mutation replaces the gene value
            instead of adding a random delta.
        sample_size : int
            Number of candidate values drawn when resolving gene
            constraints and duplicates.
        allow_duplicate_genes : bool
            If False, duplicate genes inside a single solution are
            resolved by sampling new values.

        Raises
        ------
        TypeError
            If ``logger`` is neither None nor a ``logging.Logger``.
        TypeError
            If any of the bool flags is not a bool.
        ValueError
            If ``sample_size`` is not a positive integer or
            ``random_seed`` is of an unsupported type.
        """
        # If no logger is passed, then create a logger that logs the messages only to the console.
        if logger is None:
            # Create a logger named with the module name.
            logger = logging.getLogger(__name__)
            # Set the logger log level to 'DEBUG' to log all kinds of messages.
            logger.setLevel(logging.DEBUG)

            # Clear any attached handlers to the logger from the previous runs.
            logger.handlers.clear()

            # Create the handlers.
            stream_handler = logging.StreamHandler()
            # Set the handler log level to 'DEBUG' to log all kinds of messages received from the logger.
            stream_handler.setLevel(logging.DEBUG)

            # Create the formatter that just includes the log message.
            formatter = logging.Formatter('%(message)s')

            # Add the formatter to the handler.
            stream_handler.setFormatter(formatter)

            # Add the handler to the logger.
            logger.addHandler(stream_handler)
        else:
            # Validate that the passed logger is of type 'logging.Logger'.
            if isinstance(logger, logging.Logger):
                pass
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'logger' parameter is 'logging.Logger' but {type(logger)} found.")

        # Create the 'self.logger' attribute to hold the logger.
        self.logger = logger

        self.random_seed = random_seed
        if random_seed is None:
            pass
        else:
            numpy.random.seed(self.random_seed)
            random.seed(self.random_seed)

        # If suppress_warnings is bool and its value is False, then print warning messages.
        if type(suppress_warnings) is bool:
            self.suppress_warnings = suppress_warnings
        else:
            self.valid_parameters = False
            raise TypeError(f"The expected type of the 'suppress_warnings' parameter is bool but {type(suppress_warnings)} found.")

        # Validating mutation_by_replacement
        if not (type(mutation_by_replacement) is bool):
            self.valid_parameters = False
            raise TypeError(f"The expected type of the 'mutation_by_replacement' parameter is bool but {type(mutation_by_replacement)} found.")

        self.mutation_by_replacement = mutation_by_replacement

        # Validate the sample_size parameter.
        if type(sample_size) in self.supported_int_types:
            if sample_size > 0:
                pass
            else:
                self.valid_parameters = False
                raise ValueError(f"The value of the sample_size parameter must be > 0 but the value ({sample_size}) found.")
        else:
            self.valid_parameters = False
            raise TypeError(f"The type of the sample_size parameter must be integer but the value ({sample_size}) of type {type(sample_size)} found.")

        self.sample_size = sample_size

        # Validate allow_duplicate_genes
        if not (type(allow_duplicate_genes) is bool):
            self.valid_parameters = False
            raise TypeError(f"The expected type of the 'allow_duplicate_genes' parameter is bool but {type(allow_duplicate_genes)} found.")

        self.allow_duplicate_genes = allow_duplicate_genes
    
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
        # Validate gene_space
        self.gene_space_nested = False
        if type(gene_space) is type(None):
            pass
        elif type(gene_space) is range:
            if len(gene_space) == 0:
                self.valid_parameters = False
                raise ValueError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
        elif type(gene_space) in [list, numpy.ndarray]:
            if len(gene_space) == 0:
                self.valid_parameters = False
                raise ValueError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
            else:
                for index, el in enumerate(gene_space):
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
                        if len(el.items()) == 2:
                            if ('low' in el.keys()) and ('high' in el.keys()):
                                pass
                            else:
                                self.valid_parameters = False
                                raise ValueError(f"When an element in the 'gene_space' parameter is of type dict, then it can have the keys 'low', 'high', and 'step' (optional) but the following keys found: {el.keys()}")
                        elif len(el.items()) == 3:
                            if ('low' in el.keys()) and ('high' in el.keys()) and ('step' in el.keys()):
                                pass
                            else:
                                self.valid_parameters = False
                                raise ValueError(f"When an element in the 'gene_space' parameter is of type dict, then it can have the keys 'low', 'high', and 'step' (optional) but the following keys found: {el.keys()}")
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"When an element in the 'gene_space' parameter is of type dict, then it must have only 2 items but ({len(el.items())}) items found.")
                        self.gene_space_nested = True
                    elif not (type(el) in self.supported_int_float_types):
                        self.valid_parameters = False
                        raise TypeError(f"Unexpected type {type(el)} for the element indexed {index} of 'gene_space'. The accepted types are list/tuple/range/numpy.ndarray of numbers, a single number (int/float), or None.")

        elif type(gene_space) is dict:
            if len(gene_space.items()) == 2:
                if ('low' in gene_space.keys()) and ('high' in gene_space.keys()):
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When the 'gene_space' parameter is of type dict, then it can have only the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space.keys()}")
            elif len(gene_space.items()) == 3:
                if ('low' in gene_space.keys()) and ('high' in gene_space.keys()) and ('step' in gene_space.keys()):
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When the 'gene_space' parameter is of type dict, then it can have only the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space.keys()}")
            else:
                self.valid_parameters = False
                raise ValueError(f"When the 'gene_space' parameter is of type dict, then it must have only 2 items but ({len(gene_space.items())}) items found.")

        else:
            self.valid_parameters = False
            raise TypeError(f"The expected type of 'gene_space' is list, range, or numpy.ndarray but {type(gene_space)} found.")

        self.gene_space = gene_space

    def _validate_init_range(self,
                             init_range_low,
                             init_range_high,
                             num_genes,
                             initial_population):
        """
        Validate the ``init_range_low`` and ``init_range_high``
        parameters used to build the initial population when the user
        does not pass one explicitly. Both may be a scalar (one range
        shared by every gene) or a per-gene iterable.

        Sets ``self.init_range_low`` and ``self.init_range_high`` on
        the GA instance.

        Parameters
        ----------
        init_range_low : numeric or iterable
            Lower bound(s) for the random initial gene values.
        init_range_high : numeric or iterable
            Upper bound(s) for the random initial gene values.
        num_genes : int or None
            Number of genes per solution. Used to check the length of
            the per-gene iterables.
        initial_population : list / numpy.ndarray or None
            The user-provided initial population, if any. Only used to
            skip the length check when the population is being
            inferred from it.

        Raises
        ------
        TypeError
            If either parameter is not a supported type.
        ValueError
            If the per-gene iterables have a length different from
            ``num_genes``.
        """
        # Validate init_range_low and init_range_high
        if type(init_range_low) in self.supported_int_float_types:
            if type(init_range_high) in self.supported_int_float_types:
                if init_range_low == init_range_high:
                    if not self.suppress_warnings:
                        warnings.warn("The values of the 2 parameters 'init_range_low' and 'init_range_high' are equal and this might return the same value for some genes in the initial population.")
            else:
                self.valid_parameters = False
                raise TypeError(f"Type mismatch between the 2 parameters 'init_range_low' {type(init_range_low)} and 'init_range_high' {type(init_range_high)}.")
        elif type(init_range_low) in [list, tuple, numpy.ndarray]:
            # Get the number of genes before validating the num_genes parameter.
            if num_genes is None:
                if initial_population is None:
                    self.valid_parameters = False
                    raise TypeError("When the parameter 'initial_population' is None, then the 2 parameters 'sol_per_pop' and 'num_genes' cannot be None too.")
                elif not len(init_range_low) == len(initial_population[0]):
                    self.valid_parameters = False
                    raise ValueError(f"The length of the 'init_range_low' parameter is {len(init_range_low)} which is different from the number of genes {len(initial_population[0])}.")
            elif not len(init_range_low) == num_genes:
                self.valid_parameters = False
                raise ValueError(f"The length of the 'init_range_low' parameter is {len(init_range_low)} which is different from the number of genes {num_genes}.")

            if type(init_range_high) in [list, tuple, numpy.ndarray]:
                if len(init_range_low) == len(init_range_high):
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"Size mismatch between the 2 parameters 'init_range_low' {len(init_range_low)} and 'init_range_high' {len(init_range_high)}.")

                # Validate the values in init_range_low
                for val in init_range_low:
                    if type(val) in self.supported_int_float_types:
                        pass
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'init_range_low' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")

                # Validate the values in init_range_high
                for val in init_range_high:
                    if type(val) in self.supported_int_float_types:
                        pass
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'init_range_high' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")
            else:
                self.valid_parameters = False
                raise TypeError(f"Type mismatch between the 2 parameters 'init_range_low' {type(init_range_low)} and 'init_range_high' {type(init_range_high)}. Both of them can be either numeric or iterable (list/tuple/numpy.ndarray).")
        else:
            self.valid_parameters = False
            raise TypeError(f"The expected type of the 'init_range_low' parameter is numeric or list/tuple/numpy.ndarray but {type(init_range_low)} found.")

        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
    
    def _validate_gene_type(self,
                            gene_type,
                            num_genes,
                            initial_population):
        """
        Validate the ``gene_type`` parameter and store it on the GA
        instance. A gene type may be:

        - a single Python or numpy numeric type that applies to every
          gene (``self.gene_type_single`` is set to True);
        - a ``[type, precision]`` pair applied to every gene;
        - a per-gene list of types or ``[type, precision]`` pairs
          (``self.gene_type_single`` is set to False).

        Parameters
        ----------
        gene_type : type, list, or tuple
            The gene type specification.
        num_genes : int or None
            Number of genes per solution. Used to check the length of
            a per-gene specification.
        initial_population : list / numpy.ndarray or None
            The user-provided initial population, if any. Used to
            decide whether ``num_genes`` is already known.

        Raises
        ------
        TypeError
            If ``gene_type`` (or any of its elements) is not a
            supported numeric type.
        ValueError
            If the per-gene specification has a length different from
            ``num_genes``, or the precision is not an integer.
        """
        # Validate gene_type
        if gene_type in self.supported_int_float_types:
            self.gene_type = [gene_type, None]
            self.gene_type_single = True
        # A single data type of float with precision.
        elif len(gene_type) == 2 and gene_type[0] in self.supported_float_types and (type(gene_type[1]) in self.supported_int_types or gene_type[1] is None):
            self.gene_type = gene_type
            self.gene_type_single = True
        # A single data type of integer with precision None ([int, None]).
        elif len(gene_type) == 2 and gene_type[0] in self.supported_int_types and gene_type[1] is None:
            self.gene_type = gene_type
            self.gene_type_single = True
        # Raise an exception for a single data type of int with integer precision.
        elif len(gene_type) == 2 and gene_type[0] in self.supported_int_types and (type(gene_type[1]) in self.supported_int_types or gene_type[1] is None):
            self.gene_type_single = False
            raise ValueError(f"Integers cannot have precision. Please use the integer data type directly instead of {gene_type}.")
        elif type(gene_type) in [list, tuple, numpy.ndarray]:
            # Get the number of genes before validating the num_genes parameter.
            if num_genes is None:
                if initial_population is None:
                    self.valid_parameters = False
                    raise TypeError("When the parameter 'initial_population' is None, then the 2 parameters 'sol_per_pop' and 'num_genes' cannot be None too.")
                elif not len(gene_type) == len(initial_population[0]):
                    self.valid_parameters = False
                    raise ValueError(f"When the parameter 'gene_type' is nested, then it can be either [float, int<precision>] or with length equal to the number of genes parameter. Instead, value {gene_type} with len(gene_type) ({len(gene_type)}) != number of genes ({len(initial_population[0])}) found.")
            elif not len(gene_type) == num_genes:
                self.valid_parameters = False
                raise ValueError(f"When the parameter 'gene_type' is nested, then it can be either [float, int<precision>] or with length equal to the value passed to the 'num_genes' parameter. Instead, value {gene_type} with len(gene_type) ({len(gene_type)}) != len(num_genes) ({num_genes}) found.")
            for gene_type_idx, gene_type_val in enumerate(gene_type):
                if gene_type_val in self.supported_int_float_types:
                    # If the gene type is float and no precision is passed or an integer, set its precision to None.
                    gene_type[gene_type_idx] = [gene_type_val, None]
                elif type(gene_type_val) in [list, tuple, numpy.ndarray]:
                    # A float type is expected in a list/tuple/numpy.ndarray of length 2.
                    if len(gene_type_val) == 2:
                        if gene_type_val[0] in self.supported_float_types:
                            if type(gene_type_val[1]) in self.supported_int_types:
                                pass
                            else:
                                self.valid_parameters = False
                                raise TypeError(f"In the 'gene_type' parameter, the precision for float gene data types must be an integer but the element {gene_type_val} at index {gene_type_idx} has a precision of {gene_type_val[1]} with type {gene_type_val[0]}.")
                        elif gene_type_val[0] in self.supported_int_types:
                            if gene_type_val[1] is None:
                                pass
                            else:
                                self.valid_parameters = False
                                raise TypeError(f"In the 'gene_type' parameter, either do not set a precision for integer data types or set it to None. But the element {gene_type_val} at index {gene_type_idx} has a precision of {gene_type_val[1]} with type {gene_type_val[0]}.")
                        else:
                            self.valid_parameters = False
                            raise TypeError(
                                f"In the 'gene_type' parameter, a precision is expected only for float gene data types but the element {gene_type_val} found at index {gene_type_idx}.\nNote that the data type must be at index 0 of the item followed by precision at index 1.")
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"In the 'gene_type' parameter, a precision is specified in a list/tuple/numpy.ndarray of length 2 but value ({gene_type_val}) of type {type(gene_type_val)} with length {len(gene_type_val)} found at index {gene_type_idx}.")
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When a list/tuple/numpy.ndarray is assigned to the 'gene_type' parameter, then its elements must be of integer, floating-point, list, tuple, or numpy.ndarray data types but the value ({gene_type_val}) of type {type(gene_type_val)} found at index {gene_type_idx}.")
            self.gene_type = gene_type
            self.gene_type_single = False
        else:
            self.valid_parameters = False
            raise ValueError(f"The value passed to the 'gene_type' parameter must be either a single integer, floating-point, list, tuple, or numpy.ndarray but ({gene_type}) of type {type(gene_type)} found.")
    
    
    def _build_initial_population(self,
                                  initial_population,
                                  sol_per_pop,
                                  num_genes,
                                  gene_space,
                                  allow_duplicate_genes,
                                  gene_constraint):
        """
        Build or accept the initial population and store it on the GA
        instance. When ``initial_population`` is None, the population
        is generated from scratch by ``initialize_population`` using
        ``sol_per_pop`` and ``num_genes``. Otherwise the user-provided
        array is validated, cast to the right gene types, and
        de-duplicated when ``allow_duplicate_genes`` is False.

        Sets ``self.population``, ``self.initial_population``,
        ``self.sol_per_pop``, ``self.num_genes`` and ``self.pop_size``
        as side effects.

        Parameters
        ----------
        initial_population : list / numpy.ndarray or None
            User-provided initial population. When None, the
            population is built from ``sol_per_pop`` and ``num_genes``.
        sol_per_pop : int or None
            Number of solutions per population. Required when
            ``initial_population`` is None.
        num_genes : int or None
            Number of genes per solution. Required when
            ``initial_population`` is None.
        gene_space : see ``_validate_gene_space``
            The gene space used by the duplicate resolver.
        allow_duplicate_genes : bool
            If False, duplicate genes inside a single solution are
            resolved.
        gene_constraint : list or None
            Per-gene callable constraints; passed through to
            ``initialize_population``.

        Raises
        ------
        TypeError
            If ``initial_population`` is not a list / tuple /
            numpy.ndarray, or its values are not numeric.
        ValueError
            If ``sol_per_pop`` or ``num_genes`` is non-positive, or
            ``initial_population`` is not 2-dimensional.
        """
        # Build the initial population
        if initial_population is None:
            if (sol_per_pop is None) or (num_genes is None):
                self.valid_parameters = False
                raise TypeError("Error creating the initial population:\n\nWhen the parameter 'initial_population' is None, then the 2 parameters 'sol_per_pop' and 'num_genes' cannot be None too.\nThere are 2 options to prepare the initial population:\n1) Assigning the initial population to the 'initial_population' parameter. In this case, the values of the 2 parameters sol_per_pop and num_genes will be deduced.\n2) Assign integer values to the 'sol_per_pop' and 'num_genes' parameters so that PyGAD can create the initial population automatically.")
            elif (type(sol_per_pop) is int) and (type(num_genes) is int):
                # Validating the number of solutions in the population (sol_per_pop)
                if sol_per_pop <= 0:
                    self.valid_parameters = False
                    raise ValueError(f"The number of solutions in the population (sol_per_pop) must be > 0 but ({sol_per_pop}) found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n")
                # Validating the number of gene.
                if (num_genes <= 0):
                    self.valid_parameters = False
                    raise ValueError(f"The number of genes cannot be <= 0 but ({num_genes}) found.\n")
                # When initial_population=None and the 2 parameters sol_per_pop and num_genes have valid integer values, then the initial population is created.
                # Inside the initialize_population() method, the initial_population attribute is assigned to keep the initial population accessible.
                self.num_genes = num_genes  # Number of genes in the solution.

                # In case the 'gene_space' parameter is nested, then make sure the number of its elements equals to the number of genes.
                if self.gene_space_nested:
                    if len(gene_space) != self.num_genes:
                        self.valid_parameters = False
                        raise ValueError(f"When the parameter 'gene_space' is nested, then its length must be equal to the value passed to the 'num_genes' parameter. Instead, length of gene_space ({len(gene_space)}) != num_genes ({self.num_genes})")

                # Number of solutions in the population.
                self.sol_per_pop = sol_per_pop
                self.initialize_population(allow_duplicate_genes=allow_duplicate_genes,
                                           gene_type=self.gene_type,
                                           gene_constraint=gene_constraint)
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of both the sol_per_pop and num_genes parameters is int but {type(sol_per_pop)} and {type(num_genes)} found.")
        elif not type(initial_population) in [list, tuple, numpy.ndarray]:
            self.valid_parameters = False
            raise TypeError(f"The value assigned to the 'initial_population' parameter is expected to be of type list, tuple, or ndarray but {type(initial_population)} found.")
        elif numpy.array(initial_population).ndim != 2:
            self.valid_parameters = False
            raise ValueError(f"A 2D list is expected to the initial_population parameter but a ({numpy.array(initial_population).ndim}-D) list found.")
        else:
            # Validate the type of each value in the 'initial_population' parameter.
            for row_idx in range(len(initial_population)):
                for col_idx in range(len(initial_population[0])):
                    if type(initial_population[row_idx][col_idx]) in self.supported_int_float_types:
                        pass
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"The values in the initial population can be integers or floats but the value ({initial_population[row_idx][col_idx]}) of type {type(initial_population[row_idx][col_idx])} found.")

            # Change the data type and round all genes within the initial population.
            self.initial_population = self.change_population_dtype_and_round(initial_population)

            # Check if duplicates are allowed. If not, then solve any existing duplicates in the passed initial population.
            if self.allow_duplicate_genes == False:
                for initial_solution_idx, initial_solution in enumerate(self.initial_population):
                    if self.gene_space is None:
                        self.initial_population[initial_solution_idx], _, _ = self.solve_duplicate_genes_randomly(solution=initial_solution,
                                                                                                                  min_val=self.init_range_low,
                                                                                                                  max_val=self.init_range_high,
                                                                                                                  mutation_by_replacement=True,
                                                                                                                  gene_type=self.gene_type,
                                                                                                                  sample_size=self.sample_size)
                    else:
                        self.initial_population[initial_solution_idx], _, _ = self.solve_duplicate_genes_by_space(solution=initial_solution,
                                                                                                                  gene_type=self.gene_type,
                                                                                                                  sample_size=self.sample_size,
                                                                                                                  mutation_by_replacement=True,
                                                                                                                  build_initial_pop=True)

            # A NumPy array holding the initial population.
            self.population = self.initial_population.copy()
            # Number of genes in the solution.
            self.num_genes = self.initial_population.shape[1]
            # Number of solutions in the population.
            self.sol_per_pop = self.initial_population.shape[0]
            # The population size.
            self.pop_size = (self.sol_per_pop, self.num_genes)

        # Change the data type and round all genes within the initial population.
        self.initial_population = self.change_population_dtype_and_round(self.initial_population)
        self.population = self.initial_population.copy()
    
    def _validate_mutation_range(self,
                                 random_mutation_min_val,
                                 random_mutation_max_val):
        """
        Validate the random mutation range parameters and store them
        on the GA instance. Both parameters may be scalars (one range
        shared by every gene) or per-gene iterables.

        Sets ``self.random_mutation_min_val`` and
        ``self.random_mutation_max_val``.

        Parameters
        ----------
        random_mutation_min_val : numeric or iterable
            Lower bound(s) for the random delta added during mutation.
        random_mutation_max_val : numeric or iterable
            Upper bound(s) for the random delta added during mutation.

        Raises
        ------
        TypeError
            If either parameter is not a supported numeric type.
        ValueError
            If the per-gene iterables have a length different from
            ``num_genes``.
        """
        # Validate random_mutation_min_val and random_mutation_max_val
        if type(random_mutation_min_val) in self.supported_int_float_types:
            if type(random_mutation_max_val) in self.supported_int_float_types:
                if random_mutation_min_val == random_mutation_max_val:
                    if not self.suppress_warnings:
                        warnings.warn("The values of the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val' are equal and this might cause a fixed mutation to some genes.")
            else:
                self.valid_parameters = False
                raise TypeError(f"Type mismatch between the 2 parameters 'random_mutation_min_val' {type(random_mutation_min_val)} and 'random_mutation_max_val' {type(random_mutation_max_val)}.")
        elif type(random_mutation_min_val) in [list, tuple, numpy.ndarray]:
            if len(random_mutation_min_val) == self.num_genes:
                pass
            else:
                self.valid_parameters = False
                raise ValueError(f"The length of the 'random_mutation_min_val' parameter is {len(random_mutation_min_val)} which is different from the number of genes {self.num_genes}.")
            if type(random_mutation_max_val) in [list, tuple, numpy.ndarray]:
                if len(random_mutation_min_val) == len(random_mutation_max_val):
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"Size mismatch between the 2 parameters 'random_mutation_min_val' {len(random_mutation_min_val)} and 'random_mutation_max_val' {len(random_mutation_max_val)}.")

                # Validate the values in random_mutation_min_val
                for val in random_mutation_min_val:
                    if type(val) in self.supported_int_float_types:
                        pass
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'random_mutation_min_val' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")

                # Validate the values in random_mutation_max_val
                for val in random_mutation_max_val:
                    if type(val) in self.supported_int_float_types:
                        pass
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'random_mutation_max_val' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")
            else:
                self.valid_parameters = False
                raise TypeError(f"Type mismatch between the 2 parameters 'random_mutation_min_val' {type(random_mutation_min_val)} and 'random_mutation_max_val' {type(random_mutation_max_val)}.")
        else:
            self.valid_parameters = False
            raise TypeError(f"The expected type of the 'random_mutation_min_val' parameter is numeric or list/tuple/numpy.ndarray but {type(random_mutation_min_val)} found.")

        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val
    
    
    def _validate_gene_constraint(self,
                                  gene_constraint):
        """
        Validate the ``gene_constraint`` parameter. The constraint is
        a list with one entry per gene; each entry is either None (no
        constraint) or a callable that filters a list of candidate
        values down to the subset that satisfies the constraint.

        Sets ``self.gene_constraint`` on the GA instance.

        Parameters
        ----------
        gene_constraint : list, tuple, or None
            One callable per gene (or None to disable). Length must
            equal ``self.num_genes``.

        Raises
        ------
        TypeError
            If ``gene_constraint`` is not a list / tuple, or any
            element is not None and not callable.
        ValueError
            If the list length does not match ``self.num_genes``.
        """
        # Validate that gene_constraint is a list or tuple and every element inside it is either None or callable.
        if gene_constraint:
            if type(gene_constraint) in [list, tuple]:
                if len(gene_constraint) == self.num_genes:
                    for constraint_idx, item in enumerate(gene_constraint):
                        # Check whether the element is None or a callable.
                        if item is None:
                            pass
                        elif item and callable(item):
                            if item.__code__.co_argcount == 2:
                                # Every callable is valid if it receives 2 arguments.
                                # The 2 arguments: 1) solution 2) A list or numpy.ndarray of values to check if they meet the constraint.
                                pass
                            else:
                                self.valid_parameters = False
                                raise ValueError(f"Every callable inside the gene_constraint parameter must accept 2 arguments representing 1) The solution/chromosome where the gene exists 2) A list or NumPy array of values to check if they meet the constraint. But the callable at index {constraint_idx} named '{item.__code__.co_name}' accepts {item.__code__.co_argcount} argument(s).")
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"The expected type of an element in the 'gene_constraint' parameter is None or a callable (e.g. function). But {item} at index {constraint_idx} of type {type(item)} found.")
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The number of constraints ({len(gene_constraint)}) in the 'gene_constraint' parameter must be equal to the number of genes ({self.num_genes}).")
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'gene_constraint' parameter is either a list or tuple. But the value {gene_constraint} of type {type(gene_constraint)} found.")
        else:
            # gene_constraint is None and not used.
            pass

        self.gene_constraint = gene_constraint
    
    def _validate_crossover(self,
                            crossover_type,
                            crossover_probability,
                            sbx_crossover_eta=30):
        """
        Validate the ``crossover_type`` and ``crossover_probability``
        parameters and store them on the GA instance. ``crossover_type``
        may be:

        - one of the built-in strings (``"single_point"``,
          ``"two_points"``, ``"uniform"``, ``"scattered"``);
        - a callable that takes ``(parents, offspring_size)`` and
          returns the offspring array;
        - None to skip the crossover step entirely.

        ``crossover_probability`` is the per-parent probability of
        being selected for mating; only used by the built-in operators.

        Sets ``self.crossover`` (the operator function) plus
        ``self.crossover_type`` and ``self.crossover_probability``.

        Parameters
        ----------
        crossover_type : str, callable, or None
            The crossover operator selector.
        crossover_probability : float or None
            Per-parent crossover probability between 0 and 1
            inclusive, or None to disable.

        Raises
        ------
        TypeError
            If ``crossover_type`` is neither a string, callable, nor
            None.
        ValueError
            If ``crossover_type`` is an unknown string, the callable
            has the wrong number of parameters, or
            ``crossover_probability`` is outside [0, 1].
        """
        # crossover: Refers to the method that applies the crossover operator based on the selected type of crossover in the crossover_type property.
        # Validating the crossover type: crossover_type
        if crossover_type is None:
            self.crossover = None
        elif inspect.ismethod(crossover_type):
            # Check if the crossover_type is a method that accepts 3 parameters.
            if len(inspect.signature(crossover_type).parameters) == 3:
                # The crossover method assigned to the crossover_type parameter is validated.
                self.crossover = crossover_type
            else:
                self.valid_parameters = False
                raise ValueError(f"When 'crossover_type' is assigned to a method, then this crossover method must accept 3 parameters:\n1) The selected parents.\n2) The size of the offspring to be produced.\n3) The instance from the pygad.GA class.\n\nThe passed crossover method named '{crossover_type.__code__.co_name}' accepts {len(inspect.signature(crossover_type).parameters)} parameter(s).")
        elif inspect.isfunction(crossover_type):
            # Check if the crossover_type is a function that accepts 3 parameters.
            if len(inspect.signature(crossover_type).parameters) == 3:
                # The crossover function assigned to the crossover_type parameter is validated.
                self.crossover = crossover_type
            else:
                self.valid_parameters = False
                raise ValueError(f"When 'crossover_type' is assigned to a function, then this crossover function must accept 3 parameters:\n1) The selected parents.\n2) The size of the offspring to be produced.3) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed crossover function named '{crossover_type.__code__.co_name}' accepts {len(inspect.signature(crossover_type).parameters)} parameter(s).")
        elif callable(crossover_type) and not inspect.isclass(crossover_type):
            # The object must have the __call__() method.
            if hasattr(crossover_type, '__call__'):
                # Check if the __call__() method accepts 3 parameters.
                if len(inspect.signature(crossover_type).parameters) == 3:
                    # The crossover class instance assigned to the crossover_type parameter is validated.
                    self.crossover = crossover_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'crossover_type' is assigned a class instance, then its __call__ method must accept 3 parameters:\n1) The selected parents.\n2) The size of the offspring to be produced.\n3) The instance from the pygad.GA class.\n\nThe passed instance of the class named '{crossover_type.__class__.__name__}' accepts {len(inspect.signature(crossover_type).parameters)} parameter(s).")
            else:
                self.valid_parameters = False
                raise ValueError("When 'crossover_type' is assigned a class instance, then its __call__ method must be implemented and accept 3 parameters.")
        elif not (type(crossover_type) is str):
            self.valid_parameters = False
            raise TypeError(f"The expected type of the 'crossover_type' parameter is either callable or str but {type(crossover_type)} found.")
        else:  # type crossover_type is str
            crossover_type = crossover_type.lower()
            if crossover_type == "single_point":
                self.crossover = self.single_point_crossover
            elif crossover_type == "two_points":
                self.crossover = self.two_points_crossover
            elif crossover_type == "uniform":
                self.crossover = self.uniform_crossover
            elif crossover_type == "scattered":
                self.crossover = self.scattered_crossover
            elif crossover_type == "sbx":
                self.crossover = self.sbx_crossover
            else:
                self.valid_parameters = False
                raise TypeError(f"Undefined crossover type. \nThe assigned value to the crossover_type ({crossover_type}) parameter does not refer to one of the supported crossover types which are: \n-single_point (for single point crossover)\n-two_points (for two points crossover)\n-uniform (for uniform crossover)\n-scattered (for scattered crossover)\n-sbx (for simulated binary crossover).\n")

        self.crossover_type = crossover_type

        # Validate sbx_crossover_eta. It is only used when
        # crossover_type is 'sbx', but it is stored on the instance
        # in all cases so user callables can read it too.
        if type(sbx_crossover_eta) not in self.supported_int_float_types or sbx_crossover_eta <= 0:
            self.valid_parameters = False
            raise ValueError(
                f"sbx_crossover_eta must be a positive number, but got {sbx_crossover_eta!r} "
                f"of type {type(sbx_crossover_eta).__name__}."
            )
        self.sbx_crossover_eta = float(sbx_crossover_eta)

        # Calculate the value of crossover_probability
        if crossover_probability is None:
            self.crossover_probability = None
        elif type(crossover_probability) in self.supported_int_float_types:
            if 0 <= crossover_probability <= 1:
                self.crossover_probability = crossover_probability
            else:
                self.valid_parameters = False
                raise ValueError(f"The value assigned to the 'crossover_probability' parameter must be between 0 and 1 inclusive but ({crossover_probability}) found.")
        else:
            self.valid_parameters = False
            raise TypeError(f"Unexpected type for the 'crossover_probability' parameter. Float is expected but ({crossover_probability}) of type {type(crossover_probability)} found.")
    
    def _validate_mutation(self,
                           mutation_type,
                           mutation_probability,
                           mutation_num_genes,
                           mutation_percent_genes,
                           polynomial_mutation_eta=20):
        """
        Validate the mutation-related parameters and store them on the
        GA instance. ``mutation_type`` may be one of the built-in
        strings (``"random"``, ``"swap"``, ``"inversion"``,
        ``"scramble"``, ``"adaptive"``), a user-supplied callable, or
        None to skip mutation.

        The function also resolves which of ``mutation_probability``,
        ``mutation_num_genes`` and ``mutation_percent_genes`` is in
        effect and translates percentages to gene counts.

        Sets ``self.mutation`` plus ``self.mutation_type``,
        ``self.mutation_probability``, ``self.mutation_num_genes`` and
        ``self.mutation_percent_genes``.

        Parameters
        ----------
        mutation_type : str, callable, or None
            The mutation operator selector.
        mutation_probability : float, list, tuple, numpy.ndarray, or None
            Per-gene mutation probability between 0 and 1 inclusive.
            For adaptive mutation it may be a pair ``[high, low]``
            applied to below-average / above-average solutions.
        mutation_num_genes : int, list, tuple, numpy.ndarray, or None
            Number of genes to mutate per solution. For adaptive
            mutation it may be a pair ``[high, low]``.
        mutation_percent_genes : numeric, list, tuple, numpy.ndarray, or 'default'
            Percentage of genes to mutate. Ignored when
            ``mutation_probability`` or ``mutation_num_genes`` is set.

        Returns
        -------
        mutation_num_genes : int, list, tuple, or numpy.ndarray
            The resolved number of genes to mutate.
        mutation_percent_genes : numeric, list, tuple, or numpy.ndarray
            The resolved percentage of genes to mutate.

        Raises
        ------
        TypeError
            If any parameter has an unexpected type.
        ValueError
            If a probability is outside [0, 1], a count is non-positive
            or larger than ``num_genes``, or a callable has the wrong
            number of parameters.
        """
        # mutation: Refers to the method that applies the mutation operator based on the selected type of mutation in the mutation_type property.
        # Validating the mutation type: mutation_type
        # "adaptive" mutation is supported starting from PyGAD 2.10.0
        if mutation_type is None:
            self.mutation = None
        elif inspect.ismethod(mutation_type):
            # Check if the mutation_type is a method that accepts 2 parameters.
            if (len(inspect.signature(mutation_type).parameters) == 2):
                # The mutation method assigned to the mutation_type parameter is validated.
                self.mutation = mutation_type
            else:
                self.valid_parameters = False
                raise ValueError(f"When 'mutation_type' is assigned to a method, then it must accept 2 parameters:\n1) The offspring to be mutated.\n2) The instance from the pygad.GA class.\n\nThe passed mutation method named '{mutation_type.__code__.co_name}' accepts {len(inspect.signature(mutation_type).parameters)} parameter(s).")
        elif inspect.isfunction(mutation_type):
            # Check if the mutation_type is a function that accepts 2 parameters.
            if (len(inspect.signature(mutation_type).parameters) == 2):
                # The mutation function assigned to the mutation_type parameter is validated.
                self.mutation = mutation_type
            else:
                self.valid_parameters = False
                raise ValueError(f"When 'mutation_type' is assigned to a function, then this mutation function must accept 2 parameters:\n1) The offspring to be mutated.\n2) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed mutation function named '{mutation_type.__code__.co_name}' accepts {len(inspect.signature(mutation_type).parameters)} parameter(s).")
        elif callable(mutation_type) and not inspect.isclass(mutation_type):
            # The object must have the __call__() method.
            if hasattr(mutation_type, '__call__'):
                # Check if the __call__() method accepts 2 parameters.
                if len(inspect.signature(mutation_type).parameters) == 2:
                    # The mutation class instance assigned to the mutation_type parameter is validated.
                    self.mutation = mutation_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'mutation_type' is assigned a class instance, then its __call__ method must accept 2 parameters:\n1) The offspring to be mutated.\n2) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed instance of the class named '{mutation_type.__class__.__name__}' accepts {len(inspect.signature(mutation_type).parameters)} parameter(s).")
            else:
                self.valid_parameters = False
                raise ValueError("When 'mutation_type' is assigned a class instance, then its __call__ method must be implemented and accept 2 parameters.")
        elif not (type(mutation_type) is str):
            self.valid_parameters = False
            raise TypeError(f"The expected type of the 'mutation_type' parameter is either callable or str but {type(mutation_type)} found.")
        else:  # type mutation_type is str
            mutation_type = mutation_type.lower()
            if mutation_type == "random":
                self.mutation = self.random_mutation
            elif mutation_type == "swap":
                self.mutation = self.swap_mutation
            elif mutation_type == "scramble":
                self.mutation = self.scramble_mutation
            elif mutation_type == "inversion":
                self.mutation = self.inversion_mutation
            elif mutation_type == "adaptive":
                self.mutation = self.adaptive_mutation
            elif mutation_type == "polynomial":
                self.mutation = self.polynomial_mutation
            else:
                self.valid_parameters = False
                raise TypeError(f"Undefined mutation type. \nThe assigned string value to the 'mutation_type' parameter ({mutation_type}) does not refer to one of the supported mutation types which are: \n-random (for random mutation)\n-swap (for swap mutation)\n-inversion (for inversion mutation)\n-scramble (for scramble mutation)\n-adaptive (for adaptive mutation)\n-polynomial (for polynomial mutation).\n")

        self.mutation_type = mutation_type

        # Validate polynomial_mutation_eta. It is only used when
        # mutation_type is 'polynomial', but it is stored on the
        # instance in all cases so user callables can read it too.
        if (type(polynomial_mutation_eta) not in self.supported_int_float_types
                or polynomial_mutation_eta <= 0):
            self.valid_parameters = False
            raise ValueError(
                f"polynomial_mutation_eta must be a positive number, but "
                f"got {polynomial_mutation_eta!r} of type "
                f"{type(polynomial_mutation_eta).__name__}."
            )
        self.polynomial_mutation_eta = float(polynomial_mutation_eta)

        # Calculate the value of mutation_probability
        if not (self.mutation_type is None):
            if mutation_probability is None:
                self.mutation_probability = None
            elif mutation_type != "adaptive":
                # Mutation probability is fixed not adaptive.
                if type(mutation_probability) in self.supported_int_float_types:
                    if 0 <= mutation_probability <= 1:
                        self.mutation_probability = mutation_probability
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The value assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but ({mutation_probability}) found.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Unexpected type for the 'mutation_probability' parameter. A numeric value is expected but ({mutation_probability}) of type {type(mutation_probability)} found.")
            else:
                # Mutation probability is adaptive not fixed.
                if type(mutation_probability) in [list, tuple, numpy.ndarray]:
                    if len(mutation_probability) == 2:
                        for el in mutation_probability:
                            if type(el) in self.supported_int_float_types:
                                if 0 <= el <= 1:
                                    pass
                                else:
                                    self.valid_parameters = False
                                    raise ValueError(f"The values assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but ({el}) found.")
                            else:
                                self.valid_parameters = False
                                raise TypeError(f"Unexpected type for a value assigned to the 'mutation_probability' parameter. A numeric value is expected but ({el}) of type {type(el)} found.")
                        if mutation_probability[0] < mutation_probability[1]:
                            if not self.suppress_warnings:
                                warnings.warn(f"The first element in the 'mutation_probability' parameter is {mutation_probability[0]} which is smaller than the second element {mutation_probability[1]}. This means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high quality solutions while making little changes in the low quality solutions. Please make the first element higher than the second element.")
                        self.mutation_probability = mutation_probability
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When mutation_type='adaptive', then the 'mutation_probability' parameter must have only 2 elements but ({len(mutation_probability)}) element(s) found.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Unexpected type for the 'mutation_probability' parameter. When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected but ({mutation_probability}) of type {type(mutation_probability)} found.")
        else:
            pass

        # Calculate the value of mutation_num_genes
        if not (self.mutation_type is None):
            if mutation_num_genes is None:
                # The mutation_num_genes parameter does not exist. Checking whether adaptive mutation is used.
                if mutation_type != "adaptive":
                    # The percent of genes to mutate is fixed not adaptive.
                    if mutation_percent_genes == 'default'.lower():
                        mutation_percent_genes = 10
                        # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                        mutation_num_genes = numpy.uint32(
                            (mutation_percent_genes*self.num_genes)/100)
                        # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                        if mutation_num_genes == 0:
                            if self.mutation_probability is None:
                                if not self.suppress_warnings:
                                    warnings.warn(
                                        f"The percentage of genes to mutate (mutation_percent_genes={mutation_percent_genes}) resulted in selecting ({mutation_num_genes}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.")
                            mutation_num_genes = 1

                    elif type(mutation_percent_genes) in self.supported_int_float_types:
                        if mutation_percent_genes <= 0 or mutation_percent_genes > 100:
                            self.valid_parameters = False
                            raise ValueError(f"The percentage of selected genes for mutation (mutation_percent_genes) must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n")
                        else:
                            # If mutation_percent_genes equals the string "default", then it is replaced by the numeric value 10.
                            if mutation_percent_genes == 'default'.lower():
                                mutation_percent_genes = 10

                            # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                            mutation_num_genes = numpy.uint32(
                                (mutation_percent_genes*self.num_genes)/100)
                            # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                            if mutation_num_genes == 0:
                                if self.mutation_probability is None:
                                    if not self.suppress_warnings:
                                        warnings.warn(f"The percentage of genes to mutate (mutation_percent_genes={mutation_percent_genes}) resulted in selecting ({mutation_num_genes}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.")
                                mutation_num_genes = 1
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"Unexpected value or type of the 'mutation_percent_genes' parameter. It only accepts the string 'default' or a numeric value but ({mutation_percent_genes}) of type {type(mutation_percent_genes)} found.")
                else:
                    # The percent of genes to mutate is adaptive not fixed.
                    if type(mutation_percent_genes) in [list, tuple, numpy.ndarray]:
                        if len(mutation_percent_genes) == 2:
                            mutation_num_genes = numpy.zeros_like(
                                mutation_percent_genes, dtype=numpy.uint32)
                            for idx, el in enumerate(mutation_percent_genes):
                                if type(el) in self.supported_int_float_types:
                                    if el <= 0 or el > 100:
                                        self.valid_parameters = False
                                        raise ValueError(f"The values assigned to the 'mutation_percent_genes' must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n")
                                else:
                                    self.valid_parameters = False
                                    raise TypeError(f"Unexpected type for a value assigned to the 'mutation_percent_genes' parameter. An integer value is expected but ({el}) of type {type(el)} found.")
                                # At this point of the loop, the current value assigned to the parameter 'mutation_percent_genes' is validated.
                                # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                                mutation_num_genes[idx] = numpy.uint32(
                                    (mutation_percent_genes[idx]*self.num_genes)/100)
                                # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                                if mutation_num_genes[idx] == 0:
                                    if not self.suppress_warnings:
                                        warnings.warn(f"The percentage of genes to mutate ({mutation_percent_genes[idx]}) resulted in selecting ({mutation_num_genes[idx]}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.")
                                    mutation_num_genes[idx] = 1
                            if mutation_percent_genes[0] < mutation_percent_genes[1]:
                                if not self.suppress_warnings:
                                    warnings.warn(f"The first element in the 'mutation_percent_genes' parameter is ({mutation_percent_genes[0]}) which is smaller than the second element ({mutation_percent_genes[1]}).\nThis means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high quality solutions while making little changes in the low quality solutions.\nPlease make the first element higher than the second element.")
                            # At this point outside the loop, all values of the parameter 'mutation_percent_genes' are validated. Everything is OK.
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"When mutation_type='adaptive', then the 'mutation_percent_genes' parameter must have only 2 elements but ({len(mutation_percent_genes)}) element(s) found.")
                    else:
                        if self.mutation_probability is None:
                            self.valid_parameters = False
                            raise TypeError(f"Unexpected type of the 'mutation_percent_genes' parameter. When mutation_type='adaptive', then the 'mutation_percent_genes' parameter should exist and assigned a list/tuple/numpy.ndarray with 2 values but ({mutation_percent_genes}) found.")
            # The mutation_num_genes parameter exists. Checking whether adaptive mutation is used.
            elif mutation_type != "adaptive":
                # Number of genes to mutate is fixed not adaptive.
                if type(mutation_num_genes) in self.supported_int_types:
                    if mutation_num_genes <= 0:
                        self.valid_parameters = False
                        raise ValueError(f"The number of selected genes for mutation (mutation_num_genes) cannot be <= 0 but ({mutation_num_genes}) found. If you do not want to use mutation, please set mutation_type=None\n")
                    elif mutation_num_genes > self.num_genes:
                        self.valid_parameters = False
                        raise ValueError(f"The number of selected genes for mutation (mutation_num_genes), which is ({mutation_num_genes}), cannot be greater than the number of genes ({self.num_genes}).\n")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The 'mutation_num_genes' parameter is expected to be a positive integer but the value ({mutation_num_genes}) of type {type(mutation_num_genes)} found.\n")
            else:
                # Number of genes to mutate is adaptive not fixed.
                if type(mutation_num_genes) in [list, tuple, numpy.ndarray]:
                    if len(mutation_num_genes) == 2:
                        for el in mutation_num_genes:
                            if type(el) in self.supported_int_types:
                                if el <= 0:
                                    self.valid_parameters = False
                                    raise ValueError(f"The values assigned to the 'mutation_num_genes' cannot be <= 0 but ({el}) found. If you do not want to use mutation, please set mutation_type=None\n")
                                elif el > self.num_genes:
                                    self.valid_parameters = False
                                    raise ValueError(f"The values assigned to the 'mutation_num_genes' cannot be greater than the number of genes ({self.num_genes}) but ({el}) found.\n")
                            else:
                                self.valid_parameters = False
                                raise TypeError(f"Unexpected type for a value assigned to the 'mutation_num_genes' parameter. An integer value is expected but ({el}) of type {type(el)} found.")
                            # At this point of the loop, the current value assigned to the parameter 'mutation_num_genes' is validated.
                        if mutation_num_genes[0] < mutation_num_genes[1]:
                            if not self.suppress_warnings:
                                warnings.warn(f"The first element in the 'mutation_num_genes' parameter is {mutation_num_genes[0]} which is smaller than the second element {mutation_num_genes[1]}. This means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high quality solutions while making little changes in the low quality solutions. Please make the first element higher than the second element.")
                        # At this point outside the loop, all values of the parameter 'mutation_num_genes' are validated. Everything is OK.
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When mutation_type='adaptive', then the 'mutation_num_genes' parameter must have only 2 elements but ({len(mutation_num_genes)}) element(s) found.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Unexpected type for the 'mutation_num_genes' parameter. When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected but ({mutation_num_genes}) of type {type(mutation_num_genes)} found.")
        else:
            pass

        # Validating mutation_by_replacement and mutation_type
        if self.mutation_type != "random" and self.mutation_by_replacement:
            if not self.suppress_warnings:
                warnings.warn(f"The mutation_by_replacement parameter is set to True while the mutation_type parameter is not set to random but ({mutation_type}). Note that the mutation_by_replacement parameter has an effect only when mutation_type='random'.")

        # Check if crossover and mutation are both disabled.
        if (self.mutation_type is None) and (self.crossover_type is None):
            if not self.suppress_warnings:
                warnings.warn("The 2 parameters mutation_type and crossover_type are None. This disables any type of evolution the genetic algorithm can make. As a result, the genetic algorithm cannot find a better solution than the best solution in the initial population.")
        return mutation_num_genes, mutation_percent_genes

    def _validate_nsga3_num_divisions(self, parent_selection_type, nsga3_num_divisions):
        """
        Validate ``nsga3_num_divisions`` and store it on the GA
        instance. The parameter is only required when
        ``parent_selection_type`` is ``"nsga3"`` or
        ``"tournament_nsga3"``; otherwise the value is accepted as-is
        for forward compatibility.

        Parameters
        ----------
        parent_selection_type : str
            The selection operator name. Only the two NSGA-III
            variants treat ``nsga3_num_divisions`` as required.
        nsga3_num_divisions : int or None
            Number of divisions per objective axis (the ``p``
            parameter of the Das-Dennis reference grid).

        Raises
        ------
        ValueError
            If ``parent_selection_type`` is one of the NSGA-III
            variants and ``nsga3_num_divisions`` is None, not an
            integer, or not positive.
        """
        if parent_selection_type not in ("nsga3", "tournament_nsga3"):
            self.nsga3_num_divisions = nsga3_num_divisions
            return
        if nsga3_num_divisions is None:
            self.valid_parameters = False
            raise ValueError(
                f"parent_selection_type='{parent_selection_type}' requires "
                f"nsga3_num_divisions to be a positive integer. Pass "
                f"nsga3_num_divisions=<int> to GA(...)."
            )
        if (type(nsga3_num_divisions) not in self.supported_int_types
                or nsga3_num_divisions <= 0):
            self.valid_parameters = False
            raise ValueError(
                f"nsga3_num_divisions must be a positive integer when "
                f"parent_selection_type='{parent_selection_type}', but got "
                f"{nsga3_num_divisions!r} of type "
                f"{type(nsga3_num_divisions).__name__}."
            )
        self.nsga3_num_divisions = int(nsga3_num_divisions)

    def _validate_parent_selection(self,
                                   parent_selection_type,
                                   K_tournament,
                                   keep_parents,
                                   keep_elitism,
                                   nsga3_num_divisions=None):
        """
        Validate the parameters that control parent selection,
        retention and elitism. Resolves ``parent_selection_type`` to
        an actual operator (built-in string or user callable) and
        stores it on ``self.select_parents``. Also computes
        ``self.num_offspring`` from ``sol_per_pop``, ``keep_parents``
        and ``keep_elitism``.

        Parameters
        ----------
        parent_selection_type : str or callable
            One of the built-in selection names or a user-supplied
            function with three parameters (fitness, num_parents,
            ga_instance).
        K_tournament : int
            Tournament size used by the tournament-based operators.
            Clipped to ``self.sol_per_pop`` when too large.
        keep_parents : int
            Number of parents to carry over to the next generation.
            ``-1`` keeps all selected parents; ``0`` keeps none;
            positive values keep exactly that many.
        keep_elitism : int
            Number of top solutions to copy unchanged into the next
            generation. Takes priority over ``keep_parents``.
        nsga3_num_divisions : int or None
            Forwarded to ``_validate_nsga3_num_divisions``.

        Returns
        -------
        parent_selection_type : str or callable
            The (possibly lowercased) selection type stored on
            ``self``.

        Raises
        ------
        TypeError
            If ``parent_selection_type`` is not a supported type, or
            a user callable does not have three parameters, or
            ``K_tournament`` / ``keep_parents`` / ``keep_elitism`` is
            of the wrong type.
        ValueError
            If a numeric parameter is out of range, or the selection
            name is unknown.
        """
        # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
        # Validating the selected type of parent selection: parent_selection_type
        if inspect.ismethod(parent_selection_type):
            # Check if the parent_selection_type is a method that accepts 3 parameters.
            if len(inspect.signature(parent_selection_type).parameters) == 3:
                # The parent selection method assigned to the parent_selection_type parameter is validated.
                self.select_parents = parent_selection_type
            else:
                self.valid_parameters = False
                raise ValueError(f"When 'parent_selection_type' is assigned to a method, then it must accept 3 parameters:\n1) The fitness values of the current population.\n2) The number of parents needed.\n3) The instance from the pygad.GA class.\n\nThe passed parent selection method named '{parent_selection_type.__code__.co_name}' accepts {len(inspect.signature(parent_selection_type).parameters)} parameter(s).")
        elif inspect.isfunction(parent_selection_type):
            # Check if the parent_selection_type is a function that accepts 3 parameters.
            if len(inspect.signature(parent_selection_type).parameters) == 3:
                # The parent selection function assigned to the parent_selection_type parameter is validated.
                self.select_parents = parent_selection_type
            else:
                self.valid_parameters = False
                raise ValueError(f"When 'parent_selection_type' is assigned to a user-defined function, then this parent selection function must accept 3 parameters:\n1) The fitness values of the current population.\n2) The number of parents needed.\n3) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed parent selection function named '{parent_selection_type.__code__.co_name}' accepts {len(inspect.signature(parent_selection_type).parameters)} parameter(s).")
        elif callable(parent_selection_type) and not inspect.isclass(parent_selection_type):
            # The object must have the __call__() method.
            if hasattr(parent_selection_type, '__call__'):
                # Check if the __call__() method accepts 3 parameters.
                if len(inspect.signature(parent_selection_type).parameters) == 3:
                    # The parent selection class instance assigned to the parent_selection_type parameter is validated.
                    self.select_parents = parent_selection_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'parent_selection_type' is assigned a class instance, then its __call__ method must accept 3 parameters:\n1) The fitness values of the current population.\n2) The number of parents needed.\n3) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed instance of the class named '{parent_selection_type.__class__.__name__}' accepts {len(inspect.signature(parent_selection_type).parameters)} parameter(s).")
            else:
                self.valid_parameters = False
                raise ValueError("When 'parent_selection_type' is assigned a class instance, then its __call__ method must be implemented and accept 3 parameters.")
        elif not (type(parent_selection_type) is str):
            self.valid_parameters = False

            raise TypeError(f"The expected type of the 'parent_selection_type' parameter is either callable or str but {type(parent_selection_type)} found.")
        else:
            parent_selection_type = parent_selection_type.lower()
            if parent_selection_type == "sss":
                self.select_parents = self.steady_state_selection
            elif parent_selection_type == "rws":
                self.select_parents = self.roulette_wheel_selection
            elif parent_selection_type == "sus":
                self.select_parents = self.stochastic_universal_selection
            elif parent_selection_type == "random":
                self.select_parents = self.random_selection
            elif parent_selection_type == "tournament":
                self.select_parents = self.tournament_selection
            elif parent_selection_type == "tournament_nsga2": # Supported in PyGAD >= 3.2
                self.select_parents = self.tournament_selection_nsga2
            elif parent_selection_type == "nsga2": # Supported in PyGAD >= 3.2
                self.select_parents = self.nsga2_selection
            elif parent_selection_type == "tournament_nsga3":
                self.select_parents = self.tournament_selection_nsga3
            elif parent_selection_type == "nsga3":
                self.select_parents = self.nsga3_selection
            elif parent_selection_type == "rank":
                self.select_parents = self.rank_selection
            else:
                self.valid_parameters = False
                raise TypeError(f"Undefined parent selection type: {parent_selection_type}. \nThe assigned value to the 'parent_selection_type' parameter does not refer to one of the supported parent selection techniques which are: \n-sss (steady state selection)\n-rws (roulette wheel selection)\n-sus (stochastic universal selection)\n-rank (rank selection)\n-random (random selection)\n-tournament (tournament selection)\n-tournament_nsga2: (Tournament selection for NSGA-II)\n-nsga2: (NSGA-II parent selection)\n-tournament_nsga3: (Tournament selection for NSGA-III)\n-nsga3: (NSGA-III parent selection).\n")

        # For tournament selection, validate the K value.
        if parent_selection_type == "tournament":
            if type(K_tournament) in self.supported_int_types:
                if K_tournament > self.sol_per_pop:
                    K_tournament = self.sol_per_pop
                    if not self.suppress_warnings:
                        warnings.warn(f"K of the tournament selection ({K_tournament}) should not be greater than the number of solutions within the population ({self.sol_per_pop}).\nK will be clipped to be equal to the number of solutions in the population (sol_per_pop).\n")
                elif K_tournament <= 0:
                    self.valid_parameters = False
                    raise ValueError(f"K of the tournament selection cannot be <=0 but ({K_tournament}) found.\n")
            else:
                self.valid_parameters = False
                raise ValueError(f"The type of K of the tournament selection must be integer but the value ({K_tournament}) of type ({type(K_tournament)}) found.")

        self.K_tournament = K_tournament

        self._validate_nsga3_num_divisions(parent_selection_type, nsga3_num_divisions)

        # Validating the number of parents to keep in the next population: keep_parents
        # keep_parents defaults to None (sentinel) so we can tell whether the user
        # explicitly set it. Resolve None to -1 to preserve the historical default
        # behavior (keep all selected parents) byte-for-byte.
        self.keep_parents_explicitly_set = keep_parents is not None
        if keep_parents is None:
            keep_parents = -1
        if not (type(keep_parents) in self.supported_int_types):
            self.valid_parameters = False
            raise TypeError(f"Incorrect type of the value assigned to the keep_parents parameter. The value ({keep_parents}) of type {type(keep_parents)} found but an integer is expected.")
        elif keep_parents > self.sol_per_pop or keep_parents > self.num_parents_mating or keep_parents < -1:
            self.valid_parameters = False
            raise ValueError(f"Incorrect value to the keep_parents parameter: {keep_parents}. \nThe assigned value to the keep_parent parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Less than or equal to num_parents_mating\n3) Greater than or equal to -1.")

        self.keep_parents = keep_parents

        if parent_selection_type == "sss" and self.keep_parents == 0:
            if not self.suppress_warnings:
                warnings.warn("The steady-state parent (sss) selection operator is used despite that no parents are kept in the next generation.")

        # Validating the number of elitism to keep in the next population: keep_elitism
        if not (type(keep_elitism) in self.supported_int_types):
            self.valid_parameters = False
            raise TypeError(f"Incorrect type of the value assigned to the keep_elitism parameter. The value ({keep_elitism}) of type {type(keep_elitism)} found but an integer is expected.")
        elif keep_elitism > self.sol_per_pop or keep_elitism < 0:
            self.valid_parameters = False
            raise ValueError(f"Incorrect value to the keep_elitism parameter: {keep_elitism}. \nThe assigned value to the keep_elitism parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Greater than or equal to 0.")

        self.keep_elitism = keep_elitism

        # keep_elitism takes precedence over keep_parents: when keep_elitism > 0,
        # keep_parents is ignored. Warn if the user explicitly set keep_parents while
        # keep_elitism is non-zero, instead of silently ignoring it.
        if self.keep_parents_explicitly_set and self.keep_elitism != 0:
            if not self.suppress_warnings:
                warnings.warn(f"Both keep_parents (={self.keep_parents}) and keep_elitism (={self.keep_elitism}) are set. Because keep_elitism is greater than 0, it takes precedence and keep_parents is ignored. To make keep_parents take effect, set keep_elitism=0.")

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

    def _validate_fitness_func(self,
                               fitness_func,
                               fitness_batch_size):
        """
        Validate the ``fitness_func`` and ``fitness_batch_size``
        parameters and store them on the GA instance. The fitness
        function must be a method or function (or a class with a
        ``__call__`` method) that takes three parameters: the GA
        instance, a solution (or a batch), and the solution index (or
        a batch of indices).

        Sets ``self.fitness_func`` and ``self.fitness_batch_size``.

        Parameters
        ----------
        fitness_func : callable
            The fitness function described above.
        fitness_batch_size : int or None
            When set, batches of this many solutions are passed to
            ``fitness_func`` at once. ``None`` or ``1`` evaluates one
            solution per call.

        Raises
        ------
        TypeError
            If ``fitness_func`` is not callable.
        ValueError
            If ``fitness_func`` does not accept three parameters, or
            ``fitness_batch_size`` is not a positive integer.
        """
        # Check if the fitness_func is a method.
        if inspect.ismethod(fitness_func):
            # Check if the fitness method accepts 3 parameters.
            if len(inspect.signature(fitness_func).parameters) == 3:
                self.fitness_func = fitness_func
            else:
                self.valid_parameters = False
                raise ValueError(f"In PyGAD 2.20.0, if a method is used to calculate the fitness value, then it must accept 3 parameters\n1) The instance of the 'pygad.GA' class.\n2) A solution to calculate its fitness value.\n3) The solution's index within the population.\n\nThe passed fitness method named '{fitness_func.__code__.co_name}' accepts {len(inspect.signature(fitness_func).parameters)} parameter(s).")
        elif inspect.isfunction(fitness_func):
            # Check if the fitness function accepts 3 parameters.
            if len(inspect.signature(fitness_func).parameters) == 3:
                self.fitness_func = fitness_func
            else:
                self.valid_parameters = False
                raise ValueError(f"In PyGAD 2.20.0, the fitness function must accept 3 parameters:\n1) The instance of the 'pygad.GA' class.\n2) A solution to calculate its fitness value.\n3) The solution's index within the population.\n\nThe passed fitness function named '{fitness_func.__code__.co_name}' accepts {len(inspect.signature(fitness_func).parameters)} parameter(s).")
        elif callable(fitness_func) and not inspect.isclass(fitness_func):
            # The object must have the __call__() method.
            if hasattr(fitness_func, '__call__'):
                # Check if the __call__() method accepts 3 parameters.
                if len(inspect.signature(fitness_func).parameters) == 3:
                    # The fitness class instance assigned to the fitness_func parameter is validated.
                    self.fitness_func = fitness_func
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'fitness_func' is assigned a class instance, then its __call__ method must accept 3 parameters:\n1) The instance of the 'pygad.GA' class.\n2) A solution to calculate its fitness value.\n3) The solution's index within the population.\n\nThe passed instance of the class named '{fitness_func.__class__.__name__}' accepts {len(inspect.signature(fitness_func).parameters)} parameter(s).")
            else:
                self.valid_parameters = False
                raise ValueError("When 'fitness_func' is assigned a class instance, then its __call__ method must be implemented and accept 3 parameters.")
        else:
            self.valid_parameters = False
            
            raise TypeError(f"The value assigned to the fitness_func parameter is expected to be a function or a method but {type(fitness_func)} found.")

        if fitness_batch_size is None:
            pass
        elif not (type(fitness_batch_size) in self.supported_int_types):
            self.valid_parameters = False
            raise TypeError(f"The value assigned to the fitness_batch_size parameter is expected to be integer but the value ({fitness_batch_size}) of type {type(fitness_batch_size)} found.")
        elif fitness_batch_size <= 0 or fitness_batch_size > self.sol_per_pop:
            self.valid_parameters = False
            raise ValueError(f"The value assigned to the fitness_batch_size parameter must be:\n1) Greater than 0.\n2) Less than or equal to sol_per_pop ({self.sol_per_pop}).\nBut the value ({fitness_batch_size}) found.")

        self.fitness_batch_size = fitness_batch_size
    
    def _validate_callbacks(self,
                            on_start,
                            on_fitness,
                            on_parents,
                            on_crossover,
                            on_mutation,
                            on_generation,
                            on_stop):
        """
        Validate the seven optional lifecycle callbacks and store
        them on the GA instance under matching ``self.on_*``
        attributes. Each callback must be a function or method with
        the expected number of parameters.

        Parameters
        ----------
        on_start : callable or None
            Called once before the generational loop. Receives the
            GA instance.
        on_fitness : callable or None
            Called after the fitness of the current population has
            been evaluated. Receives the GA instance and the fitness
            array.
        on_parents : callable or None
            Called after the parent selection step. Receives the GA
            instance and the selected parents.
        on_crossover : callable or None
            Called after the crossover step. Receives the GA instance
            and the crossover offspring.
        on_mutation : callable or None
            Called after the mutation step. Receives the GA instance
            and the mutated offspring.
        on_generation : callable or None
            Called after each generation completes. Receives the GA
            instance. Returning the string ``"stop"`` ends the run.
        on_stop : callable or None
            Called once after the generational loop ends. Receives
            the GA instance and the last-generation fitness array.

        Raises
        ------
        TypeError
            If a callback is not callable.
        ValueError
            If a callback does not have the expected number of
            parameters.
        """
        # Check if the on_start exists.
        if not (on_start is None):
            if inspect.ismethod(on_start):
                # Check if the on_start method accepts 1 parameter.
                if len(inspect.signature(on_start).parameters) == 1:
                    self.on_start = on_start
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The method assigned to the on_start parameter must accept only 1 parameter representing the instance of the genetic algorithm. The passed method named '{on_start.__code__.co_name}' accepts {len(inspect.signature(on_start).parameters)} parameter(s).")
            # Check if the on_start is a function.
            elif inspect.isfunction(on_start):
                # Check if the on_start function accepts only a single parameter.
                if len(inspect.signature(on_start).parameters) == 1:
                    self.on_start = on_start
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The function assigned to the on_start parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{on_start.__code__.co_name}' accepts {len(inspect.signature(on_start).parameters)} parameter(s).")
            elif callable(on_start) and not inspect.isclass(on_start):
                # The object must have the __call__() method.
                if hasattr(on_start, '__call__'):
                    # Check if the __call__() method accepts 1 parameter.
                    if len(inspect.signature(on_start).parameters) == 1:
                        # The on_start class instance assigned to the on_start parameter is validated.
                        self.on_start = on_start
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When 'on_start' is assigned a class instance, then its __call__ method must accept only 1 parameter representing the instance of the genetic algorithm.\n\nThe passed instance of the class named '{on_start.__class__.__name__}' accepts {len(inspect.signature(on_start).parameters)} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise ValueError("When 'on_start' is assigned a class instance, then its __call__ method must be implemented and accept 1 parameter.")
            else:
                self.valid_parameters = False
                
                raise TypeError(f"The value assigned to the on_start parameter is expected to be of type function but {type(on_start)} found.")
        else:
            self.on_start = None

        # Check if the on_fitness exists.
        if not (on_fitness is None):
            # Check if the on_fitness is a method.
            if inspect.ismethod(on_fitness):
                # Check if the on_fitness method accepts 2 parameters.
                if len(inspect.signature(on_fitness).parameters) == 2:
                    self.on_fitness = on_fitness
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The method assigned to the on_fitness parameter must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The fitness values of all solutions.\nThe passed method named '{on_fitness.__code__.co_name}' accepts {len(inspect.signature(on_fitness).parameters)} parameter(s).")
            # Check if the on_fitness is a function.
            elif inspect.isfunction(on_fitness):
                # Check if the on_fitness function accepts 2 parameters.
                if len(inspect.signature(on_fitness).parameters) == 2:
                    self.on_fitness = on_fitness
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The function assigned to the on_fitness parameter must accept 2 parameters representing the instance of the genetic algorithm and the fitness values of all solutions.\nThe passed function named '{on_fitness.__code__.co_name}' accepts {on_fitness.__code__.co_argcount} parameter(s).")
            elif callable(on_fitness) and not inspect.isclass(on_fitness):
                # The object must have the __call__() method.
                if hasattr(on_fitness, '__call__'):
                    # Check if the __call__() method accepts 2 parameters.
                    if len(inspect.signature(on_fitness).parameters) == 2:
                        # The on_fitness class instance assigned to the on_fitness parameter is validated.
                        self.on_fitness = on_fitness
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When 'on_fitness' is assigned a class instance, then its __call__ method must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The fitness values of all solutions.\n\nThe passed instance of the class named '{on_fitness.__class__.__name__}' accepts {len(inspect.signature(on_fitness).parameters)} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise ValueError("When 'on_fitness' is assigned a class instance, then its __call__ method must be implemented and accept 2 parameters.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the on_fitness parameter is expected to be of type function but {type(on_fitness)} found.")
        else:
            self.on_fitness = None

        # Check if the on_parents exists.
        if not (on_parents is None):
            # Check if the on_parents is a method.
            if inspect.ismethod(on_parents):
                # Check if the on_parents method accepts 2 parameters.
                if len(inspect.signature(on_parents).parameters) == 2:
                    self.on_parents = on_parents
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The method assigned to the on_parents parameter must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The fitness values of all solutions.\nThe passed method named '{on_parents.__code__.co_name}' accepts {len(inspect.signature(on_parents).parameters)} parameter(s).")
            # Check if the on_parents is a function.
            elif inspect.isfunction(on_parents):
                # Check if the on_parents function accepts 2 parameters.
                if len(inspect.signature(on_parents).parameters) == 2:
                    self.on_parents = on_parents
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The function assigned to the on_parents parameter must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The fitness values of all solutions.\nThe passed function named '{on_parents.__code__.co_name}' accepts {len(inspect.signature(on_parents).parameters)} parameter(s).")
            elif callable(on_parents) and not inspect.isclass(on_parents):
                # The object must have the __call__() method.
                if hasattr(on_parents, '__call__'):
                    # Check if the __call__() method accepts 2 parameters.
                    if len(inspect.signature(on_parents).parameters) == 2:
                        # The on_parents class instance assigned to the on_parents parameter is validated.
                        self.on_parents = on_parents
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When 'on_parents' is assigned a class instance, then its __call__ method must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The fitness values of all solutions.\n\nThe passed instance of the class named '{on_parents.__class__.__name__}' accepts {len(inspect.signature(on_parents).parameters)} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise ValueError("When 'on_parents' is assigned a class instance, then its __call__ method must be implemented and accept 2 parameters.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the on_parents parameter is expected to be of type function but {type(on_parents)} found.")
        else:
            self.on_parents = None

        # Check if the on_crossover exists.
        if not (on_crossover is None):
            # Check if the on_crossover is a method.
            if inspect.ismethod(on_crossover):
                # Check if the on_crossover method accepts 2 parameters.
                if len(inspect.signature(on_crossover).parameters) == 2:
                    self.on_crossover = on_crossover
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The method assigned to the on_crossover parameter must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The offspring generated using crossover.\nThe passed method named '{on_crossover.__code__.co_name}' accepts {len(inspect.signature(on_crossover).parameters)} parameter(s).")
            # Check if the on_crossover is a function.
            elif inspect.isfunction(on_crossover):
                # Check if the on_crossover function accepts 2 parameters.
                if len(inspect.signature(on_crossover).parameters) == 2:
                    self.on_crossover = on_crossover
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The function assigned to the on_crossover parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring generated using crossover.\nThe passed function named '{on_crossover.__code__.co_name}' accepts {len(inspect.signature(on_crossover).parameters)} parameter(s).")
            elif callable(on_crossover) and not inspect.isclass(on_crossover):
                # The object must have the __call__() method.
                if hasattr(on_crossover, '__call__'):
                    # Check if the __call__() method accepts 2 parameters.
                    if len(inspect.signature(on_crossover).parameters) == 2:
                        # The on_crossover class instance assigned to the on_crossover parameter is validated.
                        self.on_crossover = on_crossover
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When 'on_crossover' is assigned a class instance, then its __call__ method must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The offspring generated using crossover.\n\nThe passed instance of the class named '{on_crossover.__class__.__name__}' accepts {len(inspect.signature(on_crossover).parameters)} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise ValueError("When 'on_crossover' is assigned a class instance, then its __call__ method must be implemented and accept 2 parameters.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the on_crossover parameter is expected to be of type function but {type(on_crossover)} found.")
        else:
            self.on_crossover = None

        # Check if the on_mutation exists.
        if not (on_mutation is None):
            # Check if the on_mutation is a method.
            if inspect.ismethod(on_mutation):
                # Check if the on_mutation method accepts 2 parameters.
                if len(inspect.signature(on_mutation).parameters) == 2:
                    self.on_mutation = on_mutation
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The method assigned to the on_mutation parameter must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The offspring after applying the mutation operation.\nThe passed method named '{on_mutation.__code__.co_name}' accepts {len(inspect.signature(on_mutation).parameters)} parameter(s).")
            # Check if the on_mutation is a function.
            elif inspect.isfunction(on_mutation):
                # Check if the on_mutation function accepts 2 parameters.
                if len(inspect.signature(on_mutation).parameters) == 2:
                    self.on_mutation = on_mutation
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The function assigned to the on_mutation parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring after applying the mutation operation.\nThe passed function named '{on_mutation.__code__.co_name}' accepts {len(inspect.signature(on_mutation).parameters)} parameter(s).")
            elif callable(on_mutation) and not inspect.isclass(on_mutation):
                # The object must have the __call__() method.
                if hasattr(on_mutation, '__call__'):
                    # Check if the __call__() method accepts 2 parameters.
                    if len(inspect.signature(on_mutation).parameters) == 2:
                        # The on_mutation class instance assigned to the on_mutation parameter is validated.
                        self.on_mutation = on_mutation
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When 'on_mutation' is assigned a class instance, then its __call__ method must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) The offspring after applying the mutation operation.\n\nThe passed instance of the class named '{on_mutation.__class__.__name__}' accepts {len(inspect.signature(on_mutation).parameters)} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise ValueError("When 'on_mutation' is assigned a class instance, then its __call__ method must be implemented and accept 2 parameters.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the on_mutation parameter is expected to be of type function but {type(on_mutation)} found.")
        else:
            self.on_mutation = None

        # Check if the on_generation exists.
        if not (on_generation is None):
            # Check if the on_generation is a method.
            if inspect.ismethod(on_generation):
                # Check if the on_generation method accepts 1 parameter.
                if len(inspect.signature(on_generation).parameters) == 1:
                    self.on_generation = on_generation
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The method assigned to the on_generation parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed method named '{on_generation.__code__.co_name}' accepts {len(inspect.signature(on_generation).parameters)} parameter(s).")
            # Check if the on_generation is a function.
            elif inspect.isfunction(on_generation):
                # Check if the on_generation function accepts only a single parameter.
                if len(inspect.signature(on_generation).parameters) == 1:
                    self.on_generation = on_generation
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The function assigned to the on_generation parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{on_generation.__code__.co_name}' accepts {len(inspect.signature(on_generation).parameters)} parameter(s).")
            elif callable(on_generation) and not inspect.isclass(on_generation):
                # The object must have the __call__() method.
                if hasattr(on_generation, '__call__'):
                    # Check if the __call__() method accepts 1 parameter.
                    if len(inspect.signature(on_generation).parameters) == 1:
                        # The on_generation class instance assigned to the on_generation parameter is validated.
                        self.on_generation = on_generation
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When 'on_generation' is assigned a class instance, then its __call__ method must accept only 1 parameter representing the instance of the genetic algorithm.\n\nThe passed instance of the class named '{on_generation.__class__.__name__}' accepts {len(inspect.signature(on_generation).parameters)} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise ValueError("When 'on_generation' is assigned a class instance, then its __call__ method must be implemented and accept 1 parameter.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the on_generation parameter is expected to be of type function but {type(on_generation)} found.")
        else:
            self.on_generation = None

        # Check if the on_stop exists.
        if not (on_stop is None):
            # Check if the on_stop is a method.
            if inspect.ismethod(on_stop):
                # Check if the on_stop method accepts 2 parameters.
                if len(inspect.signature(on_stop).parameters) == 2:
                    self.on_stop = on_stop
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The method assigned to the on_stop parameter must accept 2 parameters:\n1) The instance of the genetic algorithm.\n2) A list of the fitness values of the solutions in the last population.\n\nThe passed method named '{on_stop.__code__.co_name}' accepts {len(inspect.signature(on_stop).parameters)} parameter(s).")
            # Check if the on_stop is a function.
            elif inspect.isfunction(on_stop):
                # Check if the on_stop function accepts 2 parameters.
                if len(inspect.signature(on_stop).parameters) == 2:
                    self.on_stop = on_stop
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The function assigned to the on_stop parameter must accept 2 parameters representing the instance of the genetic algorithm and a list of the fitness values of the solutions in the last population.\nThe passed function named '{on_stop.__code__.co_name}' accepts {len(inspect.signature(on_stop).parameters)} parameter(s).")
            elif callable(on_stop) and not inspect.isclass(on_stop):
                # The object must have the __call__() method.
                if hasattr(on_stop, '__call__'):
                    # Check if the __call__() method accepts 2 parameters.
                    if len(inspect.signature(on_stop).parameters) == 2:
                        # The on_stop class instance assigned to the on_stop parameter is validated.
                        self.on_stop = on_stop
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When 'on_stop' is assigned a class instance, then its __call__ method must accept 2 parameters: \n1) The instance of the genetic algorithm.\n2) A list of the fitness values of the solutions in the last population.\n\nThe passed instance of the class named '{on_stop.__class__.__name__}' accepts {len(inspect.signature(on_stop).parameters)} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise ValueError("When 'on_stop' is assigned a class instance, then its __call__ method must be implemented and accept 2 parameters.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the 'on_stop' parameter is expected to be of type function but {type(on_stop)} found.")
        else:
            self.on_stop = None

    def _validate_stop_criteria(self,
                                stop_criteria):
        """
        Validate the ``stop_criteria`` parameter and store the parsed
        criteria on ``self.stop_criteria`` for later use by ``run``.
        Each criterion follows the form ``"keyword_value"`` (or
        ``"keyword_v1_v2_..."`` for multi-objective ``reach``).
        Supported keywords:

        - ``"reach"``: stop when the best fitness is at least the
          target value.
        - ``"saturate"``: stop when the best fitness does not change
          for the given number of generations.
        - ``"time"``: stop when the time spent inside ``run()`` is
          at least the given number of seconds.
        - ``"evaluations"``: stop when the number of fitness function
          calls made inside ``run()`` reaches the given count.

        Parameters
        ----------
        stop_criteria : str, list, tuple, or None
            A single criterion string, an iterable of criterion
            strings, or ``None`` to run for all generations.

        Raises
        ------
        TypeError
            If ``stop_criteria`` is not a string, list, tuple, or
            None, or if a list element is not a string.
        ValueError
            If a criterion uses an unknown keyword or its value is
            not a number.
        """
        self.stop_criteria = []
        self.supported_stop_words = ["reach", "saturate", "time", "evaluations"]
        if stop_criteria is None:
            # None: Stop after passing through all generations.
            self.stop_criteria = None
        elif type(stop_criteria) is str:
            # reach_{target_fitness}: Stop if the target fitness value is reached.
            # saturate_{num_generations}: Stop if the fitness value does not change (saturates) for the given number of generations.
            criterion = stop_criteria.split("_")
            stop_word = criterion[0]
            # criterion[1] might be a single or multiple numbers.
            number = criterion[1:]
            if stop_word in self.supported_stop_words:
                pass
            else:
                self.valid_parameters = False
                raise ValueError(f"In the 'stop_criteria' parameter, the supported stop words are '{self.supported_stop_words}' but '{stop_word}' found.")

            if len(criterion) == 2:
                # There is only a single number.
                number = number[0]
                if number.replace(".", "").replace("-", "").isnumeric():
                    number = float(number)
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The value following the stop word in the 'stop_criteria' parameter must be a number but the value ({number}) of type {type(number)} found.")

                self.stop_criteria.append([stop_word, number])
            elif len(criterion) > 2:
                number = self.validate_multi_stop_criteria(stop_word, number)
                self.stop_criteria.append([stop_word] + number)
            else:
                self.valid_parameters = False
                raise ValueError(f"The format of a single criterion in the 'stop_criteria' parameter is 'word_number' but '{stop_criteria}' found.")

        elif type(stop_criteria) in [list, tuple, numpy.ndarray]:
            # Remove duplicate criteria by converting the list to a set then back to a list.
            stop_criteria = list(set(stop_criteria))
            for idx, val in enumerate(stop_criteria):
                if type(val) is str:
                    criterion = val.split("_")
                    stop_word = criterion[0]
                    number = criterion[1:]
                    if len(criterion) == 2:
                        # There is only a single number.
                        number = number[0]
                        if stop_word in self.supported_stop_words:
                            pass
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"In the 'stop_criteria' parameter, the supported stop words are {self.supported_stop_words} but '{stop_word}' found.")

                        if number.replace(".", "").replace("-", "").isnumeric():
                            number = float(number)
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"The value following the stop word in the 'stop_criteria' parameter must be a number but the value ({number}) of type {type(number)} found.")

                        self.stop_criteria.append([stop_word, number])
                    elif len(criterion) > 2:
                        number = self.validate_multi_stop_criteria(stop_word, number)
                        self.stop_criteria.append([stop_word] + number)
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The format of a single criterion in the 'stop_criteria' parameter is 'word_number' but {criterion} found.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"When the 'stop_criteria' parameter is assigned a tuple/list/numpy.ndarray, then its elements must be strings but the value ({val}) of type {type(val)} found at index {idx}.")
        else:
            self.valid_parameters = False
            raise TypeError(f"The expected value of the 'stop_criteria' is a single string or a list/tuple/numpy.ndarray of strings but the value ({stop_criteria}) of type {type(stop_criteria)} found.")
    
    def _validate_parallel_processing(self,
                                      parallel_processing):
        """
        Validate the ``parallel_processing`` parameter and store the
        parsed value on ``self.parallel_processing``. Supported forms:

        - ``None`` or ``0``: no parallel processing.
        - positive int N: use up to N threads.
        - ``["thread", N]`` or ``["process", N]``: pick the executor
          family and the worker count (``N`` may be a positive int or
          ``None`` for the default).

        Parameters
        ----------
        parallel_processing : None, int, list, or tuple
            The parallel processing specification.

        Raises
        ------
        TypeError
            If ``parallel_processing`` is of an unsupported type.
        ValueError
            If the first element is not ``"process"`` / ``"thread"``,
            the worker count is invalid, or the list length is not 2.
        """
        # Validate the parallel_processing parameter.
        if parallel_processing is None:
            self.parallel_processing = None
        elif type(parallel_processing) in self.supported_int_types:
            if parallel_processing > 0:
                self.parallel_processing = ["thread", parallel_processing]
            else:
                self.valid_parameters = False
                raise ValueError(f"When the 'parallel_processing' parameter is assigned an integer, then the integer must be positive but the value ({parallel_processing}) found.")
        elif type(parallel_processing) in [list, tuple]:
            if len(parallel_processing) == 2:
                if type(parallel_processing[0]) is str:
                    if parallel_processing[0] in ["process", "thread"]:
                        if (type(parallel_processing[1]) in self.supported_int_types and parallel_processing[1] > 0) or (parallel_processing[1] == 0) or (parallel_processing[1] is None):
                            if parallel_processing[1] == 0:
                                # If the number of processes/threads is 0, this means no parallel processing is used. It is equivalent to setting parallel_processing=None.
                                self.parallel_processing = None
                            else:
                                # Whether the second value is None or a positive integer.
                                self.parallel_processing = parallel_processing
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then the second element must be an integer but the value ({parallel_processing[1]}) of type {type(parallel_processing[1])} found.")
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then the value of the first element must be either 'process' or 'thread' but the value ({parallel_processing[0]}) found.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then the first element must be of type 'str' but the value ({parallel_processing[0]}) of type {type(parallel_processing[0])} found.")
            else:
                self.valid_parameters = False
                raise ValueError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then it must have 2 elements but ({len(parallel_processing)}) found.")
        else:
            self.valid_parameters = False
            raise ValueError(f"Unexpected value ({parallel_processing}) of type ({type(parallel_processing)}) assigned to the 'parallel_processing' parameter. The accepted values for this parameter are:\n1) None: (Default) It means no parallel processing is used.\n2) A positive integer referring to the number of threads to be used (i.e. threads, not processes, are used.\n3) list/tuple: If a list or a tuple of exactly 2 elements is assigned, then:\n\t*1) The first element can be either 'process' or 'thread' to specify whether processes or threads are used, respectively.\n\t*2) The second element can be:\n\t\t**1) A positive integer to select the maximum number of processes or threads to be used.\n\t\t**2) 0 to indicate that parallel processing is not used. This is identical to setting 'parallel_processing=None'.\n\t\t**3) None to use the default value as calculated by the concurrent.futures module.")

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
        and ``save_solutions``. Also re-checks the
        ``mutation_percent_genes`` / ``mutation_num_genes`` pair now
        that ``num_genes`` has been resolved.

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

        # Validate num_generations
        if type(num_generations) in self.supported_int_types:
            if num_generations >= 0:
                self.num_generations = num_generations
            else:
                raise ValueError(f"The value assigned to the 'num_generations' parameter must be a non-negative integer >= 0. But the value {num_generations} found.")
        else:
            self.valid_parameters = False
            raise ValueError(f"Unexpected value ({num_generations}) of type ({type(num_generations)}) assigned to the 'num_generations' parameter. It must be assigned a non-negative integer.")

        # Validate save_best_solutions
        if type(save_best_solutions) is bool:
            if save_best_solutions == True:
                if not self.suppress_warnings:
                    warnings.warn("Use the 'save_best_solutions' parameter with caution as it may cause memory overflow when either the number of generations or number of genes is large.")
        else:
            self.valid_parameters = False
            raise TypeError(f"The value passed to the 'save_best_solutions' parameter must be of type bool but {type(save_best_solutions)} found.")

        # Validate save_solutions
        if type(save_solutions) is bool:
            if save_solutions == True:
                if not self.suppress_warnings:
                    warnings.warn("Use the 'save_solutions' parameter with caution as it may cause memory overflow when either the number of generations, number of genes, or number of solutions in population is large.")
        else:
            self.valid_parameters = False
            raise TypeError(f"The value passed to the 'save_solutions' parameter must be of type bool but {type(save_solutions)} found.")

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

        # At this point, all necessary parameters validation is done successfully, and we are sure that the parameters are valid.
        # Set to True when all the parameters passed in the GA class constructor are valid.
        self.valid_parameters = True

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

        self._validate_gene_space(gene_space)

        self._validate_init_range(init_range_low,
                                  init_range_high,
                                  num_genes,
                                  initial_population)

        self._validate_gene_type(gene_type,
                                 num_genes,
                                 initial_population)

        # Call the unpack_gene_space() method in the pygad.helper.unique.Unique class.
        self.gene_space_unpacked = self.unpack_gene_space(range_min=self.init_range_low,
                                                          range_max=self.init_range_high)

        self._build_initial_population(initial_population,
                                       sol_per_pop,
                                       num_genes,
                                       gene_space,
                                       allow_duplicate_genes,
                                       gene_constraint)

        # In case the 'gene_space' parameter is nested, then make sure the number of its elements equals to the number of genes.
        if self.gene_space_nested:
            if len(gene_space) != self.num_genes:
                self.valid_parameters = False
                raise ValueError(f"When the parameter 'gene_space' is nested, then its length must be equal to the value passed to the 'num_genes' parameter. Instead, length of gene_space ({len(gene_space)}) != num_genes ({self.num_genes})")

        self._validate_mutation_range(random_mutation_min_val,
                                      random_mutation_max_val)

        self._validate_gene_constraint(gene_constraint)

        # Validating the number of parents to be selected for mating (num_parents_mating)
        if num_parents_mating <= 0:
            self.valid_parameters = False
            raise ValueError(f"The number of parents mating (num_parents_mating) parameter must be > 0 but ({num_parents_mating}) found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n")

        # Validating the number of parents to be selected for mating: num_parents_mating
        if num_parents_mating > self.sol_per_pop:
            self.valid_parameters = False
            raise ValueError(f"The number of parents to select for mating ({num_parents_mating}) cannot be greater than the number of solutions in the population ({self.sol_per_pop}) (i.e., num_parents_mating must always be <= sol_per_pop).\n")

        self.num_parents_mating = num_parents_mating

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

    def validate_multi_stop_criteria(self, stop_word, number):
        """
        Validate one ``(keyword, value)`` element of a
        multi-objective stop criterion. Only the ``"reach"`` keyword
        accepts multiple numeric values (one per objective).

        Parameters
        ----------
        stop_word : str
            The criterion keyword. Must be ``"reach"`` to be valid for
            the multi-objective case.
        number : str
            The numeric value (as it appeared in the criterion
            string). The method parses it into a float.

        Returns
        -------
        number : float
            The parsed numeric value.

        Raises
        ------
        ValueError
            If ``stop_word`` is not ``"reach"``, or ``number`` is not
            a numeric string.
        """
        if stop_word == 'reach':
            pass
        else:
            self.valid_parameters = False
            raise ValueError(f"Passing multiple numbers following the keyword in the 'stop_criteria' parameter is expected only with the 'reach' keyword but the keyword ({stop_word}) found.")

        for idx, num in enumerate(number):
            if num.replace(".", "").replace("-", "").isnumeric():
                number[idx] = float(num)
            else:
                self.valid_parameters = False
                raise ValueError(f"The value(s) following the stop word in the 'stop_criteria' parameter must be numeric but the value ({num}) of type {type(num)} found.")
        return number
