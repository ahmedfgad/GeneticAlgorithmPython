"""
Internal helpers for describing and drawing a configured PyGAD lifecycle.

The description reads GA settings and already available fitness only.
It never runs user code. Matplotlib is imported only when drawing.
"""

import reprlib
import textwrap

import numpy


def _lifecycle_handler_name(handler):
    """Return a readable function, method, or callable-instance name."""
    return getattr(handler, "__name__", type(handler).__name__) + "()"


def _lifecycle_parameter_text(value):
    """Abbreviate large settings so gene lists cannot fill the chart."""
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    if isinstance(value, numpy.ndarray):
        # Do not change NumPy's global print options to format a chart.
        array_text = numpy.array2string(value, threshold=6, edgeitems=2,
                                       max_line_width=60).replace("\n", " ")
        return array_text if len(array_text) <= 100 else array_text[:97] + "..."
    formatter = reprlib.Repr()
    formatter.maxlist = 4
    formatter.maxtuple = 4
    formatter.maxdict = 3
    formatter.maxstring = 65
    formatter.maxother = 65
    return formatter.repr(value)


def _describe_lifecycle(ga_instance, show_parameters=True):
    """
    Build stages, connections, and configuration for one GA instance.

    Stage identifiers describe execution points rather than operator
    names. Both fitness evaluations therefore remain distinct, even
    though they call the same fitness function. Connections include
    the zero-generation exit and the return to the loop's entry.

    Returns
    -------
    lifecycle : dict
        ``stages`` is an ordered list of stage dictionaries.
        ``connections`` lists source, target, label, and route.
        ``configuration`` lists labels and abbreviated setting values.
        All helpers in this module are internal.
    """
    stages = []

    def add_stage(identifier, title, kind="operation", handler=None, parameters=None):
        """Append a stage, keeping handler names in the compact view."""
        details = []
        if handler is not None:
            details.append(_lifecycle_handler_name(handler))
        if show_parameters and parameters:
            details.extend(parameters)
        stages.append({"id": identifier, "title": title,
                       "kind": kind, "details": details})

    def add_callback(name):
        """Only show callbacks that the GA instance actually uses."""
        callback = getattr(ga_instance, name)
        if callback is not None:
            add_stage(name, name + "()", "callback", handler=callback)

    population_shape = f"({ga_instance.sol_per_pop}, {ga_instance.num_genes})"
    offspring_shape = f"({ga_instance.num_offspring}, {ga_instance.num_genes})"
    fitness_parameters = []
    if ga_instance.last_generation_fitness is not None:
        fitness_shape = numpy.shape(ga_instance.last_generation_fitness)
        fitness_parameters.append(f"Population fitness: {fitness_shape}")
    if ga_instance.fitness_batch_size is not None and ga_instance.fitness_batch_size > 1:
        fitness_parameters.append(f"Batch size: {ga_instance.fitness_batch_size}")
    fitness_parameters.append("Reuse available fitness where applicable")

    add_stage("population", "Population Ready", "population",
              parameters=[f"Shape: {population_shape}", "Prepared before run()"])
    add_callback("on_start")
    add_stage("initial_fitness", "Evaluate Initial Fitness", handler=ga_instance.fitness_func,
              parameters=fitness_parameters)
    if ga_instance.parent_selection_type in ("nsga3", "tournament_nsga3"):
        add_stage("reference_points", "Prepare NSGA-III Reference Points",
                  parameters=[f"Divisions: {ga_instance.nsga3_num_divisions}",
                              "Grow population and evaluate added solutions if needed"])

    add_stage("generation_check", "Generations Remaining?", "decision",
              parameters=[f"{ga_instance.num_generations} generations per run()"])
    add_callback("on_fitness")

    selection_parameters = [f"Parents: ({ga_instance.num_parents_mating}, {ga_instance.num_genes})"]
    if ga_instance.parent_selection_type in ("tournament", "tournament_nsga2", "tournament_nsga3"):
        selection_parameters.append(f"Tournament size: {ga_instance.K_tournament}")
    add_stage("selection", "Select Parents", handler=ga_instance.select_parents,
              parameters=selection_parameters)
    add_callback("on_parents")

    crossover_parameters = [f"Offspring: {offspring_shape}"]
    if ga_instance.crossover_type is not None:
        if not callable(ga_instance.crossover_type):
            if ga_instance.crossover_probability is not None:
                crossover_parameters.append(f"Probability: {ga_instance.crossover_probability}")
            if ga_instance.crossover_type == "sbx":
                crossover_parameters.append(f"Distribution index: {ga_instance.sbx_crossover_eta}")
        add_stage("crossover", "Crossover", handler=ga_instance.crossover,
                  parameters=crossover_parameters)
    # These callbacks run even when the corresponding operator is None.
    add_callback("on_crossover")

    mutation_parameters = [f"Offspring: {offspring_shape}"]
    if ga_instance.mutation_type is not None:
        if ga_instance.mutation_type in ("random", "adaptive", "polynomial"):
            if ga_instance.mutation_type == "polynomial":
                probability = ga_instance.mutation_probability
                if probability is None:
                    probability = 1.0 / ga_instance.num_genes
                mutation_parameters.extend([f"Probability per gene: {probability}",
                                            f"Distribution index: {ga_instance.polynomial_mutation_eta}"])
            elif ga_instance.mutation_probability is not None:
                mutation_parameters.append("Probability per gene: " +
                                           _lifecycle_parameter_text(ga_instance.mutation_probability))
            else:
                mutation_parameters.append("Genes to mutate: " +
                                           _lifecycle_parameter_text(ga_instance.mutation_num_genes))
            if ga_instance.mutation_type in ("random", "adaptive"):
                if ga_instance.gene_space is None:
                    mutation_parameters.append("Random range: " + _lifecycle_parameter_text(
                        (ga_instance.random_mutation_min_val, ga_instance.random_mutation_max_val)))
                else:
                    mutation_parameters.append("Use configured gene space")
                mutation_parameters.append("Replace gene values" if ga_instance.mutation_by_replacement
                                           else "Add random values when no gene space is set")
            if ga_instance.mutation_type == "adaptive":
                mutation_parameters.append("Evaluate offspring fitness to choose mutation amount")
                mutation_parameters.append("Control values: low-quality, high-quality offspring")
        add_stage("mutation", "Mutation", handler=ga_instance.mutation,
                  parameters=mutation_parameters)
    add_callback("on_mutation")

    # Elitism overrides keep_parents; display the effective policy only.
    if ga_instance.keep_elitism > 0:
        retention_text = f"Keep {ga_instance.keep_elitism} elite solution(s)"
    elif ga_instance.keep_parents == -1:
        retention_text = f"Keep all {ga_instance.num_parents_mating} selected parents"
    elif ga_instance.keep_parents > 0:
        retention_text = f"Keep {ga_instance.keep_parents} parent(s)"
    else:
        retention_text = "Keep no parents or elite solutions"
    add_stage("update_population", "Update Population",
              parameters=[retention_text, f"Add {ga_instance.num_offspring} offspring",
                          f"Population: {population_shape}"])
    add_stage("generation_fitness", "Evaluate Updated Population", handler=ga_instance.fitness_func,
              parameters=fitness_parameters)
    add_callback("on_generation")

    stopping_parameters = []
    if ga_instance.on_generation is not None:
        stopping_parameters.append('on_generation() returns "stop"')
    if ga_instance.stop_criteria is not None:
        for criterion in ga_instance.stop_criteria:
            stopping_parameters.append("_".join(str(value) for value in criterion))
    if stopping_parameters:
        add_stage("early_stop", "Stop Early?", "decision",
                  parameters=["Any condition below:"] + stopping_parameters)
    last_generation_stage = stages[-1]["id"]

    add_stage("finalize", "Finalize Results",
              parameters=["Refresh final parents and elitism", "Record best solution fitness"])
    add_callback("on_stop")
    add_stage("complete", "Run Complete", "end")

    connections = []
    for source, target in zip(stages, stages[1:]):
        # The body returns to the generation check, rather than falling
        # through to finalization after the first generation.
        if source["id"] == last_generation_stage:
            continue
        label = "Yes" if source["id"] == "generation_check" else ""
        connections.append({"source": source["id"], "target": target["id"],
                            "label": label, "route": "forward"})
    connections.append({"source": "generation_check", "target": "finalize",
                        "label": "No", "route": "finish"})
    connections.append({"source": last_generation_stage, "target": "generation_check",
                        "label": "No" if stopping_parameters else "Next Generation",
                        "route": "repeat"})
    if stopping_parameters:
        connections.append({"source": "early_stop", "target": "finalize",
                            "label": "Yes", "route": "finish"})

    configuration = []
    if show_parameters:
        if ga_instance.gene_type_single:
            gene_types = [ga_instance.gene_type]
        else:
            gene_types = ga_instance.gene_type
        gene_type_names = []
        for gene_type, precision in gene_types:
            name = getattr(gene_type, "__name__", str(gene_type))
            if precision is not None:
                name += f" ({precision} decimal places)"
            gene_type_names.append(name)
        gene_type_text = (gene_type_names[0] if ga_instance.gene_type_single
                          else "Per gene: " + _lifecycle_parameter_text(gene_type_names))
        configuration.extend([("Population", population_shape),
                              ("Generations Per Run", str(ga_instance.num_generations)),
                              ("Gene Type", gene_type_text)])
        if ga_instance.last_generation_fitness is not None:
            first_fitness = ga_instance.last_generation_fitness[0]
            objectives = len(first_fitness) if isinstance(first_fitness, (list, tuple, numpy.ndarray)) else 1
            configuration.append(("Objectives", str(objectives)))
        else:
            configuration.append(("Objectives", "Known after fitness evaluation"))
        if ga_instance.gene_space is not None:
            configuration.append(("Gene Space", _lifecycle_parameter_text(ga_instance.gene_space)))
        else:
            configuration.append(("Initial Population Range", _lifecycle_parameter_text(
                (ga_instance.init_range_low, ga_instance.init_range_high))))
        if ga_instance.gene_constraint is not None:
            constraint_count = sum(constraint is not None for constraint in ga_instance.gene_constraint)
            configuration.append(("Gene Constraints", f"{constraint_count} constrained gene(s)"))
        configuration.append(("Allow Duplicate Genes", str(ga_instance.allow_duplicate_genes)))
        if ga_instance.parallel_processing is not None:
            configuration.append(("Parallel Fitness Evaluation", _lifecycle_parameter_text(ga_instance.parallel_processing)))
        if ga_instance.random_seed is not None:
            configuration.append(("Random Seed", str(ga_instance.random_seed)))
        configuration.extend([("Save Solutions", str(ga_instance.save_solutions)),
                              ("Save Best Solutions", str(ga_instance.save_best_solutions))])

    return {"stages": stages, "connections": connections,
            "configuration": configuration}


def _draw_lifecycle(lifecycle, matplt, title, font_size):
    """
    Render a lifecycle description as a vertically arranged flowchart.

    Card heights follow wrapped text lengths. The repeat connection
    runs on the left and exit connections run on the right, keeping
    arrows outside the cards. Coordinate units correspond to inches
    at the default font size; scaling the figure keeps text readable.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
    from matplotlib.path import Path
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    text_measurement = TextToPath()

    def wrap_text(text, width_inches, text_font_size, weight="normal"):
        """Wrap using font measurements, including long handler names."""
        font_properties = FontProperties(size=text_font_size, weight=weight)
        # Coordinates scale with font_size, so compare widths at the
        # corresponding figure scale. Math characters remain literal,
        # just as they do in the rendered labels (parse_math=False).
        available_width = width_inches * 72 * font_size / 11
        approximate_columns = max(1, int(available_width / (text_font_size * 0.55)))
        wrapped_lines = []
        for line in textwrap.wrap(text, width=approximate_columns):
            while line:
                fitted_length = len(line)
                while fitted_length > 1:
                    measured_width, _, _ = text_measurement.get_text_width_height_descent(
                        line[:fitted_length], font_properties, ismath=False)
                    if measured_width <= available_width:
                        break
                    fitted_length -= 1
                if fitted_length < len(line):
                    last_space = line.rfind(" ", 0, fitted_length + 1)
                    if last_space > 0:
                        fitted_length = last_space
                wrapped_lines.append(line[:fitted_length].strip())
                line = line[fitted_length:].lstrip()
        return wrapped_lines

    colors = {"operation": ("#eef4ff", "#4773ba"),
              "population": ("#e8eef9", "#4773ba"),
              "callback": ("#e6f5ef", "#26866c"),
              "decision": ("#fff4da", "#b38325"),
              "end": ("#253e65", "#253e65")}
    text_color = "#25354b"
    arrow_color = "#718198"
    stage_center = 3.4
    stage_width = 4.5
    stage_gap = 0.34
    stage_positions = {}
    wrapped_stages = []
    figure_width = 10.4 if lifecycle["configuration"] else 7.0
    title_lines = wrap_text(title, figure_width - 1.4, font_size * 1.4, "bold") or [""]
    current_top = 0.82 + 0.27 * len(title_lines)

    for stage in lifecycle["stages"]:
        # Put decision questions and their conditions inside the middle
        # half of the diamond, where the sloping sides leave room for text.
        text_width = stage_width / 2 - 0.2 if stage["kind"] == "decision" else stage_width - 0.4
        title_text = wrap_text(stage["title"], text_width, font_size, "bold")
        detail_lines = []
        for detail in stage["details"]:
            detail_lines.extend(wrap_text(detail, text_width, font_size * 0.88))
        stage_height = 0.30 + 0.20 * len(title_text) + 0.17 * len(detail_lines)
        if stage["kind"] == "decision":
            stage_height = max(0.95, 2 * stage_height)
        stage_positions[stage["id"]] = {"top": current_top, "bottom": current_top + stage_height,
                                         "center": current_top + stage_height / 2,
                                         "height": stage_height}
        wrapped_stages.append((stage, title_text, detail_lines))
        current_top += stage_height + stage_gap

    configuration_lines = []
    for label, value in lifecycle["configuration"]:
        configuration_lines.append((wrap_text(label, 2.65, font_size * 0.86, "bold"),
                                    wrap_text(value, 2.65, font_size * 0.84)))
    configuration_height = 0.65 + sum(0.20 * len(label_lines) + 0.17 * len(value_lines) + 0.17
                                     for label_lines, value_lines in configuration_lines)
    figure_height = max(current_top + 0.55, configuration_height + 2.0)
    figure_scale = font_size / 11
    fig, axes = matplt.subplots(figsize=(figure_width * figure_scale, figure_height * figure_scale))
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes.set_xlim(0, figure_width)
    axes.set_ylim(figure_height, 0)
    axes.axis("off")
    axes.text(0.7, 0.28, "\n".join(title_lines), fontsize=font_size * 1.4,
              weight="bold", color=text_color, va="top", parse_math=False)
    axes.text(0.7, current_top + 0.12, "Configured Lifecycle | Operators / Callbacks / Decisions",
              fontsize=font_size * 0.82, color=arrow_color, va="top", parse_math=False)

    for stage, title_text, detail_lines in wrapped_stages:
        position = stage_positions[stage["id"]]
        fill_color, border_color = colors[stage["kind"]]
        if stage["kind"] == "decision":
            card = Polygon([(stage_center, position["top"]),
                            (stage_center + stage_width / 2, position["center"]),
                            (stage_center, position["bottom"]),
                            (stage_center - stage_width / 2, position["center"])],
                           facecolor=fill_color, edgecolor=border_color, linewidth=1.2)
        else:
            card = FancyBboxPatch((stage_center - stage_width / 2, position["top"]),
                                 stage_width, position["height"],
                                 boxstyle="round,pad=0,rounding_size=0.10",
                                 facecolor=fill_color, edgecolor=border_color, linewidth=1.2)
        axes.add_patch(card)
        text_top = position["center"] - (0.20 * len(title_text) + 0.17 * len(detail_lines)) / 2
        stage_text_color = "#ffffff" if stage["kind"] == "end" else text_color
        axes.text(stage_center, text_top, "\n".join(title_text), ha="center", va="top",
                  fontsize=font_size, weight="bold", color=stage_text_color,
                  linespacing=1.2, parse_math=False)
        if detail_lines:
            axes.text(stage_center, text_top + 0.20 * len(title_text) + 0.03,
                      "\n".join(detail_lines), ha="center", va="top",
                      fontsize=font_size * 0.88, color=stage_text_color,
                      linespacing=1.2, parse_math=False)

    for connection in lifecycle["connections"]:
        source = stage_positions[connection["source"]]
        target = stage_positions[connection["target"]]
        if connection["route"] == "forward":
            points = [(stage_center, source["bottom"]), (stage_center, target["top"])]
            label_position = (stage_center + 0.15, (source["bottom"] + target["top"]) / 2)
        elif connection["route"] == "finish":
            exit_column = 6.25
            points = [(stage_center + stage_width / 2, source["center"]),
                      (exit_column, source["center"]), (exit_column, target["center"]),
                      (stage_center + stage_width / 2, target["center"])]
            label_position = (5.83, source["center"] - 0.13)
        else:
            repeat_column = 0.52
            points = [(stage_center - stage_width / 2, source["center"]),
                      (repeat_column, source["center"]), (repeat_column, target["center"]),
                      (stage_center - stage_width / 2, target["center"])]
            label_position = (repeat_column - 0.20, (source["center"] + target["center"]) / 2)
        arrow_path = Path(points, [Path.MOVETO] + [Path.LINETO] * (len(points) - 1))
        axes.add_patch(FancyArrowPatch(path=arrow_path, arrowstyle="-|>",
                                       mutation_scale=11 * figure_scale,
                                       color=arrow_color, linewidth=1.2, zorder=0))
        if connection["label"]:
            axes.text(*label_position, connection["label"], fontsize=font_size * 0.82,
                      color=arrow_color, va="center",
                      rotation=90 if connection["route"] == "repeat" else 0,
                      parse_math=False)

    if configuration_lines:
        panel_left = 6.85
        panel_top = stage_positions["population"]["top"]
        axes.add_patch(FancyBboxPatch((panel_left, panel_top), 3.15, configuration_height,
                                     boxstyle="round,pad=0,rounding_size=0.10",
                                     facecolor="#f7f9fc", edgecolor="#dce3ed"))
        axes.text(panel_left + 0.22, panel_top + 0.22, "Configuration", weight="bold",
                  fontsize=font_size * 1.05, color=text_color, va="top", parse_math=False)
        configuration_top = panel_top + 0.65
        for label_lines, value_lines in configuration_lines:
            axes.text(panel_left + 0.22, configuration_top, "\n".join(label_lines), weight="bold",
                      fontsize=font_size * 0.86, color=text_color, va="top", parse_math=False)
            axes.text(panel_left + 0.22, configuration_top + 0.20 * len(label_lines), "\n".join(value_lines),
                      fontsize=font_size * 0.84, color=text_color, va="top",
                      linespacing=1.2, parse_math=False)
            configuration_top += 0.20 * len(label_lines) + 0.17 * len(value_lines) + 0.17

    # Axes with axis("off") still occupy their full canvas in a tight export.
    # Fit the canvas to the actual labels, cards, and connectors instead.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = Bbox.union([artist.get_window_extent(renderer)
                         for artist in list(axes.texts) + list(axes.patches)])
    data_bounds = bounds.transformed(axes.transData.inverted())
    margin = 0.08
    left, right = data_bounds.xmin - margin, data_bounds.xmax + margin
    top, bottom = data_bounds.ymin - margin, data_bounds.ymax + margin
    axes.set_xlim(left, right)
    axes.set_ylim(bottom, top)
    fig.set_size_inches((right - left) * figure_scale,
                        (bottom - top) * figure_scale)
    return fig
