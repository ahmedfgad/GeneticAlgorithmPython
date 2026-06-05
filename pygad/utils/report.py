"""
PDF report generation for a completed GA run.

The ``Report`` class is mixed into ``pygad.GA``. Call
``ga_instance.generate_report(filename)`` after ``run()`` finishes to
get a PDF that bundles the run configuration, the best solution(s),
and every applicable plot.

The report relies on two optional dependencies, ``matplotlib`` and
``reportlab``. Both are installed by ``pip install pygad[report]``.
The imports happen on first use so users who never call
``generate_report`` keep the lean install.

The title page shows the PyGAD logo. The image is bundled with the
package, so no network access is needed. If the file is missing the
report is still built without it.
"""

import io
import os


# The PyGAD logo shown on the title page of the report. The file sits
# next to this module and is shipped with the package.
LOGO_FILENAME = "pygad_logo.png"


# Default order in which sections appear in the report. Used by
# ``generate_report`` when the caller does not pass an explicit
# ``sections`` list.
REPORT_DEFAULT_SECTIONS = ("title",
                           "configuration",
                           "run_summary",
                           "best_solution",
                           "plots",
                           "notes")

# Inventory of every plot the report can include, in the order the
# report renders them. Each entry has:
#   name        : section label that appears in the PDF
#   method      : name of the GA method that draws the plot
#   requires_moo: True if the plot only works for multi-objective runs
#   requires    : tuple of attribute names that must be truthy
#   kwargs      : dict of keyword arguments passed to the method
REPORT_PLOTS = (
    {"name": "Best fitness per generation",
     "method": "plot_fitness",
     "requires_moo": False,
     "requires": (),
     "kwargs": {}},
    {"name": "New solutions per generation",
     "method": "plot_new_solution_rate",
     "requires_moo": False,
     "requires": ("save_solutions",),
     "kwargs": {}},
    {"name": "Per-gene drift",
     "method": "plot_genes",
     "requires_moo": False,
     "requires": ("save_solutions",),
     "kwargs": {"graph_type": "plot", "solutions": "all"}},
    {"name": "Min/Mean/Max fitness band",
     "method": "plot_fitness_band",
     "requires_moo": False,
     "requires": ("save_solutions",),
     "kwargs": {}},
    {"name": "Population diversity",
     "method": "plot_population_diversity",
     "requires_moo": False,
     "requires": ("save_solutions",),
     "kwargs": {}},
    {"name": "Pareto front",
     "method": "plot_pareto_front_curve",
     "requires_moo": True,
     "requires": (),
     "kwargs": {},
     "max_objectives": 3},
    {"name": "Parallel coordinates of the Pareto front",
     "method": "plot_pareto_front_pcp",
     "requires_moo": True,
     "requires": (),
     "kwargs": {}},
    {"name": "Pairwise scatter matrix of the Pareto front",
     "method": "plot_pareto_front_scatter_matrix",
     "requires_moo": True,
     "requires": (),
     "kwargs": {}},
    {"name": "Pareto front heatmap",
     "method": "plot_pareto_front_heatmap",
     "requires_moo": True,
     "requires": (),
     "kwargs": {}},
    {"name": "Hypervolume of the non-dominated set",
     "method": "plot_non_dominated_hypervolume",
     "requires_moo": True,
     "requires": ("save_solutions",),
     "kwargs": {}},
    {"name": "Pareto front evolution",
     "method": "plot_pareto_front_evolution",
     "requires_moo": True,
     "requires": ("save_solutions",),
     "kwargs": {"every_k": 10},
     "max_objectives": 3},
)


# Names of the GA constructor parameters the report shows in the
# configuration table. Grouped by topic so the table reads well.
CONFIGURATION_GROUPS = (
    ("Population", ["num_generations", "num_parents_mating", "sol_per_pop",
                    "num_genes", "init_range_low", "init_range_high",
                    "gene_type", "gene_space", "allow_duplicate_genes",
                    "gene_constraint", "sample_size"]),
    ("Parent selection", ["parent_selection_type", "K_tournament",
                          "nsga3_num_divisions", "keep_parents",
                          "keep_elitism"]),
    ("Crossover", ["crossover_type", "crossover_probability",
                   "sbx_crossover_eta"]),
    ("Mutation", ["mutation_type", "mutation_probability",
                  "mutation_percent_genes", "mutation_num_genes",
                  "polynomial_mutation_eta", "mutation_by_replacement",
                  "random_mutation_min_val", "random_mutation_max_val"]),
    ("Stopping criteria", ["stop_criteria"]),
    ("Run-time", ["fitness_batch_size", "parallel_processing",
                  "random_seed", "suppress_warnings"]),
    ("History", ["save_solutions", "save_best_solutions"]),
)


class Report:

    def __init__(self):
        pass

    def generate_report(self,
                        filename,
                        title=None,
                        sections=None,
                        include_plots=None,
                        figure_size_inches=(7.0, 4.5),
                        notes=None,
                        page_size="letter"):
        """
        Build a PDF report of the current GA run and write it to disk.

        Parameters
        ----------
        filename : str
            Output path. ``.pdf`` is appended automatically if missing.
        title : str or None
            Title shown on the first page. Defaults to ``"PyGAD run
            report"``.
        sections : iterable of str or None
            Sections to include and their order. Valid entries are
            ``"title"``, ``"configuration"``, ``"run_summary"``,
            ``"best_solution"``, ``"plots"``, and ``"notes"``. When
            ``None``, every section is included in their default order.
        include_plots : iterable of str, "all", or None
            Plots to embed under the ``"plots"`` section.
            ``None`` or ``"all"`` lets the report auto-select every
            plot whose preconditions are met by this run (the right
            number of objectives, ``save_solutions`` set, and so on).
            Pass a list of plot method names (e.g.
            ``["plot_fitness", "plot_pareto_front_curve"]``) to include
            only those.
        figure_size_inches : (float, float)
            Width and height (in inches) used when each plot is drawn
            for the report. The figures inside the PDF preserve this
            aspect ratio.
        notes : str or None
            Free-form text rendered in the optional ``"notes"``
            section.
        page_size : str
            ``"letter"`` (default) or ``"A4"``.

        Returns
        -------
        filename : str
            The path of the PDF file that was written.

        Raises
        ------
        ImportError
            If ``reportlab`` or ``matplotlib`` is not installed.
        RuntimeError
            If the GA has not completed at least one generation.
        ValueError
            If ``sections`` or ``include_plots`` contain an unknown
            entry, or if ``page_size`` is unknown.
        """
        if self.generations_completed < 1:
            raise RuntimeError(
                "generate_report() can only be called after at least one "
                "generation has completed. Call run() first.")

        reportlab_modules = _pdf_report_import_reportlab()
        matplt = _pdf_report_import_matplotlib()

        section_list = _pdf_report_resolve_sections(sections)
        page_size_obj = _pdf_report_resolve_page_size(page_size, reportlab_modules)

        if not filename.endswith(".pdf"):
            filename = filename + ".pdf"

        story = []
        styles = reportlab_modules["styles"].getSampleStyleSheet()
        for section_name in section_list:
            if section_name == "title":
                story.extend(_pdf_report_build_title_section(
                    self, title, styles, reportlab_modules))
            elif section_name == "configuration":
                story.extend(_pdf_report_build_configuration_section(
                    self, styles, reportlab_modules))
            elif section_name == "run_summary":
                story.extend(_pdf_report_build_run_summary_section(
                    self, styles, reportlab_modules))
            elif section_name == "best_solution":
                story.extend(_pdf_report_build_best_solution_section(
                    self, styles, reportlab_modules))
            elif section_name == "plots":
                story.extend(_pdf_report_build_plots_section(
                    self,
                    include_plots,
                    figure_size_inches,
                    styles,
                    reportlab_modules,
                    matplt))
            elif section_name == "notes":
                story.extend(_pdf_report_build_notes_section(
                    notes, styles, reportlab_modules))

        doc = reportlab_modules["SimpleDocTemplate"](
            filename,
            pagesize=page_size_obj,
            title=title or "PyGAD run report",
            author="PyGAD",
        )
        doc.build(story)
        return filename


def _pdf_report_import_reportlab():
    """
    Import reportlab on first use. Returns a dict with the names the
    report builder needs, so the calling code does not have to repeat
    the imports.
    """
    try:
        from reportlab.lib import colors, pagesizes, styles
        from reportlab.lib.units import inch
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ImportError(
            "generate_report requires reportlab. Install it with: "
            "pip install pygad[report] (or pip install reportlab)."
        ) from exc
    return {
        "colors": colors,
        "pagesizes": pagesizes,
        "styles": styles,
        "inch": inch,
        "ImageReader": ImageReader,
        "Image": Image,
        "PageBreak": PageBreak,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _pdf_report_import_matplotlib():
    """
    Import matplotlib on first use. The Agg backend is forced so the
    report can be generated in headless environments.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as matplt
    except ImportError as exc:
        raise ImportError(
            "generate_report requires matplotlib. Install it with: "
            "pip install pygad[report] (or pip install matplotlib)."
        ) from exc
    return matplt


def _pdf_report_resolve_sections(sections):
    if sections is None:
        return list(REPORT_DEFAULT_SECTIONS)
    requested = list(sections)
    unknown = set(requested) - set(REPORT_DEFAULT_SECTIONS)
    if unknown:
        raise ValueError(
            f"Unknown report sections: {sorted(unknown)}. Allowed: "
            f"{list(REPORT_DEFAULT_SECTIONS)}.")
    return requested


def _pdf_report_resolve_page_size(page_size, reportlab_modules):
    name = page_size.lower()
    if name == "letter":
        return reportlab_modules["pagesizes"].LETTER
    if name == "a4":
        return reportlab_modules["pagesizes"].A4
    raise ValueError(
        f"Unknown page_size {page_size!r}. Allowed values: 'letter', 'A4'.")


def _pdf_report_is_multi_objective(ga):
    """Return True when the last fitness row is iterable (MOO)."""
    if getattr(ga, "last_generation_fitness", None) is None:
        return False
    first = ga.last_generation_fitness[0]
    return hasattr(first, "__len__")


def _pdf_report_num_objectives(ga):
    """Return the number of objectives, or 1 for single-objective runs."""
    if not _pdf_report_is_multi_objective(ga):
        return 1
    return len(ga.last_generation_fitness[0])


def _pdf_report_build_title_section(ga, title, styles, modules):
    Paragraph = modules["Paragraph"]
    Spacer = modules["Spacer"]
    inch = modules["inch"]

    elements = []

    logo_image = _pdf_report_build_logo_image(modules)
    if logo_image is not None:
        elements.append(logo_image)
        elements.append(Spacer(1, 0.2 * inch))

    title_text = title or "PyGAD run report"
    import pygad as _pygad_module
    subtitle = f"PyGAD version: {_pygad_module.__version__}"
    elements.extend([
        Paragraph(title_text, styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph(subtitle, styles["Normal"]),
        Spacer(1, 0.25 * inch),
    ])
    return elements


def _pdf_report_read_logo_bytes():
    """
    Read the bundled PyGAD logo and return its bytes. Return None if the
    file is missing or cannot be read, so the report is still built
    without it.
    """
    logo_path = os.path.join(os.path.dirname(__file__), LOGO_FILENAME)
    try:
        with open(logo_path, "rb") as logo_file:
            return logo_file.read() or None
    except OSError:
        return None


def _pdf_report_build_logo_image(modules, target_width_inches=2.0):
    """
    Build a centered reportlab Image for the logo, scaled to
    target_width_inches while keeping the aspect ratio. Return None when
    the logo cannot be read, so a missing or broken image never breaks
    the report.
    """
    logo_png = _pdf_report_read_logo_bytes()
    if logo_png is None:
        return None

    Image = modules["Image"]
    ImageReader = modules["ImageReader"]
    inch = modules["inch"]
    try:
        natural_width, natural_height = ImageReader(
            io.BytesIO(logo_png)).getSize()
        if not natural_width or not natural_height:
            return None
        width = target_width_inches * inch
        height = width * natural_height / natural_width
        image = Image(io.BytesIO(logo_png), width=width, height=height)
        image.hAlign = "CENTER"
        return image
    except Exception:
        return None


def _pdf_report_build_configuration_section(ga, styles, modules):
    Paragraph = modules["Paragraph"]
    Spacer = modules["Spacer"]
    Table = modules["Table"]
    TableStyle = modules["TableStyle"]
    colors = modules["colors"]
    inch = modules["inch"]

    elements = [Paragraph("Configuration", styles["Heading1"])]
    for group_name, parameter_names in CONFIGURATION_GROUPS:
        rows = [[Paragraph("<b>Parameter</b>", styles["BodyText"]),
                 Paragraph("<b>Value</b>", styles["BodyText"])]]
        for parameter_name in parameter_names:
            if not hasattr(ga, parameter_name):
                continue
            value = getattr(ga, parameter_name)
            rows.append([
                Paragraph(parameter_name, styles["BodyText"]),
                Paragraph(_pdf_report_format_value(value), styles["BodyText"]),
            ])
        if len(rows) <= 1:
            continue
        elements.append(Paragraph(group_name, styles["Heading3"]))
        table = Table(rows, colWidths=[2.2 * inch, 4.0 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.15 * inch))
    elements.append(modules["PageBreak"]())
    return elements


def _pdf_report_build_run_summary_section(ga, styles, modules):
    Paragraph = modules["Paragraph"]
    Spacer = modules["Spacer"]
    Table = modules["Table"]
    TableStyle = modules["TableStyle"]
    colors = modules["colors"]
    inch = modules["inch"]

    rows = [[Paragraph("<b>Item</b>", styles["BodyText"]),
             Paragraph("<b>Value</b>", styles["BodyText"])]]
    rows.append([Paragraph("Problem type", styles["BodyText"]),
                 Paragraph("Multi-objective" if _pdf_report_is_multi_objective(ga)
                           else "Single-objective", styles["BodyText"])])
    rows.append([Paragraph("Number of objectives", styles["BodyText"]),
                 Paragraph(str(_pdf_report_num_objectives(ga)), styles["BodyText"])])
    rows.append([Paragraph("Generations completed", styles["BodyText"]),
                 Paragraph(str(ga.generations_completed), styles["BodyText"])])
    rows.append([Paragraph("Final population size", styles["BodyText"]),
                 Paragraph(str(ga.sol_per_pop), styles["BodyText"])])
    rows.append([Paragraph("Best solution generation", styles["BodyText"]),
                 Paragraph(str(ga.best_solution_generation),
                           styles["BodyText"])])
    if not _pdf_report_is_multi_objective(ga):
        rows.append([Paragraph("Best fitness", styles["BodyText"]),
                     Paragraph(_pdf_report_format_value(ga.best_solutions_fitness[-1]
                                             if ga.best_solutions_fitness else "n/a"),
                               styles["BodyText"])])

    table = Table(rows, colWidths=[2.2 * inch, 4.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return [
        Paragraph("Run summary", styles["Heading1"]),
        table,
        Spacer(1, 0.2 * inch),
    ]


def _pdf_report_build_best_solution_section(ga, styles, modules):
    Paragraph = modules["Paragraph"]
    Spacer = modules["Spacer"]
    inch = modules["inch"]
    try:
        best_solution, best_fitness, best_idx = ga.best_solution()
    except Exception as exc:
        return [
            Paragraph("Best solution", styles["Heading1"]),
            Paragraph(f"Could not compute best_solution(): {exc}",
                      styles["BodyText"]),
            Spacer(1, 0.2 * inch),
        ]
    elements = [Paragraph("Best solution", styles["Heading1"])]
    elements.append(Paragraph(f"Population index: {best_idx}",
                              styles["BodyText"]))
    elements.append(Paragraph(f"Fitness: {_pdf_report_format_value(best_fitness)}",
                              styles["BodyText"]))
    elements.append(Paragraph(f"Solution: {_pdf_report_format_value(list(best_solution))}",
                              styles["BodyText"]))
    elements.append(Spacer(1, 0.2 * inch))
    return elements


def _pdf_report_build_plots_section(ga,
                         include_plots,
                         figure_size_inches,
                         styles,
                         modules,
                         matplt):
    Paragraph = modules["Paragraph"]
    Spacer = modules["Spacer"]
    Image = modules["Image"]
    PageBreak = modules["PageBreak"]
    inch = modules["inch"]

    plot_method_names = _pdf_report_select_plot_methods(ga, include_plots)
    elements = [Paragraph("Plots", styles["Heading1"])]
    if not plot_method_names:
        elements.append(Paragraph(
            "No plots were applicable for this run. Set save_solutions=True "
            "for over-generation plots and use a multi-objective fitness "
            "function for Pareto-related plots.",
            styles["BodyText"]))
        return elements

    for entry in REPORT_PLOTS:
        method_name = entry["method"]
        if method_name not in plot_method_names:
            continue
        elements.append(Paragraph(entry["name"], styles["Heading2"]))
        figure_data = _pdf_report_render_plot_to_png(ga, entry, figure_size_inches, matplt)
        if figure_data is None:
            elements.append(Paragraph(
                f"Plot {method_name} could not be drawn for this run.",
                styles["BodyText"]))
            continue
        image_width_inches, image_height_inches = figure_size_inches
        image = Image(io.BytesIO(figure_data),
                      width=image_width_inches * inch,
                      height=image_height_inches * inch)
        elements.append(image)
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(PageBreak())
    return elements


def _pdf_report_build_notes_section(notes, styles, modules):
    if not notes:
        return []
    Paragraph = modules["Paragraph"]
    Spacer = modules["Spacer"]
    inch = modules["inch"]
    return [
        Paragraph("Notes", styles["Heading1"]),
        Paragraph(str(notes), styles["BodyText"]),
        Spacer(1, 0.2 * inch),
    ]


def _pdf_report_select_plot_methods(ga, include_plots):
    """
    Return the set of plot method names that the report will include.
    When the caller passes ``None`` or ``"all"``, auto-pick every plot
    whose preconditions are satisfied by the current GA state.
    """
    if include_plots is None or include_plots == "all":
        requested = None
    else:
        requested = list(include_plots)
        valid_method_names = {entry["method"] for entry in REPORT_PLOTS}
        unknown = set(requested) - valid_method_names
        if unknown:
            raise ValueError(
                f"Unknown plot method(s) in include_plots: {sorted(unknown)}. "
                f"Allowed values: {sorted(valid_method_names)}.")

    is_moo = _pdf_report_is_multi_objective(ga)
    num_objectives = _pdf_report_num_objectives(ga)
    selected = []
    for entry in REPORT_PLOTS:
        if requested is not None and entry["method"] not in requested:
            continue
        if entry["requires_moo"] and not is_moo:
            continue
        if not all(getattr(ga, name, False) for name in entry["requires"]):
            continue
        max_objectives = entry.get("max_objectives")
        if max_objectives is not None and num_objectives > max_objectives:
            continue
        selected.append(entry["method"])
    return selected


def _pdf_report_render_plot_to_png(ga, plot_entry, figure_size_inches, matplt):
    """
    Call the requested plot method on the GA, capture the matplotlib
    figure, save it as PNG bytes, and close it so the figure stack does
    not grow unbounded. Returns ``None`` when the plot method raises.
    """
    method = getattr(ga, plot_entry["method"])
    kwargs = dict(plot_entry["kwargs"])
    try:
        figure = method(**kwargs)
    except Exception:
        return None
    if figure is None:
        return None
    figure.set_size_inches(*figure_size_inches)
    buffer = io.BytesIO()
    try:
        figure.tight_layout()
    except Exception:
        # tight_layout fails on some figure layouts (e.g. those with a
        # nested gridspec); ignore and keep the original layout.
        pass
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    matplt.close(figure)
    return buffer.getvalue()


def _pdf_report_format_value(value):
    """Compact, human-readable rendering for the configuration table."""
    if callable(value) and hasattr(value, "__name__"):
        return f"<callable {value.__name__}>"
    if isinstance(value, (list, tuple)) and len(value) > 8:
        head = ", ".join(_pdf_report_format_value(v) for v in value[:6])
        return f"[{head}, ... (+{len(value) - 6} more)]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
