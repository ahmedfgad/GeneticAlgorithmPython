"""
The pygad.visualize.plot module has methods to create plots.
"""

import numpy
# import matplotlib.pyplot
import pygad

def get_matplotlib():
    """
    Lazy-import ``matplotlib.pyplot``. Importing it at module scope
    would force every PyGAD user to pay the matplotlib import cost
    even when no plot is ever drawn. The plot methods call this
    helper instead so the import happens on first use.

    Returns
    -------
    matplt : module
        The imported ``matplotlib.pyplot`` module.
    """
    import matplotlib.pyplot as matplt
    return matplt

class Plot:

    def __init__():
        pass

    def plot_fitness(self, 
                     title="PyGAD - Generation vs. Fitness", 
                     xlabel="Generation", 
                     ylabel="Fitness", 
                     linewidth=3, 
                     font_size=14, 
                     plot_type="plot",
                     color="#64f20c",
                     label=None,
                     save_dir=None):

        """
        Draw, show, and return a figure that traces the best fitness
        across generations. For multi-objective problems, one curve
        per objective is drawn on the same axes.

        Must be called after at least one generation has completed;
        otherwise it raises ``RuntimeError``.

        Parameters
        ----------
        title : str
            Figure title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        linewidth : numeric or iterable
            Line width. Pass an iterable in multi-objective mode to
            give each objective its own width.
        font_size : numeric
            Font size used for the title and axis labels.
        plot_type : str
            One of ``"plot"``, ``"scatter"``, or ``"bar"``.
        color : str or iterable
            Curve color. Pass an iterable in multi-objective mode to
            color each objective independently.
        label : iterable or None
            Per-objective legend label for multi-objective problems.
            Ignored for single-objective problems.
        save_dir : str or None
            If set, the figure is saved to this path before being
            shown.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure that was created.

        Raises
        ------
        RuntimeError
            If no generation has completed yet.
        """

        if self.generations_completed < 1:
            self.logger.error("The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")
            raise RuntimeError("The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")

        matplt = get_matplotlib()

        fig = matplt.figure()
        if type(self.best_solutions_fitness[0]) in [list, tuple, numpy.ndarray] and len(self.best_solutions_fitness[0]) > 1:
            # Multi-objective optimization problem.
            if type(linewidth) in pygad.GA.supported_int_float_types:
                linewidth = [linewidth]
                linewidth.extend([linewidth[0]]*len(self.best_solutions_fitness[0]))
            elif type(linewidth) in [list, tuple, numpy.ndarray]:
                pass

            if type(color) is str:
                color = [color]
                color.extend([None]*len(self.best_solutions_fitness[0]))
            elif type(color) in [list, tuple, numpy.ndarray]:
                pass
            
            if label is None:
                label = [None]*len(self.best_solutions_fitness[0])

            # Loop through each objective to plot its fitness.
            for objective_idx in range(len(self.best_solutions_fitness[0])):
                # Return the color, line width, and label of the current plot.
                current_color = color[objective_idx]
                current_linewidth = linewidth[objective_idx]
                current_label = label[objective_idx]
                # Return the fitness values for the current objective function across all generations.
                fitness = numpy.array(self.best_solutions_fitness)[:, objective_idx]
                if plot_type == "plot":
                    matplt.plot(fitness, 
                                           linewidth=current_linewidth, 
                                           color=current_color,
                                           label=current_label)
                elif plot_type == "scatter":
                    matplt.scatter(range(len(fitness)), 
                                              fitness, 
                                              linewidth=current_linewidth, 
                                              color=current_color,
                                              label=current_label)
                elif plot_type == "bar":
                    matplt.bar(range(len(fitness)), 
                                          fitness, 
                                          linewidth=current_linewidth, 
                                          color=current_color,
                                          label=current_label)
        else:
            # Single-objective optimization problem.
            if plot_type == "plot":
                matplt.plot(self.best_solutions_fitness, 
                                       linewidth=linewidth, 
                                       color=color)
            elif plot_type == "scatter":
                matplt.scatter(range(len(self.best_solutions_fitness)), 
                                          self.best_solutions_fitness, 
                                          linewidth=linewidth, 
                                          color=color)
            elif plot_type == "bar":
                matplt.bar(range(len(self.best_solutions_fitness)), 
                                      self.best_solutions_fitness, 
                                      linewidth=linewidth, 
                                      color=color)
        matplt.title(title, fontsize=font_size)
        matplt.xlabel(xlabel, fontsize=font_size)
        matplt.ylabel(ylabel, fontsize=font_size)
        # Create a legend out of the labels.
        # Check if there is at least 1 labeled artist.
        # If not, the matplt.legend() method will raise a warning.
        if not (matplt.gca().get_legend_handles_labels()[0] == []):
            matplt.legend()

        if not save_dir is None:
            matplt.savefig(fname=save_dir, 
                                      bbox_inches='tight')
        matplt.show()

        return fig

    def plot_new_solution_rate(self,
                               title="PyGAD - Generation vs. New Solution Rate", 
                               xlabel="Generation", 
                               ylabel="New Solution Rate", 
                               linewidth=3, 
                               font_size=14, 
                               plot_type="plot",
                               color="#64f20c",
                               save_dir=None):

        """
        Draw, show, and return a figure that plots how many new
        (previously unseen) solutions appear in each generation. A
        flat curve means the population is repeating itself; a high
        curve means it is still exploring.

        Requires ``save_solutions=True`` in the GA constructor and at
        least one completed generation.

        Parameters
        ----------
        title : str
            Figure title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        linewidth : numeric
            Line width of the curve.
        font_size : numeric
            Font size for title and axis labels.
        plot_type : str
            One of ``"plot"``, ``"scatter"``, or ``"bar"``.
        color : str
            Curve color.
        save_dir : str or None
            If set, the figure is saved to this path before being
            shown.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure that was created.

        Raises
        ------
        RuntimeError
            If no generation has completed yet, or if
            ``save_solutions`` is False.
        """

        if self.generations_completed < 1:
            self.logger.error("The plot_new_solution_rate() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")
            raise RuntimeError("The plot_new_solution_rate() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")

        if self.save_solutions == False:
            self.logger.error("The plot_new_solution_rate() method works only when save_solutions=True in the constructor of the pygad.GA class.")
            raise RuntimeError("The plot_new_solution_rate() method works only when save_solutions=True in the constructor of the pygad.GA class.")

        unique_solutions = set()
        num_unique_solutions_per_generation = []
        for generation_idx in range(self.generations_completed):
            
            len_before = len(unique_solutions)

            start = generation_idx * self.sol_per_pop
            end = start + self.sol_per_pop
        
            for sol in self.solutions[start:end]:
                unique_solutions.add(tuple(sol))
        
            len_after = len(unique_solutions)
        
            generation_num_unique_solutions = len_after - len_before
            num_unique_solutions_per_generation.append(generation_num_unique_solutions)

        matplt = get_matplotlib()

        fig = matplt.figure()
        if plot_type == "plot":
            matplt.plot(num_unique_solutions_per_generation, linewidth=linewidth, color=color)
        elif plot_type == "scatter":
            matplt.scatter(range(self.generations_completed), num_unique_solutions_per_generation, linewidth=linewidth, color=color)
        elif plot_type == "bar":
            matplt.bar(range(self.generations_completed), num_unique_solutions_per_generation, linewidth=linewidth, color=color)
        matplt.title(title, fontsize=font_size)
        matplt.xlabel(xlabel, fontsize=font_size)
        matplt.ylabel(ylabel, fontsize=font_size)

        if not save_dir is None:
            matplt.savefig(fname=save_dir, 
                                      bbox_inches='tight')
        matplt.show()

        return fig

    def plot_genes(self, 
                   title="PyGAD - Gene", 
                   xlabel="Gene", 
                   ylabel="Value", 
                   linewidth=3, 
                   font_size=14,
                   plot_type="plot",
                   graph_type="plot",
                   fill_color="#64f20c",
                   color="black",
                   solutions="all",
                   save_dir=None):

        """
        Draw, show, and return a figure with one subplot per gene,
        showing how that gene's value drifts across generations. The
        plot can be drawn as a line ("plot"), as a boxplot per gene,
        or as a histogram of values per gene.

        Requires ``save_solutions=True`` (when ``solutions="all"``)
        or ``save_best_solutions=True`` (when ``solutions="best"``)
        in the GA constructor and at least one completed generation.

        Parameters
        ----------
        title : str
            Figure title.
        xlabel : str
            X-axis label (used by the boxplot view).
        ylabel : str
            Y-axis label (used by the boxplot view).
        linewidth : numeric
            Line width.
        font_size : numeric
            Font size for the title and labels.
        plot_type : str
            One of ``"plot"``, ``"scatter"``, or ``"bar"``. Used when
            ``graph_type="plot"``.
        graph_type : str
            One of ``"plot"``, ``"boxplot"``, or ``"histogram"``.
        fill_color : str
            Fill color for the graph (curves, bars or boxes).
        color : str
            Outline / accent color, mainly used by the boxplot view
            for the whiskers, caps and medians.
        solutions : str
            ``"all"`` to plot every saved solution; ``"best"`` to plot
            only the best solution of each generation.
        save_dir : str or None
            If set, the figure is saved to this path before being
            shown.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure that was created.

        Raises
        ------
        RuntimeError
            If no generation has completed yet, if the required
            ``save_solutions`` / ``save_best_solutions`` flag is
            False, or if ``solutions`` is anything other than
            ``"all"`` / ``"best"``.
        """

        if self.generations_completed < 1:
            self.logger.error("The plot_genes() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")
            raise RuntimeError("The plot_genes() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")

        matplt = get_matplotlib()

        if type(solutions) is str:
            if solutions == 'all':
                if self.save_solutions:
                    solutions_to_plot = numpy.array(self.solutions)
                else:
                    self.logger.error("The plot_genes() method with solutions='all' can only be called if 'save_solutions=True' in the pygad.GA class constructor.")
                    raise RuntimeError("The plot_genes() method with solutions='all' can only be called if 'save_solutions=True' in the pygad.GA class constructor.")
            elif solutions == 'best':
                if self.save_best_solutions:
                    solutions_to_plot = self.best_solutions
                else:
                    self.logger.error("The plot_genes() method with solutions='best' can only be called if 'save_best_solutions=True' in the pygad.GA class constructor.")
                    raise RuntimeError("The plot_genes() method with solutions='best' can only be called if 'save_best_solutions=True' in the pygad.GA class constructor.")
            else:
                self.logger.error("The solutions parameter can be either 'all' or 'best' but {solutions} found.")
                raise RuntimeError("The solutions parameter can be either 'all' or 'best' but {solutions} found.")
        else:
            self.logger.error("The solutions parameter must be a string but {solutions_type} found.".format(solutions_type=type(solutions)))
            raise RuntimeError("The solutions parameter must be a string but {solutions_type} found.".format(solutions_type=type(solutions)))

        if graph_type == "plot":
            # num_rows will be always be >= 1
            # num_cols can only be 0 if num_genes=1
            num_rows = int(numpy.ceil(self.num_genes/5.0))
            num_cols = int(numpy.ceil(self.num_genes/num_rows))
    
            if num_cols == 0:
                figsize = (10, 8)
                # There is only a single gene
                fig, ax = matplt.subplots(num_rows, figsize=figsize)
                if plot_type == "plot":
                    ax.plot(solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                elif plot_type == "scatter":
                    ax.scatter(range(self.generations_completed + 1), solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                elif plot_type == "bar":
                    ax.bar(range(self.generations_completed + 1), solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                ax.set_xlabel(0, fontsize=font_size)
            else:
                fig, axs = matplt.subplots(num_rows, num_cols)
    
                if num_cols == 1 and num_rows == 1:
                    fig.set_figwidth(5 * num_cols)
                    fig.set_figheight(4)
                    axs.plot(solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                    axs.set_xlabel("Gene " + str(0), fontsize=font_size)
                elif num_cols == 1 or num_rows == 1:
                    fig.set_figwidth(5 * num_cols)
                    fig.set_figheight(4)
                    for gene_idx in range(len(axs)):
                        if plot_type == "plot":
                            axs[gene_idx].plot(solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                        elif plot_type == "scatter":
                            axs[gene_idx].scatter(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                        elif plot_type == "bar":
                            axs[gene_idx].bar(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                        axs[gene_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                else:
                    gene_idx = 0
                    fig.set_figwidth(25)
                    fig.set_figheight(4*num_rows)
                    for row_idx in range(num_rows):
                        for col_idx in range(num_cols):
                            if gene_idx >= self.num_genes:
                                # axs[row_idx, col_idx].remove()
                                break
                            if plot_type == "plot":
                                axs[row_idx, col_idx].plot(solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                            elif plot_type == "scatter":
                                axs[row_idx, col_idx].scatter(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                            elif plot_type == "bar":
                                axs[row_idx, col_idx].bar(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                            axs[row_idx, col_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                            gene_idx += 1
    
            fig.suptitle(title, fontsize=font_size, y=1.001)
            matplt.tight_layout()

        elif graph_type == "boxplot":
            # Width scales with the number of genes so the boxes do not
            # crowd, but never shrinks below a readable default.
            fig = matplt.figure(figsize=(max(8, 0.7 * self.num_genes), 5))

            # Create an axes instance
            ax = fig.add_subplot(111)
            # Matplotlib 3.9 renamed the boxplot `labels=` kwarg to
            # `tick_labels=` and will drop the old name in 3.11. Use whichever
            # name this matplotlib supports so we work on either side of the
            # rename without forcing users to upgrade matplotlib.
            import inspect
            _tick_kw = (
                "tick_labels"
                if "tick_labels" in inspect.signature(ax.boxplot).parameters
                else "labels"
            )
            boxplots = ax.boxplot(solutions_to_plot,
                                   patch_artist=True,
                                   **{_tick_kw: range(self.num_genes)})
            # adding horizontal grid lines
            ax.yaxis.grid(True)
    
            for box in boxplots['boxes']:
                # change outline color
                box.set(color='black', linewidth=linewidth)
                # change fill color https://color.adobe.com/create/color-wheel
                box.set_facecolor(fill_color)

            for whisker in boxplots['whiskers']:
                whisker.set(color=color, linewidth=linewidth)
            for median in boxplots['medians']:
                median.set(color=color, linewidth=linewidth)
            for cap in boxplots['caps']:
                cap.set(color=color, linewidth=linewidth)
    
            matplt.title(title, fontsize=font_size)
            matplt.xlabel(xlabel, fontsize=font_size)
            matplt.ylabel(ylabel, fontsize=font_size)
            matplt.tight_layout()

        elif graph_type == "histogram":
            # num_rows will always be >= 1
            # num_cols can only be 0 if num_genes=1
            num_rows = int(numpy.ceil(self.num_genes/5.0))
            num_cols = int(numpy.ceil(self.num_genes/num_rows))
    
            if num_cols == 0:
                figsize = (10, 8)
                # There is only a single gene
                fig, ax = matplt.subplots(num_rows, 
                                                     figsize=figsize)
                ax.hist(solutions_to_plot[:, 0], color=fill_color)
                ax.set_xlabel(0, fontsize=font_size)
            else:
                fig, axs = matplt.subplots(num_rows, num_cols)
    
                if num_cols == 1 and num_rows == 1:
                    fig.set_figwidth(4 * num_cols)
                    fig.set_figheight(3)
                    axs.hist(solutions_to_plot[:, 0], 
                             color=fill_color,
                             rwidth=0.95)
                    axs.set_xlabel("Gene " + str(0), fontsize=font_size)
                elif num_cols == 1 or num_rows == 1:
                    fig.set_figwidth(4 * num_cols)
                    fig.set_figheight(3)
                    for gene_idx in range(len(axs)):
                        axs[gene_idx].hist(solutions_to_plot[:, gene_idx], 
                                           color=fill_color,
                                           rwidth=0.95)
                        axs[gene_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                else:
                    gene_idx = 0
                    fig.set_figwidth(20)
                    fig.set_figheight(3*num_rows)
                    for row_idx in range(num_rows):
                        for col_idx in range(num_cols):
                            if gene_idx >= self.num_genes:
                                # axs[row_idx, col_idx].remove()
                                break
                            axs[row_idx, col_idx].hist(solutions_to_plot[:, gene_idx], 
                                                       color=fill_color,
                                                       rwidth=0.95)
                            axs[row_idx, col_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                            gene_idx += 1
    
            fig.suptitle(title, fontsize=font_size, y=1.001)
            matplt.tight_layout()

        if not save_dir is None:
            matplt.savefig(fname=save_dir, 
                                      bbox_inches='tight')

        matplt.show()

        return fig

    def plot_pareto_front_curve(self,
                                title="Pareto Front Curve",
                                xlabel="Objective 1",
                                ylabel="Objective 2",
                                zlabel="Objective 3",
                                linewidth=3,
                                font_size=14,
                                label="Pareto Front",
                                color="#FF6347",
                                color_fitness="#4169E1",
                                grid=True,
                                alpha=0.7,
                                marker="o",
                                save_dir=None):
        """
        Show the Pareto front of the current population.

        For 2 objectives: scatter of the population plus a curve
        through the non-dominated points. For 3 objectives: 3D
        scatter with the non-dominated points highlighted.

        For 4 or more objectives, 2D / 3D scatter no longer reads
        well. Use ``plot_pareto_front_pcp``, ``plot_pareto_front_scatter_matrix``,
        or ``plot_pareto_front_heatmap`` instead.

        Parameters
        ----------
        title : str
            Figure title.
        xlabel, ylabel, zlabel : str
            Axis labels. ``zlabel`` is only used for 3 objectives.
        linewidth : numeric
            Line width (2D mode).
        font_size : numeric
            Font size for the title and axis labels.
        label : str
            Legend label for the Pareto front.
        color : str
            Colour of the Pareto curve (2D) or non-dominated markers (3D).
        color_fitness : str
            Colour of the population scatter points.
        grid : bool
            Draw grid lines.
        alpha : float
            Transparency of the Pareto curve / population markers.
        marker : str
            Marker style for the scatter points.
        save_dir : str or None
            If set, saves the figure to this path before showing it.

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed, the problem is
            single-objective, or M > 3.
        """

        if self.generations_completed < 1:
            self.logger.error(f"The plot_pareto_front_curve() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")
            raise RuntimeError(f"The plot_pareto_front_curve() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")

        num_objectives = self._num_objectives_or_raise("plot_pareto_front_curve()")

        if num_objectives not in (2, 3):
            self.logger.error(f"The plot_pareto_front_curve() method supports 2 or 3 objectives but there are {num_objectives}. For higher dimensions use plot_pareto_front_pcp(), plot_pareto_front_scatter_matrix(), or plot_pareto_front_heatmap().")
            raise RuntimeError(f"The plot_pareto_front_curve() method supports 2 or 3 objectives but there are {num_objectives}. For higher dimensions use plot_pareto_front_pcp(), plot_pareto_front_scatter_matrix(), or plot_pareto_front_heatmap().")

        last_fitness = numpy.asarray(self.last_generation_fitness)
        non_dominated_fitness = self._last_generation_pareto_front()

        matplt = get_matplotlib()

        if num_objectives == 2:
            fig = matplt.figure()
            matplt.scatter(last_fitness[:, 0],
                           last_fitness[:, 1],
                           marker=marker,
                           color=color_fitness,
                           label='Fitness',
                           alpha=1.0)
            order = numpy.argsort(non_dominated_fitness[:, 0])
            matplt.plot(non_dominated_fitness[order, 0],
                        non_dominated_fitness[order, 1],
                        marker=marker,
                        label=label,
                        alpha=alpha,
                        color=color,
                        linewidth=linewidth)
            matplt.title(title, fontsize=font_size)
            matplt.xlabel(xlabel, fontsize=font_size)
            matplt.ylabel(ylabel, fontsize=font_size)
            matplt.legend()
            matplt.grid(grid)
        else:
            fig = matplt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(last_fitness[:, 0],
                       last_fitness[:, 1],
                       last_fitness[:, 2],
                       marker=marker,
                       color=color_fitness,
                       label='Fitness',
                       alpha=0.6)
            ax.scatter(non_dominated_fitness[:, 0],
                       non_dominated_fitness[:, 1],
                       non_dominated_fitness[:, 2],
                       marker=marker,
                       color=color,
                       label=label,
                       alpha=alpha,
                       s=60)
            ax.set_title(title, fontsize=font_size)
            ax.set_xlabel(xlabel, fontsize=font_size)
            ax.set_ylabel(ylabel, fontsize=font_size)
            ax.set_zlabel(zlabel, fontsize=font_size)
            ax.legend()
            if grid:
                ax.grid(True)

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')

        matplt.show()

        return fig

    # ── Helpers shared by the Pareto-front plots ─────────────────────────────

    def _num_objectives_or_raise(self, method_name):
        """
        Returns M for a MOO problem and raises for SOO. Used as the
        first guard inside every MOO-only plot method.
        """
        first_best = self.best_solutions_fitness[0]
        if type(first_best) in [list, tuple, numpy.ndarray] and len(first_best) > 1:
            return len(first_best)
        self.logger.error(f"The {method_name} method only works with multi-objective optimization problems.")
        raise RuntimeError(f"The {method_name} method only works with multi-objective optimization problems.")

    def _last_generation_pareto_front(self):
        """
        Returns the non-dominated rows of last_generation_fitness as a
        2D numpy array. Order matches the order returned by
        get_non_dominated_set, which is the order the solutions appear
        in the population.
        """
        last_fitness = numpy.asarray(self.last_generation_fitness)
        remaining_set = list(zip(range(last_fitness.shape[0]), last_fitness))
        _, non_dominated_set = self.get_non_dominated_set(remaining_set)
        indices = [item[0] for item in non_dominated_set]
        return last_fitness[indices]

    def _require_save_solutions(self, method_name):
        """Raise unless the GA was constructed with save_solutions=True."""
        if not self.save_solutions:
            self.logger.error(f"The {method_name} method requires save_solutions=True in the pygad.GA constructor.")
            raise RuntimeError(f"The {method_name} method requires save_solutions=True in the pygad.GA constructor.")

    def _per_generation_fitness(self):
        """
        Return a list of length (generations_completed + 1) where
        each entry is the fitness array of one generation. Only valid
        when save_solutions=True.
        """
        per_gen = []
        fitness_flat = numpy.asarray(self.solutions_fitness)
        sol_per_pop = self.sol_per_pop
        num_blocks = fitness_flat.shape[0] // sol_per_pop
        for g in range(num_blocks):
            per_gen.append(fitness_flat[g * sol_per_pop:(g + 1) * sol_per_pop])
        return per_gen

    def _per_generation_solutions(self):
        """
        Return a list of length (generations_completed + 1) where
        each entry is the population array of one generation. Only
        valid when save_solutions=True.
        """
        per_gen = []
        solutions_flat = numpy.asarray(self.solutions, dtype=float)
        sol_per_pop = self.sol_per_pop
        num_blocks = solutions_flat.shape[0] // sol_per_pop
        for g in range(num_blocks):
            per_gen.append(solutions_flat[g * sol_per_pop:(g + 1) * sol_per_pop])
        return per_gen

    # ── Pareto-front views for M >= 3 ────────────────────────────────────────

    def plot_pareto_front_pcp(self,
                              title="Pareto Front - Parallel Coordinates",
                              xlabel="Objective",
                              ylabel="Normalised value",
                              linewidth=1.5,
                              font_size=14,
                              color="#4169E1",
                              alpha=0.6,
                              grid=True,
                              save_dir=None):
        """
        Parallel-coordinates plot of the final non-dominated set.

        Every objective gets a vertical axis. Each non-dominated
        solution becomes a polyline that crosses all axes. Values are
        normalized per objective so axes with very different ranges
        stay comparable.

        Works for any M >= 2.

        Parameters
        ----------
        title, xlabel, ylabel : str
        linewidth : numeric
        font_size : numeric
        color : str
            Polyline color.
        alpha : float
        grid : bool
        save_dir : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed or the problem is
            single-objective.
        """
        if self.generations_completed < 1:
            self.logger.error("The plot_pareto_front_pcp() method requires at least one completed generation.")
            raise RuntimeError("The plot_pareto_front_pcp() method requires at least one completed generation.")
        num_objectives = self._num_objectives_or_raise("plot_pareto_front_pcp()")

        front = self._last_generation_pareto_front()
        min_per_obj = front.min(axis=0)
        max_per_obj = front.max(axis=0)
        spread = numpy.where(max_per_obj > min_per_obj, max_per_obj - min_per_obj, 1.0)
        normalized = (front - min_per_obj) / spread

        matplt = get_matplotlib()
        fig, ax = matplt.subplots()
        x_axis = numpy.arange(num_objectives)
        for row in normalized:
            ax.plot(x_axis, row, color=color, alpha=alpha, linewidth=linewidth)
        ax.set_xticks(x_axis)
        ax.set_xticklabels([f"f{i + 1}" for i in range(num_objectives)], fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.grid(grid)

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')
        matplt.show()
        return fig

    def plot_pareto_front_scatter_matrix(self,
                                         title="Pareto Front - Scatter Matrix",
                                         font_size=14,
                                         color="#4169E1",
                                         marker="o",
                                         alpha=0.6,
                                         grid=True,
                                         save_dir=None):
        """
        M-by-M grid of pairwise scatter plots for the final
        non-dominated set. The diagonal shows a histogram of each
        objective's values. Helpful when M >= 4 and a single 3D
        scatter no longer reads well.

        Parameters
        ----------
        title : str
        font_size : numeric
        color : str
        marker : str
        alpha : float
        grid : bool
        save_dir : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed or the problem is
            single-objective.
        """
        if self.generations_completed < 1:
            self.logger.error("The plot_pareto_front_scatter_matrix() method requires at least one completed generation.")
            raise RuntimeError("The plot_pareto_front_scatter_matrix() method requires at least one completed generation.")
        num_objectives = self._num_objectives_or_raise("plot_pareto_front_scatter_matrix()")
        front = self._last_generation_pareto_front()

        matplt = get_matplotlib()
        fig, axes = matplt.subplots(num_objectives, num_objectives,
                                    figsize=(3 * num_objectives, 3 * num_objectives))
        if num_objectives == 1:
            axes = numpy.array([[axes]])
        for i in range(num_objectives):
            for j in range(num_objectives):
                ax = axes[i, j]
                if i == j:
                    ax.hist(front[:, i], color=color, alpha=alpha)
                else:
                    ax.scatter(front[:, j], front[:, i],
                               color=color, marker=marker, alpha=alpha)
                if i == num_objectives - 1:
                    ax.set_xlabel(f"f{j + 1}", fontsize=font_size)
                if j == 0:
                    ax.set_ylabel(f"f{i + 1}", fontsize=font_size)
                ax.grid(grid)
        fig.suptitle(title, fontsize=font_size)
        fig.tight_layout()

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')
        matplt.show()
        return fig

    def plot_pareto_front_heatmap(self,
                                  title="Pareto Front - Heatmap",
                                  xlabel="Objective",
                                  ylabel="Solution",
                                  font_size=14,
                                  cmap="viridis",
                                  sort_by=0,
                                  save_dir=None):
        """
        Heatmap of the final non-dominated set. Rows are solutions,
        columns are objectives, color is the (raw) objective value.

        Parameters
        ----------
        title, xlabel, ylabel : str
        font_size : numeric
        cmap : str
            Matplotlib colormap name.
        sort_by : int or None
            Objective index to sort rows by (ascending). Pass ``None``
            to keep the original order.
        save_dir : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed or the problem is
            single-objective.
        ValueError
            If ``sort_by`` is out of range.
        """
        if self.generations_completed < 1:
            self.logger.error("The plot_pareto_front_heatmap() method requires at least one completed generation.")
            raise RuntimeError("The plot_pareto_front_heatmap() method requires at least one completed generation.")
        num_objectives = self._num_objectives_or_raise("plot_pareto_front_heatmap()")

        front = self._last_generation_pareto_front()
        if sort_by is not None:
            if not (0 <= sort_by < num_objectives):
                raise ValueError(
                    f"sort_by must be an integer in [0, {num_objectives - 1}], "
                    f"but got {sort_by}.")
            order = numpy.argsort(front[:, sort_by])
            front = front[order]

        matplt = get_matplotlib()
        fig, ax = matplt.subplots()
        image = ax.imshow(front, aspect='auto', cmap=cmap)
        ax.set_xticks(numpy.arange(num_objectives))
        ax.set_xticklabels([f"f{i + 1}" for i in range(num_objectives)])
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        fig.colorbar(image, ax=ax)

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')
        matplt.show()
        return fig

    # ── Per-generation diagnostics (need save_solutions=True) ────────────────

    def plot_fitness_band(self,
                          title="PyGAD - Population fitness band",
                          xlabel="Generation",
                          ylabel="Fitness",
                          font_size=14,
                          color="#4169E1",
                          band_alpha=0.2,
                          linewidth=2,
                          objective_index=0,
                          grid=True,
                          save_dir=None):
        """
        Per-generation min / mean / max fitness with a shaded band
        between min and max. For MOO problems, picks one objective
        via ``objective_index`` (default 0).

        Requires ``save_solutions=True``.

        Parameters
        ----------
        title, xlabel, ylabel : str
        font_size : numeric
        color : str
        band_alpha : float
            Transparency of the shaded min-max band.
        linewidth : numeric
        objective_index : int
            Which objective to plot for MOO problems. Ignored for SOO.
        grid : bool
        save_dir : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed or save_solutions is False.
        ValueError
            If ``objective_index`` is out of range for the problem.
        """
        if self.generations_completed < 1:
            self.logger.error("The plot_fitness_band() method requires at least one completed generation.")
            raise RuntimeError("The plot_fitness_band() method requires at least one completed generation.")
        self._require_save_solutions("plot_fitness_band()")

        per_gen = self._per_generation_fitness()
        first = numpy.asarray(per_gen[0])
        is_moo = first.ndim == 2 and first.shape[1] > 1
        if is_moo:
            num_objectives = first.shape[1]
            if not (0 <= objective_index < num_objectives):
                raise ValueError(
                    f"objective_index must be in [0, {num_objectives - 1}], "
                    f"but got {objective_index}.")
            min_vals = [gen[:, objective_index].min() for gen in per_gen]
            mean_vals = [gen[:, objective_index].mean() for gen in per_gen]
            max_vals = [gen[:, objective_index].max() for gen in per_gen]
        else:
            min_vals = [gen.min() for gen in per_gen]
            mean_vals = [gen.mean() for gen in per_gen]
            max_vals = [gen.max() for gen in per_gen]

        matplt = get_matplotlib()
        fig, ax = matplt.subplots()
        generations = numpy.arange(len(per_gen))
        ax.fill_between(generations, min_vals, max_vals,
                        color=color, alpha=band_alpha, label='min-max')
        ax.plot(generations, mean_vals,
                color=color, linewidth=linewidth, label='mean')
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.legend()
        ax.grid(grid)

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')
        matplt.show()
        return fig

    def plot_non_dominated_hypervolume(self,
                         reference_point=None,
                         title="PyGAD - Hypervolume per generation",
                         xlabel="Generation",
                         ylabel="Hypervolume",
                         font_size=14,
                         color="#4169E1",
                         linewidth=2,
                         grid=True,
                         save_dir=None):
        """
        Hypervolume of the non-dominated set per generation.

        Requires ``save_solutions=True``. Uses
        ``pygad.utils.quality_indicators.hypervolume``.

        Parameters
        ----------
        reference_point : array-like or None
            Reference point passed to the hypervolume function. Must
            be smaller than every fitness value on every objective.
            If ``None``, uses ``min(per_gen_fitness) - 0.1`` across all
            saved generations, which is usually a safe default.
        title, xlabel, ylabel : str
        font_size : numeric
        color : str
        linewidth : numeric
        grid : bool
        save_dir : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed, the problem is
            single-objective, or save_solutions is False.
        """
        if self.generations_completed < 1:
            self.logger.error("The plot_non_dominated_hypervolume() method requires at least one completed generation.")
            raise RuntimeError("The plot_non_dominated_hypervolume() method requires at least one completed generation.")
        self._num_objectives_or_raise("plot_non_dominated_hypervolume()")
        self._require_save_solutions("plot_non_dominated_hypervolume()")

        from pygad.utils.quality_indicators import hypervolume

        per_gen = self._per_generation_fitness()
        all_fitness = numpy.vstack([numpy.asarray(g) for g in per_gen])
        if reference_point is None:
            reference_point = all_fitness.min(axis=0) - 0.1
        reference_point = numpy.asarray(reference_point, dtype=float)

        hv_values = [hypervolume(numpy.asarray(g), reference_point) for g in per_gen]

        matplt = get_matplotlib()
        fig, ax = matplt.subplots()
        generations = numpy.arange(len(hv_values))
        ax.plot(generations, hv_values, color=color, linewidth=linewidth)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.grid(grid)

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')
        matplt.show()
        return fig

    def plot_population_diversity(self,
                                  title="PyGAD - Population diversity",
                                  xlabel="Generation",
                                  ylabel="Mean pairwise distance",
                                  font_size=14,
                                  color="#4169E1",
                                  linewidth=2,
                                  grid=True,
                                  save_dir=None):
        """
        Mean pairwise Euclidean distance between solutions per
        generation. A drop signals the population is converging or
        collapsing into a few duplicates.

        Requires ``save_solutions=True``.

        Parameters
        ----------
        title, xlabel, ylabel : str
        font_size : numeric
        color : str
        linewidth : numeric
        grid : bool
        save_dir : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed or save_solutions is False.
        """
        if self.generations_completed < 1:
            self.logger.error("The plot_population_diversity() method requires at least one completed generation.")
            raise RuntimeError("The plot_population_diversity() method requires at least one completed generation.")
        self._require_save_solutions("plot_population_diversity()")

        per_gen = self._per_generation_solutions()
        diversity = []
        for population in per_gen:
            diff = population[:, None, :] - population[None, :, :]
            distances = numpy.sqrt((diff * diff).sum(axis=2))
            # Mean over the upper triangle (each pair counted once).
            n = distances.shape[0]
            if n < 2:
                diversity.append(0.0)
                continue
            upper = distances[numpy.triu_indices(n, k=1)]
            diversity.append(float(upper.mean()))

        matplt = get_matplotlib()
        fig, ax = matplt.subplots()
        generations = numpy.arange(len(diversity))
        ax.plot(generations, diversity, color=color, linewidth=linewidth)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.grid(grid)

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')
        matplt.show()
        return fig

    def plot_pareto_front_evolution(self,
                                    every_k=1,
                                    title="Pareto Front Evolution",
                                    xlabel="Objective 1",
                                    ylabel="Objective 2",
                                    zlabel="Objective 3",
                                    font_size=14,
                                    cmap="viridis",
                                    marker="o",
                                    alpha=0.7,
                                    grid=True,
                                    save_dir=None):
        """
        Overlay the non-dominated set every ``every_k`` generations.
        Colour goes from early generations to late so you can see the
        front converging.

        Works for 2 or 3 objectives. Requires ``save_solutions=True``.

        Parameters
        ----------
        every_k : int
            Plot every k-th generation. ``every_k=1`` plots all of them.
        title, xlabel, ylabel, zlabel : str
        font_size : numeric
        cmap : str
        marker : str
        alpha : float
        grid : bool
        save_dir : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If no generation has completed, the problem is
            single-objective, save_solutions is False, or M > 3.
        ValueError
            If ``every_k`` is not a positive integer.
        """
        if self.generations_completed < 1:
            self.logger.error("The plot_pareto_front_evolution() method requires at least one completed generation.")
            raise RuntimeError("The plot_pareto_front_evolution() method requires at least one completed generation.")
        num_objectives = self._num_objectives_or_raise("plot_pareto_front_evolution()")
        if num_objectives not in (2, 3):
            self.logger.error(f"The plot_pareto_front_evolution() method supports 2 or 3 objectives but there are {num_objectives}.")
            raise RuntimeError(f"The plot_pareto_front_evolution() method supports 2 or 3 objectives but there are {num_objectives}.")
        self._require_save_solutions("plot_pareto_front_evolution()")
        if not (isinstance(every_k, int) and every_k > 0):
            raise ValueError(f"every_k must be a positive integer, but got {every_k}.")

        per_gen = self._per_generation_fitness()
        # Pick generations to draw. Always include the last one.
        indices = list(range(0, len(per_gen), every_k))
        if indices[-1] != len(per_gen) - 1:
            indices.append(len(per_gen) - 1)

        matplt = get_matplotlib()
        colormap = matplt.get_cmap(cmap)
        fig = matplt.figure()
        if num_objectives == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        for plot_idx, gen_idx in enumerate(indices):
            fraction = plot_idx / max(len(indices) - 1, 1)
            color = colormap(fraction)
            gen_fitness = numpy.asarray(per_gen[gen_idx])
            remaining_set = list(zip(range(gen_fitness.shape[0]), gen_fitness))
            _, non_dominated_set = self.get_non_dominated_set(remaining_set)
            front_indices = [item[0] for item in non_dominated_set]
            front = gen_fitness[front_indices]
            if num_objectives == 2:
                ax.scatter(front[:, 0], front[:, 1],
                           color=color, marker=marker, alpha=alpha,
                           label=f"gen {gen_idx}")
            else:
                ax.scatter(front[:, 0], front[:, 1], front[:, 2],
                           color=color, marker=marker, alpha=alpha,
                           label=f"gen {gen_idx}")

        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        if num_objectives == 3:
            ax.set_zlabel(zlabel, fontsize=font_size)
        if len(indices) <= 8:
            ax.legend()
        if num_objectives == 2:
            ax.grid(grid)
        elif grid:
            ax.grid(True)

        if save_dir is not None:
            matplt.savefig(fname=save_dir, bbox_inches='tight')
        matplt.show()
        return fig
