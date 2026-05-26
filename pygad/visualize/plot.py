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
            Curve colour. Pass an iterable in multi-objective mode to
            colour each objective independently.
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
            Curve colour.
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
            Fill colour for the graph (curves, bars or boxes).
        color : str
            Outline / accent colour, mainly used by the boxplot view
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
            fig = matplt.figure(1, figsize=(0.7*self.num_genes, 6))

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
        Draw, show, and return a 2D Pareto front curve for a
        two-objective problem. The fitness of every solution in the
        current population is plotted as a point, and the points on
        Pareto front 0 are connected to form the curve.

        Only works for multi-objective problems with exactly two
        objectives and at least one completed generation.

        Parameters
        ----------
        title : str
            Figure title.
        xlabel : str
            X-axis label (the first objective).
        ylabel : str
            Y-axis label (the second objective).
        linewidth : numeric
            Line width of the Pareto curve.
        font_size : numeric
            Font size for the title and labels.
        label : str
            Legend label for the Pareto curve.
        color : str
            Colour of the Pareto curve.
        color_fitness : str
            Colour of the per-solution fitness scatter points.
        grid : bool
            Whether to draw the grid lines.
        alpha : float
            Transparency of the Pareto curve.
        marker : str
            Matplotlib marker style for the fitness points.
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
            If no generation has completed yet, the problem is
            single-objective, or the number of objectives is not
            exactly two.
        """

        if self.generations_completed < 1:
            self.logger.error("The plot_pareto_front_curve() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")
            raise RuntimeError("The plot_pareto_front_curve() method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed.")

        if type(self.best_solutions_fitness[0]) in [list, tuple, numpy.ndarray] and len(self.best_solutions_fitness[0]) > 1:
            # Multi-objective optimization problem.
            if len(self.best_solutions_fitness[0]) == 2:
                # Only 2 objectives. Proceed.
                pass
            else:
                # More than 2 objectives.
                self.logger.error(f"The plot_pareto_front_curve() method only supports 2 objectives but there are {self.best_solutions_fitness[0]} objectives.")
                raise RuntimeError(f"The plot_pareto_front_curve() method only supports 2 objectives but there are {self.best_solutions_fitness[0]} objectives.")
        else:
            # Single-objective optimization problem.
            self.logger.error("The plot_pareto_front_curve() method only works with multi-objective optimization problems.")
            raise RuntimeError("The plot_pareto_front_curve() method only works with multi-objective optimization problems.")

        # Plot the pareto front curve.
        remaining_set = list(zip(range(0, self.last_generation_fitness.shape[0]), self.last_generation_fitness))
        # The non-dominated set is the pareto front set.
        dominated_set, non_dominated_set = self.get_non_dominated_set(remaining_set)

        # Extract the fitness values (objective values) of the non-dominated solutions for plotting.
        pareto_front_x = [self.last_generation_fitness[item[0]][0] for item in non_dominated_set]
        pareto_front_y = [self.last_generation_fitness[item[0]][1] for item in non_dominated_set]

        # Sort the Pareto front solutions (optional but can make the plot cleaner)
        sorted_pareto_front = sorted(zip(pareto_front_x, pareto_front_y))

        matplt = get_matplotlib()

        # Plotting
        fig = matplt.figure()
        # First, plot the scatter of all points (population)
        all_points_x = [self.last_generation_fitness[i][0] for i in range(self.sol_per_pop)]
        all_points_y = [self.last_generation_fitness[i][1] for i in range(self.sol_per_pop)]
        matplt.scatter(all_points_x, 
                                  all_points_y, 
                                  marker=marker,
                                  color=color_fitness, 
                                  label='Fitness', 
                                  alpha=1.0)

        # Then, plot the Pareto front as a curve
        pareto_front_x_sorted, pareto_front_y_sorted = zip(*sorted_pareto_front)
        matplt.plot(pareto_front_x_sorted, 
                               pareto_front_y_sorted, 
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

        if not save_dir is None:
            matplt.savefig(fname=save_dir, 
                                      bbox_inches='tight')

        matplt.show()

        return fig
