"""
The pygad.visualize.plot module has methods to create plots.
"""

import numpy
# import matplotlib.pyplot
import pygad

def get_matplotlib():
    # Importing matplotlib.pyplot at the module scope causes performance issues.
    # This causes matplotlib.pyplot to be imported once pygad is imported.
    # An efficient approach is to import matplotlib.pyplot only when needed.
    # Inside each function, call get_matplotlib() to return the library object.
    # If a function called get_matplotlib() once, then the library object is reused.
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
        Creates, shows, and returns a figure that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14. Can be a list/tuple/numpy.ndarray if the problem is multi-objective optimization.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#64f20c". Can be a list/tuple/numpy.ndarray if the problem is multi-objective optimization.
            label: The label used for the legend in the figures of multi-objective problems. It is not used for single-objective problems.
            save_dir: Directory to save the figure.

        Returns the figure.
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
                # Return the fitness values for the current objective function across all best solutions acorss all generations.
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
        Creates, shows, and returns a figure that summarizes the rate of exploring new solutions. This method works only when save_solutions=True in the constructor of the pygad.GA class.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#64f20c".
            save_dir: Directory to save the figure.

        Returns the figure.
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
        Creates, shows, and returns a figure with number of subplots equal to the number of genes. Each subplot shows the gene value for each generation. 
        This method works only when save_solutions=True in the constructor of the pygad.GA class. 
        It also works only after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            graph_type: Type of the graph which can be either "plot" (default), "boxplot", or "histogram".
            fill_color: Fill color of the graph which defaults to "#64f20c". This has no effect if graph_type="plot".
            color: Color of the plot which defaults to "black".
            solutions: Defaults to "all" which means use all solutions. If "best" then only the best solutions are used.
            save_dir: Directory to save the figure.

        Returns the figure.
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
            boxeplots = ax.boxplot(solutions_to_plot, 
                                   labels=range(self.num_genes),
                                   patch_artist=True)
            # adding horizontal grid lines
            ax.yaxis.grid(True)
    
            for box in boxeplots['boxes']:
                # change outline color
                box.set(color='black', linewidth=linewidth)
                # change fill color https://color.adobe.com/create/color-wheel
                box.set_facecolor(fill_color)

            for whisker in boxeplots['whiskers']:
                whisker.set(color=color, linewidth=linewidth)
            for median in boxeplots['medians']:
                median.set(color=color, linewidth=linewidth)
            for cap in boxeplots['caps']:
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
        Creates, shows, and returns the pareto front curve. Can only be used with multi-objective problems.
        It only works with 2 objectives. 
        It also works only after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            label: The label used for the legend.
            color: Color of the plot.
            color_fitness: Color of the fitness points.
            grid: Either True or False to control the visibility of the grid.
            alpha: The transparency of the pareto front curve.
            marker: The marker of the fitness points.
            save_dir: Directory to save the figure.

        Returns the figure.
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
