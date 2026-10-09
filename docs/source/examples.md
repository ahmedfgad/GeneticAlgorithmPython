# Examples

Find complete Python scripts by topic, open their source on GitHub, or download examples that need no companion files. Each entry links back to its documentation guide. The same examples appear in **Python example** cards beside the relevant explanations in those guides.

## Running the Examples

For examples from this documentation revision, use the matching repository version of PyGAD. This is especially important for features in the Unreleased notes. Clone or download that repository revision, then install it from the repository root:

```console
python -m pip install -e ".[visualize]"
```

Run a script using its path, for example:

```console
python examples/plots/example_plot_lifecycle.py
```

For a script downloaded separately, run `python` with the path where you saved it. It still requires a compatible installed version of PyGAD. Plotting examples need Matplotlib; PDF reports need `pygad[report]`. Keras and PyTorch examples need their respective frameworks. The **Run this example** dropdowns in the guides give additional requirements and working directories.

Examples requiring datasets link to their folders instead of offering a standalone download. Their datasets are not bundled with PyGAD or the repository. Follow the linked data setup instructions and keep the repository directory layout. The TSP notebook is listed alongside the Python scripts; it uses Google Colab and a user-supplied CSV. Adapt its Colab-specific imports and CSV path before running it locally with Jupyter.

Use documentation search or your browser's find command to locate a topic or filename on this page.

<!-- python-examples-index -->

## Getting Started

| Python script | What it shows | Related information |
| --- | --- | --- |
| [example.py](../../examples/example.py) | **First GA run.** Optimize a linear equation, inspect the best solution, plot fitness, and save and reload the GA. | [Guide](steps_to_use.md) |

## Population and Genes

| Python script | What it shows | Related information |
| --- | --- | --- |
| [example_initial_population.py](../../examples/example_initial_population.py) | **Initial populations.** Generate values from per-gene ranges and nested spaces, or supply values and infer the dimensions. | [Guide](gene_values.md) |
| [example_gene_space.py](../../examples/example_gene_space.py) | **Gene spaces.** Compare shared and per-gene choices, ranges, dictionaries, fixed values, and None entries. | [Guide](gene_values.md) |
| [example_gene_constraint.py](../../examples/example_gene_constraint.py) | **Gene constraints.** Filter gene candidates with constraints that depend on other genes. | [Guide](gene_values.md) |
| [example_duplicate_gene_repair.py](../../examples/example_duplicate_gene_repair.py) | **Duplicate repair.** Repair duplicates through a chain of replacements while respecting each gene space. | [Guide](gene_values.md) |
| [example_gene_type_conversion.py](../../examples/example_gene_type_conversion.py) | **Gene types and rounding.** Preserve mixed numeric types, apply precision, and convert custom mutation outputs. | [Guide](gene_values.md) |
| [example_dynamic_population_size.py](../../examples/example_dynamic_population_size.py) | **Changing population size.** Adjust the population and related runtime settings during evolution. | [Guide](generations.md) |

## Operators and Configuration

| Python script | What it shows | Related information |
| --- | --- | --- |
| [example_custom_operators.py](../../examples/example_custom_operators.py) | **Custom GA operators.** Implement parent selection, crossover, and mutation functions. | [Guide](user_defined_operators.md) |
| [example_constructor_parameters.py](../../examples/example_constructor_parameters.py) | **Constructor settings and random seeds.** Use callable fitness signatures, NumPy counts, and independent seeded GA instances. | [Guide](pygad.md) |

## Fitness and Parallel Processing

| Python script | What it shows | Related information |
| --- | --- | --- |
| [example_fitness_batch_size.py](../../examples/example_fitness_batch_size.py) | **Batch fitness.** Return one fitness result per solution, including a shorter final batch. | [Guide](fitness_calculation.md) |
| [example_parallel_processing.py](../../examples/example_parallel_processing.py) | **Parallel fitness.** Evaluate population fitness with process workers and report the run time. | [Guide](fitness_calculation.md) |
| [benchmarks/parallel_processing.py](../../examples/benchmarks/parallel_processing.py) | **Compare fitness execution modes.** Measure complete runs for CPU, I/O, and NumPy workloads with serial, thread, process, and batch evaluation. | [Guide](fitness_calculation.md) |
| [example_fitness_wrapper.py](../../examples/example_fitness_wrapper.py) | **Extra fitness arguments.** Wrap a fitness function to pass additional values while preserving its PyGAD signature. | [Guide](custom_functions.md) |

## Lifecycle and Saved Runs

| Python script | What it shows | Related information |
| --- | --- | --- |
| [pygad_lifecycle.py](../../examples/pygad_lifecycle.py) | **Lifecycle callbacks.** Trace fitness, parent selection, crossover, mutation, generation, and stop callbacks. | [Guide](lifecycle.md) |
| [example_lifecycle_methods.py](../../examples/example_lifecycle_methods.py) | **Callbacks as methods.** Implement fitness and lifecycle callbacks with bound methods. | [Guide](custom_functions.md) |
| [example_lifecycle_classes.py](../../examples/example_lifecycle_classes.py) | **Callbacks as callable classes.** Implement fitness and lifecycle callbacks with callable class instances. | [Guide](custom_functions.md) |
| [example_summary.py](../../examples/example_summary.py) | **Text lifecycle summary.** Print the configured GA stages and their parameters. | [Guide](pygad_more.md) |
| [example_logger.py](../../examples/example_logger.py) | **Logging.** Send progress and GA messages to a configured logger. | [Guide](logging.md) |
| [example_repeated_runs.py](../../examples/example_repeated_runs.py) | **Repeated runs and checkpoints.** Continue from a saved GA and inspect the actual generation numbers in its histories. | [Guide](generations.md) |
| [example_load_fitness_function.py](../../examples/example_load_fitness_function.py) | **Change a loaded fitness function.** Replace the fitness callable after loading, or start fresh when the objective changes. | [Guide](pygad.md) |

## Multi-Objective Optimization

| Python script | What it shows | Related information |
| --- | --- | --- |
| [example_multi_objective.py](../../examples/example_multi_objective.py) | **NSGA-II optimization.** Optimize two objectives and inspect the resulting trade-offs. | [Guide](multi_objective.md) |
| [example_multi_objective_nsga3.py](../../examples/example_multi_objective_nsga3.py) | **NSGA-III optimization.** Configure reference points and optimize two objectives with NSGA-III. | [Guide](multi_objective.md) |

## Plots

| Python script | What it shows | Related information |
| --- | --- | --- |
| [plots/example_plot_fitness.py](../../examples/plots/example_plot_fitness.py) | **Best-fitness curve.** Plot best fitness across generations on the Sphere benchmark. | [Guide](visualize.md) |
| [plots/example_plot_fitness_band.py](../../examples/plots/example_plot_fitness_band.py) | **Fitness band.** Plot per-generation minimum, mean, and maximum fitness with a shaded band. | [Guide](visualize.md) |
| [plots/example_plot_genes.py](../../examples/plots/example_plot_genes.py) | **Gene histories.** Show how gene values change across saved generations. | [Guide](visualize.md) |
| [plots/example_plot_lifecycle.py](../../examples/plots/example_plot_lifecycle.py) | **Configured lifecycle.** Draw detailed and compact lifecycle charts and export SVG and PNG files. | [Guide](visualize.md) |
| [plots/example_plot_new_solution_rate.py](../../examples/plots/example_plot_new_solution_rate.py) | **New-solution rate.** Count previously unseen solutions in each generation. | [Guide](visualize.md) |
| [plots/example_plot_non_dominated_hypervolume.py](../../examples/plots/example_plot_non_dominated_hypervolume.py) | **Hypervolume history.** Track the hypervolume of the non-dominated set across generations. | [Guide](visualize.md) |
| [plots/example_plot_pareto_front_curve_2d.py](../../examples/plots/example_plot_pareto_front_curve_2d.py) | **2D Pareto front.** Plot a two-objective Pareto front after NSGA-II optimization. | [Guide](visualize.md) |
| [plots/example_plot_pareto_front_curve_3d.py](../../examples/plots/example_plot_pareto_front_curve_3d.py) | **3D Pareto front.** Plot a three-objective Pareto front after NSGA-III optimization. | [Guide](visualize.md) |
| [plots/example_plot_pareto_front_evolution.py](../../examples/plots/example_plot_pareto_front_evolution.py) | **Pareto-front evolution.** Overlay the non-dominated fronts from selected generations. | [Guide](visualize.md) |
| [plots/example_plot_pareto_front_heatmap.py](../../examples/plots/example_plot_pareto_front_heatmap.py) | **Pareto heatmap.** Compare objective values with a solutions-by-objectives heatmap. | [Guide](visualize.md) |
| [plots/example_plot_pareto_front_pcp.py](../../examples/plots/example_plot_pareto_front_pcp.py) | **Parallel coordinates.** Compare Pareto solutions across objective axes. | [Guide](visualize.md) |
| [plots/example_plot_pareto_front_scatter_matrix.py](../../examples/plots/example_plot_pareto_front_scatter_matrix.py) | **Pareto scatter matrix.** Compare every pair of objectives in a many-objective run. | [Guide](visualize.md) |
| [plots/example_plot_population_diversity.py](../../examples/plots/example_plot_population_diversity.py) | **Population diversity.** Track mean pairwise distance between solutions across generations. | [Guide](visualize.md) |

## Reports

| Python script | What it shows | Related information |
| --- | --- | --- |
| [example_generate_report.py](../../examples/example_generate_report.py) | **PDF report.** Export the run configuration, summary, best solution, and applicable plots to PDF. | [Guide](pygad.md) |

## Benchmarks

| Python script | What it shows | Related information |
| --- | --- | --- |
| [benchmarks/example_classic_sphere.py](../../examples/benchmarks/example_classic_sphere.py) | **Sphere.** Optimize the Sphere single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_rastrigin.py](../../examples/benchmarks/example_classic_rastrigin.py) | **Rastrigin.** Optimize the Rastrigin single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_rosenbrock.py](../../examples/benchmarks/example_classic_rosenbrock.py) | **Rosenbrock.** Optimize the Rosenbrock single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_griewank.py](../../examples/benchmarks/example_classic_griewank.py) | **Griewank.** Optimize the Griewank single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_schwefel.py](../../examples/benchmarks/example_classic_schwefel.py) | **Schwefel.** Optimize the Schwefel single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_ackley.py](../../examples/benchmarks/example_classic_ackley.py) | **Ackley.** Optimize the Ackley single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_himmelblau.py](../../examples/benchmarks/example_classic_himmelblau.py) | **Himmelblau.** Optimize the Himmelblau single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt1.py](../../examples/benchmarks/example_zdt1.py) | **ZDT1.** Optimize the ZDT1 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt2.py](../../examples/benchmarks/example_zdt2.py) | **ZDT2.** Optimize the ZDT2 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt3.py](../../examples/benchmarks/example_zdt3.py) | **ZDT3.** Optimize the ZDT3 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt4.py](../../examples/benchmarks/example_zdt4.py) | **ZDT4.** Optimize the ZDT4 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt6.py](../../examples/benchmarks/example_zdt6.py) | **ZDT6.** Optimize the ZDT6 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_dtlz1.py](../../examples/benchmarks/example_dtlz1.py) | **DTLZ1.** Optimize the DTLZ1 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_dtlz2.py](../../examples/benchmarks/example_dtlz2.py) | **DTLZ2.** Optimize the DTLZ2 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_dtlz3.py](../../examples/benchmarks/example_dtlz3.py) | **DTLZ3.** Optimize the DTLZ3 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_dtlz4.py](../../examples/benchmarks/example_dtlz4.py) | **DTLZ4.** Optimize the DTLZ4 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_knapsack.py](../../examples/benchmarks/example_knapsack.py) | **Knapsack.** Select items to maximize value within a weight capacity. | [Guide](benchmarks.md) |
| [benchmarks/example_tsp.py](../../examples/benchmarks/example_tsp.py) | **Travelling salesman.** Find a short tour using a permutation of four cities. | [Guide](benchmarks.md) |
| [example_travelling_salesman.ipynb](../../examples/example_travelling_salesman.ipynb) | **Travelling-salesman Colab notebook.** Explore a city-tour problem using a user-supplied CSV and interactive maps. | [Guide](benchmarks.md) · [Folder](../../examples/./) · [Data setup](../../examples/data/README.md) |

## Quality Indicators

| Python script | What it shows | Related information |
| --- | --- | --- |
| [quality_indicators/example_hypervolume.py](../../examples/quality_indicators/example_hypervolume.py) | **Hypervolume.** Measure the objective-space volume dominated by the final population. | [Guide](utils.md) |
| [quality_indicators/example_inverted_generational_distance.py](../../examples/quality_indicators/example_inverted_generational_distance.py) | **Inverted generational distance.** Measure distance from a reference front to the approximation. | [Guide](utils.md) |
| [quality_indicators/example_generational_distance.py](../../examples/quality_indicators/example_generational_distance.py) | **Generational distance.** Measure distance from the approximation to a reference front. | [Guide](utils.md) |
| [quality_indicators/example_spacing.py](../../examples/quality_indicators/example_spacing.py) | **Spacing.** Measure how evenly the approximation points are spread. | [Guide](utils.md) |

## Neural Networks

| Python script | What it shows | Related information |
| --- | --- | --- |
| [nn/example_regression.py](../../examples/nn/example_regression.py) | **Regression.** Fit a neural network to a small numeric regression problem. | [Guide](nn_regression_1.md) |
| [nn/example_XOR_classification.py](../../examples/nn/example_XOR_classification.py) | **XOR classification.** Train a neural network on the four XOR inputs. | [Guide](nn_xor.md) |
| [nn/example_classification.py](../../examples/nn/example_classification.py) | **Image classification.** Classify fruit images from prepared feature vectors. | [Guide](nn_image_classification.md) · [Folder](../../examples/nn/) · [Data setup](../../examples/data/README.md) |
| [nn/example_regression_fish.py](../../examples/nn/example_regression_fish.py) | **Fish-weight regression.** Predict fish weight from numeric measurements. | [Guide](nn_regression_2.md) · [Folder](../../examples/nn/) · [Data setup](../../examples/data/README.md) |
| [nn/extract_features.py](../../examples/nn/extract_features.py) | **Prepare image features.** Extract fruit-image features and write the arrays used by the dense classifiers. | [Guide](nn_image_classification.md) · [Folder](../../examples/nn/) · [Data setup](../../examples/data/README.md) |

## Neural Networks with the GA

| Python script | What it shows | Related information |
| --- | --- | --- |
| [gann/example_regression.py](../../examples/gann/example_regression.py) | **Regression.** Fit a neural network to a small numeric regression problem. | [Guide](gann_regression_1.md) |
| [gann/example_XOR_classification.py](../../examples/gann/example_XOR_classification.py) | **XOR classification.** Train a neural network on the four XOR inputs. | [Guide](gann_xor.md) |
| [gann/example_classification.py](../../examples/gann/example_classification.py) | **Image classification.** Classify fruit images from prepared feature vectors. | [Guide](gann_image_classification.md) · [Folder](../../examples/gann/) · [Data setup](../../examples/data/README.md) |
| [gann/example_regression_fish.py](../../examples/gann/example_regression_fish.py) | **Fish-weight regression.** Predict fish weight from numeric measurements. | [Guide](gann_regression_2.md) · [Folder](../../examples/gann/) · [Data setup](../../examples/data/README.md) |

## Convolutional Networks

| Python script | What it shows | Related information |
| --- | --- | --- |
| [cnn/example_image_classification.py](../../examples/cnn/example_image_classification.py) | **Build a CNN.** Classify fruit images using prepared image arrays. | [Guide](cnn.md) · [Folder](../../examples/cnn/) · [Data setup](../../examples/data/README.md) |
| [gacnn/example_image_classification.py](../../examples/gacnn/example_image_classification.py) | **Optimize a CNN with the GA.** Classify fruit images using prepared image arrays. | [Guide](gacnn.md) · [Folder](../../examples/gacnn/) · [Data setup](../../examples/data/README.md) |

## Keras

| Python script | What it shows | Related information |
| --- | --- | --- |
| [KerasGA/regression_example.py](../../examples/KerasGA/regression_example.py) | **Regression.** Optimize neural-network weights with the genetic algorithm. | [Guide](kerasga_regression.md) |
| [KerasGA/XOR_classification.py](../../examples/KerasGA/XOR_classification.py) | **XOR classification.** Optimize neural-network weights with the genetic algorithm. | [Guide](kerasga_xor.md) |
| [KerasGA/image_classification_Dense.py](../../examples/KerasGA/image_classification_Dense.py) | **Dense image classifier.** Train an image classifier with the genetic algorithm. | [Guide](kerasga_image_dense.md) · [Folder](../../examples/KerasGA/) · [Data setup](../../examples/data/README.md) |
| [KerasGA/image_classification_CNN.py](../../examples/KerasGA/image_classification_CNN.py) | **Convolutional image classifier.** Train an image classifier with the genetic algorithm. | [Guide](kerasga_image_conv.md) · [Folder](../../examples/KerasGA/) · [Data setup](../../examples/data/README.md) |
| [KerasGA/cancer_dataset.py](../../examples/KerasGA/cancer_dataset.py) | **Image-directory classification.** Use directory-based image input for a two-class Keras CNN. | [Guide](kerasga_image_datagen.md) · [Folder](../../examples/KerasGA/) · [Data setup](../../examples/data/README.md) |
| [KerasGA/cancer_dataset_generator.py](../../examples/KerasGA/cancer_dataset_generator.py) | **Batched image-directory classification.** Use directory-based image input for a two-class Keras CNN. | [Guide](kerasga_image_datagen.md) · [Folder](../../examples/KerasGA/) · [Data setup](../../examples/data/README.md) |

## PyTorch

| Python script | What it shows | Related information |
| --- | --- | --- |
| [TorchGA/regression_example.py](../../examples/TorchGA/regression_example.py) | **Regression.** Optimize neural-network weights with the genetic algorithm. | [Guide](torchga_regression.md) |
| [TorchGA/XOR_classification.py](../../examples/TorchGA/XOR_classification.py) | **XOR classification.** Optimize neural-network weights with the genetic algorithm. | [Guide](torchga_xor.md) |
| [TorchGA/image_classification_Dense.py](../../examples/TorchGA/image_classification_Dense.py) | **Dense image classifier.** Train an image classifier with the genetic algorithm. | [Guide](torchga_image_dense.md) · [Folder](../../examples/TorchGA/) · [Data setup](../../examples/data/README.md) |
| [TorchGA/image_classification_CNN.py](../../examples/TorchGA/image_classification_CNN.py) | **Convolutional image classifier.** Train an image classifier with the genetic algorithm. | [Guide](torchga_image_conv.md) · [Folder](../../examples/TorchGA/) · [Data setup](../../examples/data/README.md) |

## Clustering

| Python script | What it shows | Related information |
| --- | --- | --- |
| [clustering/example_clustering_2.py](../../examples/clustering/example_clustering_2.py) | **2-cluster example.** Optimize 2 cluster centers for generated two-dimensional data. | [Guide](pygad.md) |
| [clustering/example_clustering_3.py](../../examples/clustering/example_clustering_3.py) | **3-cluster example.** Optimize 3 cluster centers for generated two-dimensional data. | [Guide](pygad.md) |

<!-- /python-examples-index -->
