.. _pygadhelper-module:

``pygad.helper`` Module
=======================

This section of the PyGAD's library documentation discusses the
**pygad.helper** module.

Yet, this module has a submodule called ``unique`` that has a class
named ``Unique`` with the following helper methods. Such methods help to
check and fix duplicate values in the genes of a solution.

-  ``solve_duplicate_genes_randomly()``: Solves the duplicates in a
   solution by randomly selecting new values for the duplicating genes.

-  ``solve_duplicate_genes_by_space()``: Solves the duplicates in a
   solution by selecting values for the duplicating genes from the gene
   space

-  ``unique_int_gene_from_range()``: Finds a unique integer value for
   the gene.

-  ``unique_genes_by_space()``: Loops through all the duplicating genes
   to find unique values that from their gene spaces to solve the
   duplicates. For each duplicating gene, a call to the
   ``unique_gene_by_space()`` is made.

-  ``unique_gene_by_space()``: Returns a unique gene value for a single
   gene based on its value space to solve the duplicates.
