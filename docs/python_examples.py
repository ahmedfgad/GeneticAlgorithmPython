"""Render example cards and their index from the same catalog and templates."""

import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from docutils import nodes
from docutils.statemachine import StringList
from jinja2 import Environment, FileSystemLoader
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxDirective


def load_example_catalog(app):
    """Validate the catalog against the scripts shipped in the repository."""
    documentation_directory = Path(app.confdir).parent
    examples_directory = documentation_directory.parent / 'examples'
    catalog_path = documentation_directory / 'python_examples.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    examples_by_path = {}
    for example in catalog:
        path = example['path']
        if path in examples_by_path:
            raise ExtensionError(f"Duplicate Python example in the catalog: {path}")
        resolved_path = (examples_directory / path).resolve()
        try:
            resolved_path.relative_to(examples_directory.resolve())
        except ValueError:
            raise ExtensionError(f"Python example must be inside examples/: {path}")
        if not resolved_path.is_file():
            raise ExtensionError(f"Python example does not exist in examples/: {path}")
        if not (Path(app.confdir) / example['guide']).is_file():
            raise ExtensionError(f"Documentation guide does not exist for Python example: {path}")
        examples_by_path[path] = example

    missing_examples = sorted(path.relative_to(examples_directory).as_posix()
                              for path in examples_directory.rglob('*.py')
                              if path.relative_to(examples_directory).as_posix() not in examples_by_path)
    if missing_examples:
        raise ExtensionError('Add these scripts to docs/python_examples.json: ' + ', '.join(missing_examples))
    app.python_examples_catalog = catalog


class PythonExamples(SphinxDirective):
    """Show cards for catalog paths, or a compact index of all examples."""

    has_content = True
    render_examples_index = False

    def run(self):
        documentation_directory = Path(self.env.srcdir).parent
        self.env.note_dependency(str(documentation_directory / 'python_examples.json'))
        examples_by_path = {example['path']: example for example in self.env.app.python_examples_catalog}
        if self.render_examples_index:
            examples = list(examples_by_path.values())
        else:
            paths = [line.strip() for line in self.content if line.strip()]
            if not paths:
                raise self.error('List at least one path relative to examples/.')
            try:
                examples = [examples_by_path[path] for path in paths]
            except KeyError as error:
                raise self.error(f"Unknown Python example in docs/python_examples.json: {error.args[0]}")

        source_revision = quote(self.config.python_examples_source_revision, safe='')
        repository_url = 'https://github.com/ahmedfgad/GeneticAlgorithmPython'
        examples_with_links = []
        for example in examples:
            folder = PurePosixPath(example['path']).parent.as_posix()
            folder_url = f"{repository_url}/tree/{source_revision}/examples"
            if folder != '.':
                folder_url += '/' + quote(folder)
            examples_with_links.append(dict(
                example,
                source_url=f"{repository_url}/blob/{source_revision}/examples/{quote(example['path'])}",
                folder_url=folder_url,
                data_url=f"{repository_url}/blob/{source_revision}/examples/data/README.md",
                download_path='../../examples/' + example['path']))
        template_name = 'index.md.jinja' if self.render_examples_index else 'card.md.jinja'
        templates_directory = documentation_directory / 'python_example_templates'
        self.env.note_dependency(str(templates_directory / template_name))
        self.env.note_dependency(str(templates_directory / 'table.md.jinja'))
        template_environment = Environment(loader=FileSystemLoader(str(templates_directory)),
                                           autoescape=False, keep_trailing_newline=True)
        categories = [(category, [example for example in examples_with_links
                                  if example['category'] == category])
                      for category in dict.fromkeys(example['category'] for example in examples_with_links)]
        rendered_content = template_environment.get_template(template_name).render(
            examples=examples_with_links, categories=categories)
        container = nodes.container()
        # MyST parses the generated Markdown through its normal Sphinx state.
        # Native cards, dropdowns, and downloads also support non-HTML builders.
        self.state.nested_parse(StringList(rendered_content.splitlines()), 0,
                                container, match_titles=self.render_examples_index)
        return container.children


class PythonExamplesIndex(PythonExamples):
    """Show every catalog entry in topic tables, with links back to guides."""

    render_examples_index = True


def setup(app):
    """Register the shared example catalog and its two documentation directives."""
    app.add_config_value('python_examples_source_revision', 'master', 'env')
    app.connect('builder-inited', load_example_catalog)
    app.add_directive('python-examples', PythonExamples)
    app.add_directive('python-examples-index', PythonExamplesIndex)
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}
