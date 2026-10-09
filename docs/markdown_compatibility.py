"""Keep Markdown readable directly while retaining the Sphinx presentation."""

import argparse
import html
import json
from pathlib import Path, PurePosixPath
import re
from string import Template


DOCUMENTATION_DIRECTORY = Path(__file__).resolve().parent
EXAMPLE_BLOCK = re.compile(
    r'<!-- (?P<kind>python-examples(?:-index)?)(?P<paths>.*?)-->\n'
    r'(?P<body>.*?)\n<!-- /(?P=kind) -->', re.DOTALL)


def render_markdown_examples(catalog, paths, render_index=False):
    """Render checked-in example links and instructions from the shared catalog."""
    examples_by_path = {example['path']: example for example in catalog}
    examples = catalog if render_index else [examples_by_path[path] for path in paths]
    if not examples:
        raise ValueError('List at least one Python example in the Markdown marker.')
    templates_directory = DOCUMENTATION_DIRECTORY / 'python_example_templates'
    card_template = Template((templates_directory / 'card-source.md.template').read_text(encoding='utf-8'))
    row_template = Template((templates_directory / 'table-source.md.template').read_text(encoding='utf-8'))

    def render_table(group):
        rows = ['| Python script | What it shows | Related information |', '| --- | --- | --- |']
        for example in group:
            folder = PurePosixPath(example['path']).parent.as_posix()
            links = f"[Guide]({example['guide']})"
            if not example.get('download', True):
                links += f' · [Folder](../../examples/{folder}/) · [Data setup](../../examples/data/README.md)'
            rows.append(row_template.substitute(
                path=example['path'], title=example['title'], description=example['description'],
                links=links).strip())
        return '\n'.join(rows)

    if render_index:
        categories = dict.fromkeys(example['category'] for example in examples)
        return '\n\n'.join(f'## {category}\n\n' + render_table([
            example for example in examples if example['category'] == category]) for category in categories)

    cards = []
    group_instructions = []
    for example in examples:
        data_instructions = ''
        if example.get('data'):
            data_instructions = (f"**Data:** {example['data']} "
                                 'See the [dataset setup instructions](../../examples/data/README.md).\n\n')
        run_instructions = ''
        if example['run']:
            run_note = example.get('run_note', 'From the repository root, with the repository version of PyGAD installed:')
            run_instructions = f"{run_note}\n\n```console\n{example['run']}\n```\n\n"
        cards.append(card_template.substitute(
            title=example['title'], path=example['path'], description=example['description'],
            requirements=example['requirements'], data_instructions=data_instructions,
            run_instructions=run_instructions).strip())
        group_instructions.append((f"**{example['title']}** — Requires: {example['requirements']}\n\n"
                                   + data_instructions + run_instructions).strip())
    if len(examples) > 3:
        instructions = '<details>\n<summary>Run these examples</summary>\n\n'
        instructions += '\n\n'.join(group_instructions).strip() + '\n\n</details>'
        return '**Python examples**\n\n' + render_table(examples) + '\n\n' + instructions
    title = '**Python example**' if len(examples) == 1 else '**Python examples**'
    return title + '\n\n' + '\n\n'.join(cards)


def update_example_blocks(source, catalog, check_only=False):
    """Refresh example sections, or reject stale sections during a build."""
    marker_count = len(re.findall(r'<!-- python-examples(?:-index)?(?=\s)', source))
    closing_marker_count = len(re.findall(r'<!-- /python-examples(?:-index)? -->', source))
    if marker_count != closing_marker_count or marker_count != len(list(EXAMPLE_BLOCK.finditer(source))):
        raise ValueError('Each Python example section needs a matching closing comment.')

    def replace_block(match):
        paths = match['paths'].split()
        render_index = match['kind'] == 'python-examples-index'
        if render_index and paths:
            raise ValueError('The Python examples index includes the whole catalog and does not accept paths.')
        try:
            expected = render_markdown_examples(catalog, paths, render_index)
        except KeyError as error:
            raise ValueError(f'Unknown Python example in docs/python_examples.json: {error.args[0]}') from error
        if check_only and match['body'].strip() != expected:
            raise ValueError('Python example content is out of date. Run python docs/markdown_compatibility.py --update-examples.')
        if check_only:
            return match.group(0)
        header = '<!-- ' + match['kind'] + ('\n' + '\n'.join(paths) + '\n' if paths else ' ') + '-->'
        return header + '\n\n' + expected + '\n\n<!-- /' + match['kind'] + ' -->'
    return EXAMPLE_BLOCK.sub(replace_block, source)


def prepare_sphinx_source(app, document_name, source):
    """Restore presentation directives from portable Markdown and hidden metadata."""
    if not (Path(app.srcdir) / (document_name + '.md')).is_file():
        return
    from sphinx.errors import ExtensionError

    try:
        content = update_example_blocks(source[0], app.python_examples_catalog, check_only=True)
    except (KeyError, ValueError) as error:
        raise ExtensionError(f'{document_name}: {error}') from error

    def restore_example_section(match):
        paths = '\n'.join(match['paths'].split())
        return ':::{' + match['kind'] + '}\n' + paths + '\n:::'

    content = EXAMPLE_BLOCK.sub(restore_example_section, content)
    # Labels, toctrees, and embedded videos are build metadata, not visible prose.
    content = re.sub(r'<!-- sphinx\n(.*?)\n-->', r'\1', content, flags=re.DOTALL)

    def restore_grid(match):
        cards = []
        for line in match['body'].splitlines():
            if not line.strip():
                continue
            item = re.fullmatch(r'- \[([^]]+)\]\(([^)]+)\.md\)(?: — (.*))?', line)
            if item is None:
                raise ExtensionError(f'{document_name}: Invalid navigation item: {line}')
            title, destination, description = item.groups()
            cards.append(f':::{{grid-item-card}} {title}\n:link: {destination}\n:link-type: doc\n\n{description or ""}\n:::')
        return '::::{grid} ' + match['dimensions'].strip() + '\n:gutter: 3\n\n' + '\n\n'.join(cards) + '\n::::'

    content = re.sub(r'<!-- navigation-grid: (?P<dimensions>.*?) -->\n(?P<body>.*?)\n<!-- /navigation-grid -->',
                     restore_grid, content, flags=re.DOTALL)

    def restore_figure(match):
        # The PNG works in repository previews; Sphinx can select SVG or PNG.
        image_path = str(PurePosixPath(match['path']).with_suffix('.*'))
        return (f':::{{figure}} {image_path}\n:alt: {match["alt"]}\n'
                f':width: {match["width"]}\n:align: center\n\n{match["caption"]}\n:::')

    content = re.sub(
        r'<!-- documentation-figure: (?P<width>.*?) -->\n\n'
        r'!\[(?P<alt>.*?)\]\((?P<path>.*?)\)\n\n(?P<caption>.*?)\n\n<!-- /documentation-figure -->',
        restore_figure, content, flags=re.DOTALL)

    def restore_dropdown(match):
        title = re.sub(r'<code>(.*?)</code>', r'`\1`', html.unescape(match['title']))
        return f':::{{dropdown}} {title}\n:animate: fade-in-slide-down\n\n{match["body"].strip()}\n:::'

    content = re.sub(r'<details>\n<summary>(?P<title>.*?)</summary>\n(?P<body>.*?)\n</details>',
                     restore_dropdown, content, flags=re.DOTALL)
    for filename in ['card-source.md.template', 'table-source.md.template']:
        app.env.note_dependency(str(DOCUMENTATION_DIRECTORY / 'python_example_templates' / filename))
    source[0] = content


def setup(app):
    """Use readable source Markdown for all Sphinx output formats."""
    app.connect('source-read', prepare_sphinx_source)
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}


def main():
    """Update or check generated Markdown without installing documentation tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-examples', action='store_true', help='Refresh example sections from the catalog.')
    arguments = parser.parse_args()
    catalog = json.loads((DOCUMENTATION_DIRECTORY / 'python_examples.json').read_text(encoding='utf-8'))
    for path in sorted((DOCUMENTATION_DIRECTORY / 'source').glob('*.md')):
        source = path.read_text(encoding='utf-8')
        updated = update_example_blocks(source, catalog, check_only=not arguments.update_examples)
        if arguments.update_examples and updated != source:
            path.write_text(updated, encoding='utf-8')
    print('Python example Markdown updated.' if arguments.update_examples else 'Python example Markdown is current.')


if __name__ == '__main__':
    main()
