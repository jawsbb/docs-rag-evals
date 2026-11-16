from docs_rag.mdx import clean_mdx, parse_frontmatter

PAGE = """---
id: 'uploads'
title: 'Resumable Uploads'
description: 'Learn how to upload files.'
---

import Foo from '~/components/Foo'

Intro paragraph.

{/* an editor comment */}

<Admonition type="note" label="Heads up">

Use the direct storage hostname.

</Admonition>

<Image alt="diagram" src={{ light: '/a.png', dark: '/b.png' }} width={800} />

<Tabs scrollable defaultActiveId="js">
  <TabPanel id="js" label="JavaScript">

    ```javascript name=upload.js
    const x = a > b ? 1 : 2 // <Tag> inside code stays
    ```

  </TabPanel>
</Tabs>

Set `<SUPABASE_URL>` and pass <PROJECT_REF> in the header.

<$Partial path="note.mdx" />
"""


def test_frontmatter_is_parsed_and_removed():
    meta, body = parse_frontmatter(PAGE)
    assert meta["title"] == "Resumable Uploads"
    assert meta["description"] == "Learn how to upload files."
    assert not body.startswith("---")


def test_jsx_is_stripped_but_children_kept():
    doc = clean_mdx(PAGE)
    assert doc.title == "Resumable Uploads"
    assert "Use the direct storage hostname." in doc.body
    for noise in ("<Admonition", "</Admonition>", "<Tabs", "TabPanel", "<Image", "editor comment", "import Foo"):
        assert noise not in doc.body


def test_code_fence_is_dedented_and_untouched():
    doc = clean_mdx(PAGE)
    assert "```javascript\nconst x = a > b ? 1 : 2 // <Tag> inside code stays\n```" in doc.body


def test_placeholders_and_inline_code_survive():
    doc = clean_mdx(PAGE)
    assert "`<SUPABASE_URL>`" in doc.body
    assert "<PROJECT_REF>" in doc.body


def test_partials_are_inlined_through_resolver():
    partials = {"note.mdx": "---\ntitle: x\n---\n\n<Admonition>Partial text.</Admonition>\n"}
    doc = clean_mdx(PAGE, resolve_partial=partials.get)
    assert "Partial text." in doc.body
    assert "$Partial" not in clean_mdx(PAGE).body  # unresolved partials are dropped, not leaked


def test_attribute_with_angle_bracket_does_not_truncate_tag():
    doc = clean_mdx("<Foo render={() => a > b} />\nAfter the tag.\n")
    assert doc.body == "After the tag."


def test_markdown_table_rules_and_padding_are_collapsed():
    doc = clean_mdx("| Name     | Value |\n| -------- | ----- |\n| `a`      | 1     |\n")
    assert doc.body == "| Name | Value |\n| `a` | 1 |"
