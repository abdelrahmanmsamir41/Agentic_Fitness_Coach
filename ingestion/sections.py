"""Split per-page markdown into one section per page."""


def split_sections(pages):
    """One section per page, never merged -- each page is classified and
    extracted independently downstream, even if near-empty.

    Returns a list of {"text": str, "page": int} in document order.
    """
    sections = []
    for page in pages:
        print(f"✂️ page {page['page_index']+1} split into a section")
        sections.append({
            "page": page["page_index"],
            "text": page["markdown"].strip()
        })

    return sections
