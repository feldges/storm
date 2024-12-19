import json
import copy
import threading
from contextlib import contextmanager
from fasthtml.common import *

#------------------------------------------------------------------------------
database_path = "data/investor_reports.db"
#------------------------------------------------------------------------------

_thread_local = threading.local()

@contextmanager
def get_db_connection():
    if not hasattr(_thread_local, "db"):
        _thread_local.db = database(database_path)
    try:
        yield _thread_local.db
    finally:
        if hasattr(_thread_local, "db"):
            # Close the connection
            _thread_local.db.close()
            # Remove the db attribute
            delattr(_thread_local, "db")

# Help function to handle non-serializable contents
def handle_non_serializable(obj): return "non-serializable contents"

def dump_json(obj):
    return json.dumps(obj, default=handle_non_serializable)

def dump_url_to_info(information_table):
    url_to_info = copy.deepcopy(information_table.url_to_info)
    for url in url_to_info:
        url_to_info[url] = url_to_info[url].to_dict()
    return json.dumps(url_to_info, default=handle_non_serializable)

def dump_outline_to_file(outline):
    outline_list = outline.get_outline_as_list(add_hashtags=True, include_root=False)
    return "\n".join(outline_list)

def to_string(article) -> str:
    """
    Get outline of the article as a list.

    Returns:
        list of section and subsection names.
    """
    result = []

    def preorder_traverse(node, level):
        prefix = "#" * level
        result.append(f"{prefix} {node.section_name}".strip())
        result.append(node.content)
        for child in node.children:
            preorder_traverse(child, level + 1)

    # Adjust the initial level based on whether root is included and hashtags are added
    for child in article.root.children:
        preorder_traverse(child, level=1)
    result = [i.strip() for i in result if i is not None and i.strip()]
    return "\n\n".join(result)

def dump_article_as_plain_text(article):
    return to_string(article)

def dump_reference_to_db(article):
    reference = copy.deepcopy(article.reference)
    for url in reference["url_to_info"]:
        reference["url_to_info"][url] = reference["url_to_info"][url].to_dict()
    return dump_json(reference)

def prepare_calls_for_db(llm_call_history):
    calls_list = []
    for call in llm_call_history:
        if "kwargs" in call:
            call.pop("kwargs")
        calls_list.append(call)
    return json.dumps(calls_list)
