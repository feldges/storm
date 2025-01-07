from fasthtml.common import *
from fastcore.parallel import threaded
import sys
import os
import re
import json
from dotenv import load_dotenv
import unicodedata
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo_light'))

import demo_util
from demo_util import DemoFileIOHelper, DemoTextProcessingHelper
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import YouRM, BraveRM
from knowledge_storm.utils_db import database_path, get_db_connection

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
ydc_api_key = os.getenv("YDC_API_KEY")

app_styles = """
html {
    scroll-behavior: smooth;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}
.persona-card {
    box-sizing: border-box;
    text-align: center;
    cursor: pointer;
    padding: 1rem;
    border-radius: 8px;
    transition: background 0.2s;
}
.persona-card:hover {
    background: #f0f0f0;
}
.persona-card.selected {
    background: #e3f2fd;
}
h3, h4 {
    margin-top: 1rem;  /* Reduce from default 2.5rem */
}
.opportunity-card {
    cursor: pointer;
    transition: background-color 0.2s, transform 0.2s;
    padding: 1rem;
}
.opportunity-card:hover {
    background-color: #f0f0f0;
    transform: translateY(-2px);
}
.opportunity-card.selected {
    background: #e3f2fd;
}
.opportunity-card h4 {
    margin-top: 0;
    margin-bottom: 0.5rem;
}
.opportunity-card p {
    margin: 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
}
input {
    max-width: 100%;  /* Prevent input from overflowing its container */
    box-sizing: border-box;  /* Include padding and border in width calculation */
}

/* Add these new styles */
@media (max-width: 1200px) {
    .content-wrapper {
        flex-direction: column !important;
    }

    #table_of_contents {
        display: none !important;
    }

    #article {
        width: 100% !important;
    }
}

/* Add these new styles for table of contents */
#table_of_contents ul {
    list-style: none;
    padding-left: 0;
}

#table_of_contents a {
    text-decoration: none;
    color: #1976d2;
    transition: color 0.2s;
}

#table_of_contents a:hover {
    color: #1565c0;
}

/* Existing TOC level styles */
.toc-level-2 {
    margin-left: 1rem;
}
.toc-level-3 {
    margin-left: 2rem;
}
"""

hdrs = (MarkdownJS(), Style(app_styles))#, scripts, link_daisyui, link_pico)

app, rt = fast_app(pico=True, hdrs=hdrs)

info_card_style = "word-wrap: break-word; overflow-wrap: break-word; white-space: normal; color: #1976d2; background-color: #e3f2fd; border-radius: 4px"
error_card_style = "word-wrap: break-word; overflow-wrap: break-word; white-space: normal; color: #d32f2f; background-color: #ffebee; border-radius: 4px"

# Read data from the source directory and return a dictionary under table
# Deprecated - will be removed but the local_dir is still used in the code (while inactive),
# So it was decided to keep it for now.
local_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")

#-------------------------------------------------------------------------------
# Initiate the runner
def set_storm_runner():
    current_working_dir = local_dir
    #if not os.path.exists(current_working_dir):
    #    os.makedirs(current_working_dir)

    # configure STORM runner
    llm_configs = STORMWikiLMConfigs()
    llm_configs.init_openai_model(openai_api_key=openai_api_key, openai_type='openai')
    llm_configs.set_question_asker_lm(OpenAIModel(model='gpt-4-1106-preview', api_key=openai_api_key,
                                                  model_type='chat',
                                                  max_tokens=500, temperature=1.0, top_p=0.9))
    engine_args = STORMWikiRunnerArguments(
        output_dir=current_working_dir,
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=3,
        retrieve_top_k=5,
        database_path=database_path
    )

    # rm = YouRM(ydc_api_key=ydc_api_key, k=engine_args.search_top_k)
    rm = BraveRM(k=engine_args.search_top_k)

    runner = STORMWikiRunner(engine_args, llm_configs, rm)
    return runner
#-------------------------------------------------------------------------------

# Define helper functions
#-------------------------------------------------------------------------------
def name_to_id(name):
    # First convert accented characters to their ASCII equivalents
    normalized_name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    # Replace remaining special characters with underscore
    clean_id = re.sub(r'[^a-zA-Z0-9]+', '_', normalized_name.strip())
    return clean_id.strip('_').lower()

# The article starts sometimes with a # summary, which we do not want to render.
# If it starts with another topic with a title #, then we want to render it but need a \n before it.
def postprocess_article(article, citations):
    if article[:9].lower() == '# summary':
        article = article[9:]
    elif article[0] == '#':
        article = "\n" + article
    # Add inline citations to the article
    article = DemoTextProcessingHelper.add_inline_citation_link(article, citations)
    return article

def create_anchor(text):
    anchor = text.lower().replace(" ", "-").replace(".", "")
    return anchor

#-------------------------------------------------------------------------------
# Helper functions specific to
def parse(text):
    regex = re.compile(r']:\s+"(.*?)"\s+http')
    text = regex.sub(']: http', text)
    return text

def construct_citation_dict_from_search_result(search_results):
    if search_results is None:
        return None
    citation_dict = {}
    for url, index in search_results['url_to_unified_index'].items():
        citation_dict[index] = {'url': url,
                                'title': search_results['url_to_info'][url]['title'],
                                'snippets': search_results['url_to_info'][url]['snippets']}
    return citation_dict

def assemble_article_data(article_dict):
    """
    Constructs a dictionary containing the content and metadata of an article
    based on the available files in the article's directory. This includes the
    main article text, citations from a JSON file, and a conversation log if
    available. The function prioritizes a polished version of the article if
    both a raw and polished version exist.

    Args:
        article_dict (dict): A dictionary where keys are contents relevant
                                to the article (e.g., the article text, citations
                                in JSON format, conversation logs) and values
                                are their content.

    Returns:
        dict or None: A dictionary containing the parsed content of the article,
                    citations, and conversation log if available. Returns None
                    if neither the raw nor polished article text exists in the
                    provided file paths.
    """
    if "storm_gen_article" in article_dict or "storm_gen_article_polished" in article_dict:
        full_article_name = "storm_gen_article_polished" if "storm_gen_article_polished" in article_dict else "storm_gen_article"
        article_data = {"article": parse(article_dict[full_article_name])}
        if "url_to_info" in article_dict:
            article_data["citations"] = construct_citation_dict_from_search_result(
                json.loads(article_dict["url_to_info"]))
        if "conversation_log" in article_dict:
            article_data["conversation_log"] = json.loads(article_dict["conversation_log"])
        return article_data
    return None

def read_data_to_dict(opportunities):
    """
    Gets the opportunities data from the database and returns a nested dictionary.

    Args:
        Table opportunity: the table that contains the opportunity data

    Returns:
        dict: A dictionary where each key is an article name, and each value is a dictionary
            of opportunity data.
    """
    articles_dict = {}
    for opportunity in opportunities():
        opportunity_id = opportunity.id
        opportunity_content = opportunity.__dict__
        articles_dict[opportunity_id] = {}
        for key, value in opportunity_content.items():
            if value is not None:
                articles_dict[opportunity_id][key] = value
    return articles_dict

#-------------------------------------------------------------------------------
# Read data from the source directory and return a dictionary under table
#-------------------------------------------------------------------------------
# Create a database
with get_db_connection() as db:
    opportunities, users = db.t.opportunities, db.t.users
    if opportunities not in db.t:
        users.create(name=str, pk='name')
        opportunities.create(id=str, name=str, conversation_log=str, direct_gen_outline=str, llm_call_history=str,
                        raw_search_results=str, run_config=str, storm_gen_article_polished=str, storm_gen_article=str,
                        storm_gen_outline=str, url_to_info=str, user_name=str, status=str, pk='id')
    # Create types for the database tables
    Opportunities, Users = opportunities.dataclass(), users.dataclass()

def get_table():
    with get_db_connection() as db:
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        data = read_data_to_dict(opportunities)
        table = []
        for opportunity in opportunities():
            d = {}
            d['id'] = opportunity.id
            d['name'] = opportunity.name
            d['status'] = opportunity.status
            d['article'] = ""
            d['citations'] = []
            d['conversation_log'] = {}
            article_data = assemble_article_data(data[opportunity.id])
            if article_data is not None:
                citations_dict = article_data.get('citations', {})
                article_text = article_data.get('article', '')
                processed_text = postprocess_article(article_text, citations_dict)
                d['article'] = processed_text
                d['citations'] = article_data.get('citations', [])
                d['conversation_log'] = article_data.get('conversation_log', {})
            table.append(d)
    return data, table

def get_overall_status():
    with get_db_connection() as db:
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        in_progress = opportunities(where="status!='complete'")
        if in_progress and len(in_progress) > 0:
            oppo_in_progress = in_progress[0]
            return oppo_in_progress.name, oppo_in_progress.status
    return None, "complete"

def refresh_data():
    global data, table
    data, table = get_table()
    return data, table

def get_status(opportunity_id):
    with get_db_connection() as db:
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        opportunity = opportunities[opportunity_id]
        return opportunity.status

def set_status(opportunity_id, status):
    with get_db_connection() as db:
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        oppo = Opportunities(id=opportunity_id, status=status)
        db.t.opportunities.update(oppo)
    return status

data, table = refresh_data()

#-------------------------------------------------------------------------------
# Generate various HTML elements
#-------------------------------------------------------------------------------

def new_opportunity():
    oppo_name, oppo_status = get_overall_status()
    if oppo_name is not None:
        return Card(
            f"Currently working on the opportunity ", B(oppo_name), f" ({status_text[oppo_status]}). You have to wait for it to finish before you can start a new one.",
            aria_busy="true",
            style=info_card_style,
            hx_get="/new_opportunity",
            hx_trigger="every 5s",
            hx_swap_oob="true",
            id="new_opportunity"
        )
    else:
        return Card(
            #Div(
            Form(
                Label("Enter the investment opportunity you want to write an investment memo for:"),
                Group(
                    Input(name="opportunity_name", placeholder="e.g. Roche, Swiss healthcare"),
                    Button("Start")
                ),
                hx_post="/", target_id="opportunity_list", hx_swap="afterbegin"
            ),
        hx_swap_oob="true",
        id="new_opportunity"
    # Only use this for debugging. It allows to run only a specific section of the code.
    )#, Button("Debug", hx_post="/debug", hx_target="#opportunity_list", hx_swap="outerHTML"))

def opportunity_card(t, selected=False):
    return Card(
        H4(t["name"]),
        P(t["article"]),
        cls=f"opportunity-card {'selected' if selected else ''}",
        hx_get=f'/opportunity/{t["id"]}#main_wrapper',
        hx_target="#article",
        hx_swap='outerHTML',
        id=f'opportunity_{t["id"]}',
        hx_swap_oob="true"
    )

def table_of_contents(t):
    if t["article"] == "":
        return Div(" ", id="table_of_contents", style="width: 25%;", hx_swap_oob='true')
    toc_depth = 1
    toc = []
    for line in t["article"].splitlines():
        if line.startswith("#"):
            level = line.count("#")
            title = line.strip("# ").strip()
            anchor = create_anchor(title)
            if level <= toc_depth:
                toc.append(Li(
                    AX(title,
                       href=f'#{anchor}',
                       cls=f'toc-level-{level}',
                       style="text-decoration: none !important; color: #1976d2 !important;"),
                    style="list-style-type: none !important;"
                ))
    return Div(
        H2("Table of Contents"),
        Ul(*toc, style="list-style-type: none !important; padding-left: 0 !important;"),
        id="table_of_contents",
        style="width: 25%;",
        hx_swap_oob='true'
    )

def article(t):
    if t["article"] == "":
        return Div(H2("...Article under construction..."), id="article", style="width: 75%;", hx_swap_oob="true")
    content = [H1(t["name"])]
    # Construct the article with the structure of the markdown
    for line in t["article"].splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            level = line.count("#")
            title = line.strip("# ").strip()
            anchor = create_anchor(title)
            if level == 1:
                content.append(H2(title, id=anchor))
            elif level == 2:
                content.append(H3(title, id=anchor))
            elif level == 3:
                content.append(H4(title, id=anchor))
            else:
                content.append(P(title, id=anchor))
        else:
            # Let's keep the  "marked" to signal that this is markdown
            # In case there is a table it should still work
            content.append(P(line, cls="marked"))

    return Div(*content, id='article', style="width: 75%;", hx_swap_oob="true")

def brainstorming_process(hidden=True):
    bsp = Details(
        Summary("Brainstorming Process - open to read the conversations with experts that led to this article", role="button"),
        Div(id='personas'),
        Div(id='conversation'),
        id='brainstorming_process',
        hidden=hidden,
        hx_swap_oob="true"
    )
    return bsp

def personas(t, selected_persona=None):
    if t["conversation_log"] == {}:
        return Div("No conversation log available", id="personas", hx_swap_oob="true")

    conversations = DemoTextProcessingHelper.parse_conversation_history(t["conversation_log"])
    persona_names = [name for (name, _, _) in conversations]
    # Sort personas: "Basic Fact Writer" first, others alphabetically
    sorted_personas = sorted(persona_names, key=lambda x: (x.lower() != "basic fact writer", x))

    personas = [
        Card(
            H4(persona),
            cls=f"persona-card {'selected' if persona == selected_persona else ''}",
            hx_get=f"/conversation/{persona}?opportunity_id={t['id']}#conversation",
            hx_target="#conversation",
            hx_swap="outerHTML"
        ) for persona in sorted_personas
    ]

    return Div(
        *personas,
        id="personas",
        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;", 
        hx_swap_oob="true"
    )


# Define avatars for the robot
ROBOT_AVATAR_LEFT = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H16"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>'
ROBOT_AVATAR_RIGHT = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>'

# Define styles for chat bubbles
assistant_style = "background: #e3f2fd; border-radius: 15px; padding: 10px; margin: 5px 0; max-width: 70%;"
user_style = "background: #f5f5f5; border-radius: 15px; padding: 10px; margin: 5px 0; max-width: 70%;"
avatar_style = "width: 30px; height: 30px; margin: 0 10px;"

def conversation(t, persona):
    conversations = DemoTextProcessingHelper.parse_conversation_history(t["conversation_log"])
    (_, description, dialogue) = next((conv for conv in conversations if conv[0] == persona), None)
    d = []
    d.append(Card(
        Div(
            H5("Description of the Expert's role:", style="margin-bottom: 0.5rem;"),
            P(description, style="margin: 0;")
        ),
        style="background: #fff3e0; border-radius: 8px; padding: 1rem;"
    ))

    # Create conversation bubbles
    c = []
    for message in dialogue:
        is_assistant = message["role"].lower() == "assistant"
        content = message['content']
        avatar = ROBOT_AVATAR_RIGHT if is_assistant else ROBOT_AVATAR_LEFT
        bubble_style = assistant_style if is_assistant else user_style
        role_label = "Expert" if is_assistant else "Writer"
        message_div = Div(
            Div(
                *([Div(
                    Div(role_label, style="font-weight: bold; margin-bottom: 0.5rem;"),
                    P(content, style="margin: 0;"),
                    style=bubble_style
                  ), Img(src=avatar, style=avatar_style)] if is_assistant else
                  [Img(src=avatar, style=avatar_style), Div(
                    Div(role_label, style="font-weight: bold; margin-bottom: 0.5rem;"),
                    P(content, style="margin: 0;"),
                    style=bubble_style
                  )]),
                style=f"display: flex; align-items: center; justify-content: {'flex-end' if is_assistant else 'flex-start'};"
            ),
            style=f"display: flex; flex-direction: column; align-items: {'flex-end' if is_assistant else 'flex-start'}; margin: 1rem 0;"
        )

        c.append(message_div)

    d.append(Card(
        Div(H5("Conversation between the writer (left) and the expert (right)"), *c),
        style="border-radius: 12px; padding: 1rem; background: #f0f0f0;"
    ))

    return Div(*d, id="conversation", hx_swap_oob="true")

def format_citations_as_list(citations_dict):
    formatted_citations = []
    for key in sorted(citations_dict.keys()):
        citation = citations_dict[key]
        title = citation.get('title', 'No Title')
        url = citation.get('url', '#')
        snippets = citation.get('snippets', [])

        # Create an article element for each citation with modified layout
        citation_html = Details(
            # Reference number and title in the same header
            Summary(H4(
                f"[{key}] ",
                A(title, href=url, target="_blank"), Small(" (Expand for details)", style="font-size: 0.7em; color: var(--muted-color);")
            )),
            # Snippets displayed directly (not in Details/Summary)
            Ul(*[Li(snippet, style="margin-bottom: 0.5rem;") for snippet in snippets]) if snippets else "",
            cls="citation-card",
            style="margin-bottom: 1.5rem; padding: 1rem; background: var(--card-sectionning-background-color);"
        )

        formatted_citations.append(citation_html)

    # Container with citations
    return Div(
        H3("References"),
        *formatted_citations,
        id='citations',
        cls="citations-container",
        style="margin-top: 2rem;",
        hx_swap_oob="true"
    )

def citations(t):
    if t['citations'] == []:
        return Div("No citations available", id="citations", style="display: flex; flex-direction: row; gap: 10px; width: 100%; justify-content: space-between;", hx_swap_oob="true")
    citations_dict = t['citations']
    return format_citations_as_list(citations_dict)

def citations_list(hidden=True):
    cil = Details(
        Summary("Citations List - open to see and access the cited sources (you can also access them directly in the text below)", role="button"),
        Div(id='citations'),
        id='citations_list',
        hidden=hidden,
        hx_swap_oob="true"
    )
    return cil
#-------------------------------------------------------------------------------

def home():
    refresh_data()
    title = "Investment Opportunity Analyzer"
    cards = Div(*[opportunity_card(t) for t in table], id='opportunity_list', cls="grid")
    main_content = Div(
                Div(brainstorming_process(hidden=True)),
                # Div(citations_list(hidden=True)), # Hide the citations list for now (see search api license)
                Div(
                    Div(id='table_of_contents', style="width: 25%;"),
                    Div(id='article', style="width: 75%;"),
                    style="display: flex; flex-direction: row; gap: 20px; width: 100%;",
                    cls="content-wrapper"
                ),
            id="main_wrapper")

    content = new_opportunity(), Card(cards, main_content)
    return Titled(title, content)


@rt("/")
def get():
  return home()

@rt("/opportunity/{opportunity_id}")
def get(opportunity_id: str):
    # Find the opportunity
    opportunity = next((item for item in table if str(item['id']).lower() == str(opportunity_id).lower()), None)
    if opportunity is None:
        return "Opportunity not found"

    # Return both the article content and ALL cards (with appropriate selection states)
    return (
        table_of_contents(opportunity),
        article(opportunity),
        brainstorming_process(hidden=False),
        personas(opportunity),
        citations_list(hidden=False),
        citations(opportunity),
        # This is the list of all cards, with the correct selection state. It is neededd to adjust the selection state of the card that was clicked on
        *[opportunity_card(t, selected=(str(t['id']).lower() == str(opportunity_id).lower())) 
          for t in table]
    )

@rt("/conversation/{persona}")
def get(opportunity_id: str, persona: str):
    opportunity = next((item for item in table if str(item['id']).lower() == str(opportunity_id).lower()), None)
    if opportunity is None:
        return "Opportunity not found"
    if persona is None:
        return "Persona not found"
    return conversation(opportunity, persona), personas(opportunity, persona)

# Nice text to the user for the status
status_text = {
    'initiated': "Initiated: report generation is being initiated ...",
    'pre_writing': "Pre-writing: data is collected and outline is being generated. This takes up to two minutes ...",
    'final_writing': "Final writing: article is being written. This takes up to one minute ...",
    'complete': "Report generation done!"
}

@rt("/new_opportunity")
def get():
    return new_opportunity()

@rt("/")
def post(opportunity_name: str):
    if opportunity_name == "":
        pass_appropriateness_check = False
        return None, Card(
            Form(
                Div(
                    Div(f"You need to enter an investment opportunity name. Click on 'Try again' and enter a name.", style="flex: 1;"),
                    Button("Try again", hx_get="/new_opportunity"),
                    style="display: flex; justify-content: space-between; align-items: center;"
                ),
                hx_target="#new_opportunity"
            ),
            style=error_card_style,
            id="new_opportunity",
            hx_swap_oob="true"
        )
    opportunity_id = name_to_id(opportunity_name)

    # Check if opportunity already exists
    with get_db_connection() as db:
        opportunities = db.t.opportunities
        try:
            opportunities[opportunity_id]
            pass_appropriateness_check = False  # Opportunity already exists
            return None, Card(
            Form(
                Div(
                    Div(f"An investment memo for ", B(opportunity_name), " already exists. Click on 'Try again' and enter a different name.", style="flex: 1;"),
                    Button("Try again", hx_get="/new_opportunity"),
                    style="display: flex; justify-content: space-between; align-items: center;"
                ),
                hx_target="#new_opportunity"
            ),
            style=error_card_style,
            id="new_opportunity",
            hx_swap_oob="true"
        )
        except NotFoundError:
            pass_appropriateness_check = True   # New opportunity

    global opportunity_exists
    with get_db_connection() as db:
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        try:
            opportunities[opportunity_id]
            opportunity_exists = True
            status = opportunities[opportunity_id].status
            if status == 'complete':
                return None, Card(status_text[status], style=info_card_style, id="new_opportunity", hx_swap_oob="true")
        except NotFoundError:
            opportunity_exists = False
            db.t.opportunities.insert(Opportunities(id=opportunity_id, name=opportunity_name, status='initiated'))
    run_workflow(opportunity_name, opportunity_id)
    global preview_exists
    preview_exists = None
    return generation_preview(opportunity_id)

def generation_preview(opportunity_id):
    global preview_exists
    if get_status(opportunity_id) == 'complete':
        return (
            opportunity_generated,
            new_opportunity(),
            show_opportunity(opportunity_id)
        )
    else:
        status = get_status(opportunity_id)
        # If the opportunity exists already (from a previous run), we do not want to create a new entry in the DOM
        # First time the opportunity is being generated, it does not exist yet in the DOM
        if not preview_exists and not opportunity_exists:
            preview_exists = True
            return Card(Div("In progress...", aria_busy="true"), id=f"opportunity_{opportunity_id}", hx_vals=f'{{"opportunity_id": "{opportunity_id}"}}', hx_post=f"/generation_preview", hx_trigger='load once', hx_swap='afterbegin', hx_target="#opportunity_list"), Card(status_text[status], style=info_card_style, aria_busy="true", id="new_opportunity", hx_swap_oob="true")
        # If the opportunity is already in the DOM, we update it
        else:
            return Card(Div("In progress...", aria_busy="true"), id=f"opportunity_{opportunity_id}", hx_vals=f'{{"opportunity_id": "{opportunity_id}"}}', hx_post=f"/generation_preview", hx_trigger='every 5s', hx_swap='outerHTML', hx_target=f"#opportunity_{opportunity_id}", hx_swap_oob="true"), Card(status_text[status], style=info_card_style, aria_busy="true", id="new_opportunity", hx_swap_oob="true")

@rt("/generation_preview")
def post(opportunity_id: str):
    return generation_preview(opportunity_id)

@threaded
def run_workflow(opportunity_name, opportunity_id):
    # Initiate the workflow
    if get_status(opportunity_id) == 'initiated':
        demo_util.set_storm_runner()
        set_status(opportunity_id, 'pre_writing')
    # Pre-writing
    if get_status(opportunity_id) == 'pre_writing':
        runner = set_storm_runner()
        runner.run(
            opportunity=opportunity_name,
            opportunity_id=opportunity_id,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=False,
            do_polish_article=False,
            #callback_handler=BaseCallbackHandler() # TODO: add callback handler
        )
        set_status(opportunity_id, 'final_writing')
    # Final writing
    if get_status(opportunity_id) == 'final_writing':
        runner.run(
            opportunity=opportunity_name,
            opportunity_id=opportunity_id,
            do_research=False,
            do_generate_outline=False,
            do_generate_article=True,
            do_polish_article=True,
            remove_duplicate=False
            #callback_handler=BaseCallbackHandler() # TODO: add callback handler
        )
        runner.post_run(
            opportunity=opportunity_name,
            opportunity_id=opportunity_id,
        )
        refresh_data()
        opportunity = next((item for item in table if item['id'].lower() == opportunity_id.lower()), None)
        global opportunity_generated
        opportunity_generated = opportunity_card(opportunity)
        set_status(opportunity_id, 'complete')
        preview_exists = None

    return opportunity_generated

@threaded
@rt("/debug")
def post():
    runner = set_storm_runner()
    runner.run(
        opportunity="SpaceX a space company",
        opportunity_id="SpaceX_a_space_company",
        do_research=False,
        do_generate_outline=False,
        do_generate_article=True,
        do_polish_article=True,
        #callback_handler=BaseCallbackHandler() # TODO: add callback handler
    )
    return Div("Process run")

if __name__ == '__main__':
  # Alternative: you can use serve or uvicorn
  port = int(os.getenv("PORT", 8080))
  serve(host='0.0.0.0', port=port, reload=True)