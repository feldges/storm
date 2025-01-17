from fasthtml.common import *
from fasthtml.oauth import GoogleAppClient, OAuth
from fasthtml.common import FastHTML, RedirectResponse
from fastcore.parallel import threaded
import sys
import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import unicodedata
from copy import deepcopy
import sqlite3

database_path = os.getenv("DB_FILE", "data/investor_reports.db")

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo_light'))

import demo_util
from demo_util import DemoTextProcessingHelper
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import YouRM, BraveRM, BingSearch
# users and opportunities are tables in the database; Users and Opportunities are datamodels
from knowledge_storm.utils_db import db, users, opportunities, Users, Opportunities

load_dotenv()

#-------------------------------------------------------------------------------
# ADDED FROM BOILERPLATE
# ------------------------------------------------------------
# Configuration
try:
    with open('assets/legal/terms_of_service.md', 'r') as file:
        TERMS_OF_SERVICE = file.read()
except FileNotFoundError:
    TERMS_OF_SERVICE = "Terms of service file not found."

try:
    with open('assets/legal/privacy_policy.md', 'r') as file:
        PRIVACY_POLICY = file.read()
except FileNotFoundError:
    PRIVACY_POLICY = "Privacy policy file not found."

application_name = "Investment Reports App"
application_description = Div("""Generate teasers for any company, based on information collected on the internet.
                            This application is based on """, A("STORM", href = "https://storm.genie.stanford.edu/", target="_blank"), """, a framework developed by the Stanford University to generate Wiki pages.
                            It is experimental and may not work as expected.""")
application_description_txt = """Generate teasers for any company, based on information collected on the internet. This application is based on "STORM", a framework developed by the Stanford University to generate Wiki pages. It is experimental and may not work as expected."""

socials = Socials(title=application_name, description=application_description_txt, site_name='storm.aipe.tech', image='https://storm.aipe.tech/assets/images/investment_analyzer_screen.png', url='https://storm.aipe.tech')

boilerplate_styles = """
.dropdown {
    position: relative;
    cursor: pointer;
    width: 24px;
    height: 24px;
}

#menu-container {
    position: absolute;
    right: 0;
    top: 100%;
}

.dropdown-menu {
    position: relative;  /* Changed from absolute since it's inside container */
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    min-width: 200px;
    padding-top: 8px;
    margin-top: 5px;  /* Add space between initials and menu */
    z-index: 1001;
}

.menu-underlay {
    position: fixed;  /* Changed to fixed */
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000;
}

.dropdown-menu a {
    display: block;
    padding: 8px 16px;
    text-decoration: none;
    color: black;
}

.dropdown-menu a:hover {
    background-color: #f5f5f5;
}
"""
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

/* Add these new styles */
@media (max-width: 768px) {
    /* Adjust header title font size for smaller screens */
    .header-title {
        font-size: 1.2rem !important;  /* Smaller font size on mobile */
    }
}

@media (max-width: 480px) {
    /* Even smaller for very small screens */
    .header-title {
        font-size: 1rem !important;
    }
}
"""

headers = (MarkdownJS(), Style(boilerplate_styles + app_styles), socials, picolink, Favicon('assets/images/favicon.ico', 'assets/images/favicon.ico'))
app = FastHTML(title=application_name, hdrs=headers)

# Add a before to the app to limit access to the database
def restrict_db_access(req, session):
    auth = req.scope['auth']
    opportunities.xtra(user_id=auth)
    users.xtra(id=auth)

# For each thread, we need to enforce again the restriction to the database
def set_thread_access(auth):
    opportunities.xtra(user_id=auth)
    users.xtra(id=auth)

# Add a before to the app to check if the user has agreed to the terms of service
def check_terms_agreed(req, session):
    auth = session.get('auth')
    if not users[auth].terms_agreed:
        return RedirectResponse('/terms_of_service', status_code=303)
    return None
# ------------------------------------------------------------

# ------------------------------------------------------------
# Authentication via Google OAuth2
AUTH_CLIENT_ID = os.getenv("AUTH_CLIENT_ID")
AUTH_CLIENT_SECRET = os.getenv("AUTH_CLIENT_SECRET")

client = GoogleAppClient(
        AUTH_CLIENT_ID,
        AUTH_CLIENT_SECRET
        )

class Auth(OAuth):
    def get_auth(self, info, ident, session, state):
        email = info.email or ''
        if info.email_verified:
            try:
                u = users[ident]
            except NotFoundError:
                try:
                    u = users.insert(Users(id=ident, email=info.email, first_name=info.given_name, last_name=info.family_name))
                except sqlite3.IntegrityError:
                    u = users[ident]
            return RedirectResponse('/', status_code=303)
        return RedirectResponse(self.login_path, status_code=303)

oauth = Auth(app, client, skip=[r'/login', r'/redirect', r'/error', r'/logout', r'/health', r'/privacy_policy', r'/terms_of_service', r'/favicon\.ico', r'/static/.*', r'/assets/.*', r'.*\.css'])
# The db access restriction has to be added to the before list AFTER the OAuth authentication
app.before.append(Beforeware(restrict_db_access, skip=oauth.skip))
skip_list_check_terms = deepcopy(oauth.skip)
skip_list_check_terms.append(r'/agree_terms')
app.before.append(Beforeware(check_terms_agreed, skip=skip_list_check_terms))

# This is needed to serve the favicon.ico file (and potentially other static files)
# If you use fast_app instead of FastHTML, this is not needed as this has been integrated in fast_app already
@app.get("/{fname:path}.{ext:static}")
def get(fname:str, ext:str): return FileResponse(f'{fname}.{ext}')
# ------------------------------------------------------------

# ------------------------------------------------------------
# FastHTML Application starts here

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Handler for toggle_menu endpoint
@app.get('/toggle_menu')
def toggle_menu():
    global show_menu
    show_menu = not show_menu
    if show_menu:
        return Div(
            # First div is the underlay for catching clicks
            Div(
                cls='menu-underlay',
                hx_get='/hide_menu',
                hx_target='#menu-container',
                hx_swap='outerHTML'
            ),
            # Second div is the menu content
            Div(
                A('Terms of Service', href='/terms_of_service'),
                A('Privacy Policy', href='/privacy_policy'),
                A('Log out', href='/logout'),
                cls='dropdown-menu'
            ),
            id='menu-container'
        )
    else:
        return Div(id='menu-container')

@app.get('/hide_menu')
def hide_menu():
    global show_menu
    show_menu = not show_menu
    return Div(id='menu-container')

@app.get('/login')
def login(req):
    return Title("Login"), login_header(), Div(
            H1(application_name),
            Div(application_description, style='margin: 0 auto 10px auto; width: 100%; max-width: 600px; text-align: justify; padding: 0 20px;'),
            Div(
                A(
                    Img(src='/assets/images/google-logo.svg',
                        style='''
                            cursor: pointer;
                            transition: all 0.2s ease;
                            border: 1px solid #ddd;
                            border-radius: 4px;
                        ''',
                        onmouseover="this.style.transform='translateY(-3px) scale(1.05)'; this.style.filter='drop-shadow(0 4px 6px rgba(0,0,0,0.1))'",
                        onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.filter='none'"
                    ),
                    href=oauth.login_link(req)
                ),
                style='display: flex; justify-content: center; margin: 20px 0;'
            ),
            Div(
                Div(
                    A('Terms of Service', href='/terms_of_service', target='_blank'),
                    style='flex: 1; display: flex; justify-content: center; font-size: 0.8rem;'
                ),
                Div(
                    A('Privacy Policy', href='/privacy_policy', target='_blank'),
                    style='flex: 1; display: flex; justify-content: center; font-size: 0.8rem;'
                ),
                style='margin-top: 20px; display: flex; width: 100%; max-width: 600px;'
            ),
            style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 50vh;'
            )

@app.get('/terms_of_service')
def terms_of_service(req, session):
    if 'show_menu' in globals():
        show_menu = False
        # trigger htmx to hide the menu
    auth = session.get('auth')
    preamble = Div("")
    button = Div("")
    if auth:
        if users[auth].terms_agreed:
            preamble = Div(
                "You have already agreed to the terms of service. If you want to remove your approval, click on the button below.",
                style="margin-bottom: 20px;"
            )
            button = Div(
                "By clicking on 'Remove approval', I confirm that I want to remove my approval of the terms of service. As a consequence, I will not be able to use this application anymore.",
                A('Remove approval', href='/agree_terms?approve=False', role='button', style='margin-left: 10px;'),
                style='display: flex; flex-direction: row; align-items: center; justify-content: space-between; margin-top: 20px; width: 100%;'
            )
        else:
            preamble = Div(
                "You need to agree to the terms of service before you can use this application. Please read the terms of service and click the button to agree.",
                style="margin-bottom: 20px;"
            )
            button = Div(
                "By clicking on 'Agree', I confirm that I have read and agree with the terms of service.",
                A('Agree', href='/agree_terms', role='button', style='margin-left: 10px;'),
                style='display: flex; flex-direction: row; align-items: center; justify-content: space-between; margin-top: 20px; width: 100%;'
            )

    return Title("Terms of Service"), Div(
        Div(
            preamble,
            Div(
                TERMS_OF_SERVICE,
                cls='marked',
                style='border: 1px solid #ccc; border-radius: 8px; padding: 20px; max-width: 800px; font-size: 0.9em;'
            ),
            button,
            style='max-width: 800px;'
        ),
        style='display: flex; justify-content: center; align-items: start; min-height: 100vh; padding: 40px 20px;'
    )

@app.get('/privacy_policy')
def privacy_policy():
        return Title("Privacy Policy"), Div(
            Div(
                PRIVACY_POLICY,
                cls='marked',
                style='border: 1px solid #ccc; border-radius: 8px; padding: 20px; max-width: 800px; font-size: 0.9em;'
            ),
            style='display: flex; justify-content: center; align-items: start; min-height: 100vh; padding: 40px 20px;'
        )

@app.get('/agree_terms')
def agree_terms(req, session, approve: bool = None):
    auth = session.get('auth')
    approve = True if approve is None else approve
    users.update(id=auth, terms_agreed=approve, terms_agreed_or_rejected_date=datetime.now(), terms_agreed_date_first_time=datetime.now() if users[auth].terms_agreed_date_first_time is None else users[auth].terms_agreed_date_first_time)
    if approve:
        return RedirectResponse('/', status_code=303)
    else:
        return RedirectResponse('/logout', status_code=303)

def login_header():
   return Div(
       Div(
           # Logo on the far left
           A(
               Img(
                   src='/assets/images/aipe_logo_white.svg',
                   alt='AIPE Logo',
                   style='height: 24px; width: auto;'
               ),
               href='/',
               style='text-decoration: none; margin-left: 20px;'
           ),
           style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; width: 100%;'
       ),
       style='border-bottom: 1px solid #0055a4; background: #0055a4; width: 100%;'
   )

# Create the header for the application
show_menu = False
def app_header(user):
    initials = f"{user.first_name[0]}{user.last_name[0]}"
    return Div(
        Div(
            # Logo on the far left
            A(
                Img(
                    src='/assets/images/aipe_logo_white.svg',
                    alt='AIPE Logo',
                    style='height: 24px; width: auto;'
                ),
                href='/',
                style='text-decoration: none; margin-left: 20px;'
            ),
            # Right side container with support, feedback links and profile menu
            Div(
                # Support and Feedback links
                A("Support",
                  href="mailto:support@aipetech.com?subject=Storm%20App%20Support%20Request&body=Hello%2C%0A%0AI%20need%20assistance%20with%20the%20Storm%20application.%20Here%20are%20the%20details%20of%20my%20issue%3A%0A%0A-%20What%20happened%3A%0A-%20What%20I%20was%20trying%20to%20do%3A%0A-%20Steps%20to%20reproduce%20the%20issue%3A%0A%0ABest%20regards",
                  style="color: white; margin-right: 15px; text-decoration: none; font-size: 0.9rem;"),
                A("Feedback",
                  href="mailto:feedback@aipetech.com?subject=Storm%20App%20Feedback&body=Hello%2C%0A%0AI%20would%20like%20to%20provide%20the%20following%20feedback%20about%20the%20Storm%20application%20and%20the%20generated%20content%3A%0A%0A%0ABest%20regards",
                  style="color: white; margin-right: 15px; text-decoration: none; font-size: 0.9rem;"),
                # Profile menu
                Div(
                    # Initials circle with toggle behavior
                    Div(
                        initials,
                        style='width: 24px; height: 24px; background: white; color: #0055a4; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; position: relative;',
                        hx_get='/toggle_menu',
                        hx_target='#menu-container',
                        hx_swap='outerHTML transition:true'
                    ),
                    # Menu container - keep it simple
                    Div(id='menu-container'),
                    cls='dropdown',
                    style='margin-right: 20px;'
                ),
                style='display: flex; align-items: center;'
            ),
            style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; width: 100%;'
        ),
        style='border-bottom: 1px solid #0055a4; background: #0055a4; width: 100%;'
    )

# ------------------------------------------------------------
# This is where the real application starts
#-------------------------------------------------------------------------------

openai_api_key = os.getenv("OPENAI_API_KEY")
ydc_api_key = os.getenv("YDC_API_KEY")
bing_search_api_key = os.getenv("BING_SEARCH_API_KEY")

info_card_style = "word-wrap: break-word; overflow-wrap: break-word; white-space: normal; color: #1976d2; background-color: #e3f2fd; border-radius: 4px"
error_card_style = "word-wrap: break-word; overflow-wrap: break-word; white-space: normal; color: #d32f2f; background-color: #ffebee; border-radius: 4px"

# Read data from the source directory and return a dictionary under table
# Deprecated - will be removed but the local_dir is still used in the code (while inactive),
# So it was decided to keep it for now.
local_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")

#-------------------------------------------------------------------------------
# Initiate the runner
def set_storm_runner(auth):
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
        max_perspective=3, # 3 is the default
        search_top_k=3, # 3 is the default
        retrieve_top_k=5, # 5 is the default
        user_id=auth,
        database_path=database_path
    )

    # rm = YouRM(ydc_api_key=ydc_api_key, k=engine_args.search_top_k)
    # rm = BraveRM(k=engine_args.search_top_k)
    rm = BingSearch(bing_search_api_key=bing_search_api_key, k=engine_args.search_top_k)

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

def get_table():
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
    in_progress = opportunities(where="status!='complete'")
    if in_progress and len(in_progress) > 0:
        oppo_in_progress = in_progress[0]
        return oppo_in_progress.id, oppo_in_progress.name, oppo_in_progress.status
    return None, None, "complete"

def refresh_data():
    global data, table
    data, table = get_table()
    return data, table

def get_status(opportunity_id, auth):
    opportunities.xtra(user_id=auth)
    opportunity = opportunities[opportunity_id, auth]
    return opportunity.status

def set_status(opportunity_id, auth, status):
    opportunities.xtra(user_id=auth)
    oppo = Opportunities(id=opportunity_id, user_id=auth, status=status)
    opportunities.update(oppo)
    return status

def get_number_of_opportunities():
    return len(opportunities())

def get_max_number_of_opportunities():
    #user = users()[0]
    #return user.max_number_of_opportunities
    return 5

data, table = refresh_data()

#-------------------------------------------------------------------------------
# Generate various HTML elements
#-------------------------------------------------------------------------------
def webpage_header():
    return Div(
        Div(
            # Logo on the far left
            A(
                Img(
                    src='/assets/images/aipe_logo_white.svg',
                    alt='AIPE Logo',
                    style='height: 24px; width: auto;'
                ),
                href='/',
                style='text-decoration: none; margin-left: 20px;'
            ),
            H2(
                "Investment Analyzer",
                style='color: white; margin: 0; position: absolute; left: 50%; transform: translateX(-50%);',
                cls='header-title'  # Added class for responsive styling
            ),
            style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; width: 100%; position: relative;'
        ),
        style='border-bottom: 1px solid #0055a4; background: #0055a4; width: 100%;'
    )

oppo_id, oppo_name, oppo_status = None, None, "complete"
def new_opportunity():
    global oppo_id, oppo_name, oppo_status
    previous_oppo_id = oppo_id
    oppo_id, oppo_name, oppo_status = get_overall_status()
    if oppo_id is None:
        if previous_oppo_id is not None:
            return new_opportunity(), show_opportunity(previous_oppo_id)
        else:
            if get_number_of_opportunities() >= get_max_number_of_opportunities():
                return limit_reached()
            else:
                return (
                Div(opportunity_counter(),
                Card(
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
                )))
    else:
        return Card(
            Div(
                Div(f"Working on the opportunity ", B(oppo_name), f" in stage {status_text[oppo_status]}."),
                Div(status_description[oppo_status]),
                Div("You have to wait for it to finish before you can start a new one.")
            ),
            aria_busy="true",
            style=info_card_style,
            hx_get="/new_opportunity",
            hx_trigger="every 5s",
            hx_swap_oob="true",
            id="new_opportunity"
        )

def opportunity_counter():
    """
    Display the number of opportunities used and the percentage of opportunities used.
    This is used to display the limits the usage limits the users have.
    """
    nb_oppo = get_number_of_opportunities()
    max_nb_oppo = get_max_number_of_opportunities()
    nb_oppo_left = max_nb_oppo - nb_oppo
    percentage_counter = 0
    if max_nb_oppo != 0:
        percentage_counter = nb_oppo / max_nb_oppo
    else:
        percentage_counter = 1

    # Define color styles
    color_style = ""
    if percentage_counter <= 0.5:
        color_style = "color: #4caf50;"  # green
    elif percentage_counter <= 0.75:
        color_style = "color: #ff9800;"  # orange
    else:
        color_style = "color: #f44336;"  # red

    text_counter = ""
    if nb_oppo_left <= 0:
        text_counter = f"You have reached the limit. Please contact us to increase your limit."
    elif nb_oppo == 0:
        text_counter = f"You have {max_nb_oppo} trials"
    elif nb_oppo_left == 1:
        text_counter = f"You have only one trial left"
    else:
        text_counter = f"{nb_oppo} / {max_nb_oppo} trials used"

    return Div(text_counter,
               style=f"{color_style} font-size: 0.8rem; text-align: right;",
               id="opportunity_counter",
               hx_swap_oob="true")

def limit_reached():
    return Div(Div(opportunity_counter()),
            Card(
            Form(
                Div(f"You have reached the maximum number of opportunities. Please contact us to increase your limit.", style="flex: 1;"),
                Button("Contact Us", 
                       onclick="window.location.href='mailto:sales@aipetech.com?subject=Request%20for%20Access%20to%20Investment%20Analyzer&body=Hello,%0D%0A%0D%0AI%20am%20interested%20in%20getting%20access%20to%20the%20Investment%20Analyzer%20tool.%20I%20have%20reached%20the%20limit%20of%20my%20trial%20usage%20and%20would%20like%20to%20discuss%20pricing%20options%20and%20features%20available.%0D%0A%0D%0APlease%20contact%20me%20regarding%20available%20plans%20and%20next%20steps.%0D%0A%0D%0AThank%20you'"),
                style="display: flex; justify-content: space-between; align-items: center;"
            ),
            style=error_card_style,
            id="new_opportunity",
            hx_swap_oob="true"
        ))

def opportunity_card(t, selected=False):
    return Card(
        H4(t["name"]),
        P(t["article"][20:] if t["article"][:20].lower() == '\n# executive summary' else t["article"]),
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

    # Add disclaimer at the top
    disclaimer = Div(
        "Generated with Large Language Models (LLMs) using information from the Internet. LLMs make mistakes. Internet includes unverified information. Please double-check content.",
        style="border: 1px solid #ff9800; border-radius: 4px; padding: 5px; margin-bottom: 20px; background-color: #fff3e0; font-size: 0.8em;"
    )

    content = [disclaimer, H1(t["name"])]
    # Rest of the function remains the same
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

def home(auth):
    refresh_data()
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

    page_header = app_header(users[auth])
    content = Div(new_opportunity(), Card(cards, main_content), style="max-width: 1200px; margin: 2rem auto 0; padding: 0 20px;")
    return page_header, content


@app.get("/")
def get(auth):
    return home(auth)

@app.get("/opportunity/{opportunity_id}")
def get(opportunity_id: str):
    return show_opportunity(opportunity_id)

def show_opportunity(opportunity_id: str):
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
        # Wrap the cards in a Div with the proper ID and class
        Div(*[opportunity_card(t, selected=(str(t['id']).lower() == str(opportunity_id).lower()))
          for t in table],
          id='opportunity_list',
          cls="grid",
          hx_swap_oob="true"
        )
    )

@app.get("/conversation/{persona}")
def get(opportunity_id: str, persona: str):
    opportunity = next((item for item in table if str(item['id']).lower() == str(opportunity_id).lower()), None)
    if opportunity is None:
        return "Opportunity not found"
    if persona is None:
        return "Persona not found"
    return conversation(opportunity, persona), personas(opportunity, persona)

# Nice text to the user for the status
status_text = {
    'initiated': "'Initiated'",
    'pre_writing': "'Pre-writing'",
    'final_writing': "'Final writing'",
    'complete': "Report generation done!"
}
status_description = {
    'initiated': "If this status remains for several minutes, please contact us.",
    'pre_writing': "A writer is currently talking to four virtual experts, who are collecting data from 100+ web sites found through 30+ search queries to Bing Search, to provide grounded answers. This takes up to two minutes.",
    'final_writing': "Final writing. Several sections are being written in parallel. This takes up to one minute.",
    'complete': "Report generation done!"
}

@app.get("/new_opportunity")
def get():
    return new_opportunity()

@app.post("/")
def post(opportunity_name: str, auth):
    if get_number_of_opportunities() >= get_max_number_of_opportunities():
        return None, opportunity_counter(), limit_reached()
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

    # Opportunity does not exist yet, so we create a new entry in the database
    try:
        opportunities.insert(Opportunities(id=opportunity_id, name=opportunity_name, user_id=auth, status='initiated'))
    except sqlite3.IntegrityError:
        pass
    # Verify creation with retries
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            opportunity = opportunities[opportunity_id, auth]
            break
        except NotFoundError:
            if attempt < max_attempts - 1:
                print("Attempt", attempt+1)
                time.sleep(0.5)
            else:
                return None, Card(
                    Form(
                        Div(
                            Div(f"Failed to create the opportunity ", B(opportunity_name), " in the database. We are sorry for that. Please try again", style="flex: 1;"),
                            Button("Try again", hx_get="/new_opportunity"),
                            style="display: flex; justify-content: space-between; align-items: center;"
                        ),
                        hx_target="#new_opportunity"
                    ),
                    style=error_card_style,
                    id="new_opportunity",
                    hx_swap_oob="true"
                )

    run_workflow(opportunity_name, opportunity_id, auth)

    return generation_preview(opportunity_id, auth)

def generation_preview(opportunity_id, auth):

    if get_status(opportunity_id, auth) == 'complete':
        return (
            opportunity_generated,
            opportunity_counter(),
            new_opportunity(),
            show_opportunity(opportunity_id)
        )
    else:
        status = get_status(opportunity_id, auth)
        return None, new_opportunity()

@app.post("/generation_preview")
def post(opportunity_id: str, auth):
    return generation_preview(opportunity_id, auth)

@threaded
def run_workflow(opportunity_name, opportunity_id, auth):
    set_thread_access(auth)

    # Initiate the workflow
    if get_status(opportunity_id, auth) == 'initiated':
        set_storm_runner(auth)
        set_status(opportunity_id, auth, 'pre_writing')
    # Pre-writing
    if get_status(opportunity_id, auth) == 'pre_writing':
        runner = set_storm_runner(auth)
        runner.run(
            opportunity=opportunity_name,
            opportunity_id=opportunity_id,
            user_id=auth,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=False,
            do_polish_article=False,
            #callback_handler=BaseCallbackHandler() # TODO: add callback handler
        )
        set_status(opportunity_id, auth, 'final_writing')
    # Final writing
    if get_status(opportunity_id, auth) == 'final_writing':
        runner.run(
            opportunity=opportunity_name,
            opportunity_id=opportunity_id,
            user_id=auth,
            do_research=False,
            do_generate_outline=False,
            do_generate_article=True,
            do_polish_article=False, # Removed the article polishing step because an Executive Summary is already (almost always) included in the article
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
        set_status(opportunity_id, auth, 'complete')

    return opportunity_generated

@threaded
@app.post("/debug")
def post():
    runner = set_storm_runner(auth)
    runner.run(
        opportunity="SpaceX a space company",
        opportunity_id="SpaceX_a_space_company",
        user_id=auth,
        do_research=False,
        do_generate_outline=False,
        do_generate_article=True,
        do_polish_article=True,
        #callback_handler=BaseCallbackHandler() # TODO: add callback handler
    )
    return Div("Process run")

if __name__ == '__main__':
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    reload = os.getenv('RELOAD', 'true').lower() == 'true'
    serve(host=host, port=port, reload=reload)