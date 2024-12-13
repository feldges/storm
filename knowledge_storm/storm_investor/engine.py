import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional

import dspy

from .modules.article_generation import StormArticleGenerationModule
from .modules.article_polish import StormArticlePolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.outline_generation import StormOutlineGenerationModule
from .modules.persona_generator import StormPersonaGenerator
from .modules.storm_dataclass import StormInformationTable, StormArticle, DialogueTurn
from ..interface import Engine, LMConfigs, Retriever
from ..lm import OpenAIModel, AzureOpenAIModel
from ..utils import makeStringRed, truncate_filename
from ..utils_db import dump_json, dump_url_to_info, dump_outline_to_file, dump_article_as_plain_text, dump_reference_to_db, prepare_calls_for_db

from fasthtml.common import database


class STORMWikiLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of STORM.

    Given that different parts in STORM framework have different complexity, we use different LLM configurations
    to achieve a balance between quality and efficiency. If no specific configuration is provided, we use the default
    setup in the paper.
    """

    def __init__(self):
        self.conv_simulator_lm = (
            None  # LLM used in conversation simulator except for question asking.
        )
        self.question_asker_lm = None  # LLM used in question asking.
        self.outline_gen_lm = None  # LLM used in outline generation.
        self.article_gen_lm = None  # LLM used in article generation.
        self.article_polish_lm = None  # LLM used in article polishing.

    def init_openai_model(
        self,
        openai_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        openai_type: Literal["openai", "azure"] = "openai",
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """Legacy: Corresponding to the original setup in the NAACL'24 paper."""
        azure_kwargs = {
            "api_key": azure_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": api_base,
            "api_version": api_version,
        }

        openai_kwargs = {
            "api_key": openai_api_key,
            "api_provider": "openai",
            "temperature": temperature,
            "top_p": top_p,
            "api_base": None,
        }
        if openai_type and openai_type == "openai":
            self.conv_simulator_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            # 1/12/2024: Update gpt-4 to gpt-4-1106-preview. (Currently keep the original setup when using azure.)
            self.outline_gen_lm = OpenAIModel(
                model="gpt-4-0125-preview", max_tokens=400, **openai_kwargs
            )
            self.article_gen_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=700, **openai_kwargs
            )
            self.article_polish_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=4000, **openai_kwargs
            )
        elif openai_type and openai_type == "azure":
            self.conv_simulator_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=500,
                **azure_kwargs,
                model_type="chat",
            )
            # use combination of openai and azure-openai as azure-openai does not support gpt-4 in standard deployment
            self.outline_gen_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=400, **azure_kwargs, model_type="chat"
            )
            self.article_gen_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=700,
                **azure_kwargs,
                model_type="chat",
            )
            self.article_polish_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=4000,
                **azure_kwargs,
                model_type="chat",
            )
        else:
            logging.warning(
                "No valid OpenAI API provider is provided. Cannot use default LLM configurations."
            )

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model


@dataclass
class STORMWikiRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""

    output_dir: str = field(
        metadata={"help": "Output directory for the results. Currently disabled."},
    )
    database_path: str = field(
        default="data/investor_reports.db",
        metadata={"help": "Path to the database."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={
            "help": "Maximum number of questions in conversational question asking."
        },
    )
    max_perspective: int = field(
        default=3,
        metadata={
            "help": "Maximum number of perspectives to consider in perspective-guided question asking."
        },
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximum number of threads to use. "
            "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."
        },
    )


class STORMWikiRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(
        self, args: STORMWikiRunnerArguments, lm_configs: STORMWikiLMConfigs, rm
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs
        self.db = database(self.args.database_path)

        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)
        storm_persona_generator = StormPersonaGenerator(
            self.lm_configs.question_asker_lm
        )
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm,
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(
        self,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None,
    ) -> StormInformationTable:

        information_table, conversation_log = (
            self.storm_knowledge_curation_module.research(
                opportunity=self.opportunity,
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler,
                max_perspective=self.args.max_perspective,
                disable_perspective=False,
                return_conversation_log=True,
            )
        )
        # -------------------------------------------------------------------------------
        # Use DB instead of local file system

        db = self.db
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        oppo = Opportunities(id=self.opportunity_id, conversation_log=dump_json(conversation_log), raw_search_results=dump_url_to_info(information_table))
        db.t.opportunities.update(oppo)
        # -------------------------------------------------------------------------------

        return information_table

    def run_outline_generation_module(
        self,
        information_table: StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:

        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            opportunity=self.opportunity,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler,
        )

        # -------------------------------------------------------------------------------
        # Use DB instead of local file system

        db = self.db
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        oppo = Opportunities(id=self.opportunity_id, storm_gen_outline=dump_outline_to_file(outline), direct_gen_outline=dump_outline_to_file(draft_outline))
        db.t.opportunities.update(oppo)
        # -------------------------------------------------------------------------------

        return outline

    def run_article_generation_module(
        self,
        outline: StormArticle,
        information_table: StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:

        draft_article = self.storm_article_generation.generate_article(
            opportunity=self.opportunity,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler,
        )

        # -------------------------------------------------------------------------------
        # Use DB instead of local file system

        db = self.db
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        oppo = Opportunities(id=self.opportunity_id, storm_gen_article=dump_article_as_plain_text(draft_article), url_to_info=dump_reference_to_db(draft_article))
        db.t.opportunities.update(oppo)
        # -------------------------------------------------------------------------------

        return draft_article

    def run_article_polishing_module(
        self, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:

        polished_article = self.storm_article_polishing_module.polish_article(
            opportunity=self.opportunity,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate,
        )

        # -------------------------------------------------------------------------------
        # Use DB instead of local file system

        db = self.db
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        oppo = Opportunities(id=self.opportunity_id, storm_gen_article_polished=dump_article_as_plain_text(polished_article))
        db.t.opportunities.update(oppo)
        # -------------------------------------------------------------------------------

        return polished_article

    def post_run(self, opportunity, opportunity_id):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        config_log = self.lm_configs.log()
        self.opportunity = opportunity
        self.opportunity_id = opportunity_id


        llm_call_history = self.lm_configs.collect_and_reset_lm_history()

        # -------------------------------------------------------------------------------
        # Use DB instead of local file system

        db = self.db
        opportunities = db.t.opportunities
        Opportunities = opportunities.dataclass()
        oppo = Opportunities(id=self.opportunity_id, run_config=dump_json(config_log), llm_call_history=prepare_calls_for_db(llm_call_history))
        db.t.opportunities.update(oppo)
        # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Load conversation log table from database
    def from_conversation_log_db(self, opportunity_id):
        db = self.db
        opportunities = db.t.opportunities
        # Next line is needed to have oppo as a dataclass instead of a dict
        Opportunities = opportunities.dataclass()
        oppo = opportunities[opportunity_id]
        conversation_log_data = json.loads(oppo.conversation_log)
        conversations = []
        for item in conversation_log_data:
            dialogue_turns = [DialogueTurn(**turn) for turn in item["dlg_turns"]]
            persona = item["perspective"]
            conversations.append((persona, dialogue_turns))
        return StormInformationTable(conversations)

    def _load_information_table_from_db(self, opportunity_id):
        return self.from_conversation_log_db(opportunity_id)
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Load outline from database
    def from_outline_db(self, opportunity: str, opportunity_id: str):
        """
        Create StormArticle class instance from outline file.
        """
        db = self.db
        opportunities = db.t.opportunities
        # Next line is needed to have oppo as a dataclass instead of a dict
        Opportunities = opportunities.dataclass()
        oppo = opportunities[opportunity_id]
        storm_gen_outline = oppo.storm_gen_outline
        outline_str = storm_gen_outline
        return StormArticle.from_outline_str(opportunity=opportunity, outline_str=outline_str)

    def _load_outline_from_db(self, opportunity, opportunity_id):
        return self.from_outline_db(opportunity=opportunity, opportunity_id=opportunity_id)

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Load draft article from database

    def _load_draft_article_from_db(self, opportunity_id):
        db = self.db
        opportunities = db.t.opportunities
        # Next line is needed to have oppo as a dataclass instead of a dict
        Opportunities = opportunities.dataclass()
        oppo = opportunities[opportunity_id]
        opportunity_name = oppo.name
        article_text = oppo.storm_gen_article
        references = json.loads(oppo.url_to_info)
        return StormArticle.from_string(opportunity_name=opportunity_name, article_text=article_text, references=references)

    # -------------------------------------------------------------------------------

    def run(
        self,
        opportunity: str,
        opportunity_id: str,
        ground_truth_url: str = "",
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = False,
        callback_handler: BaseCallbackHandler = BaseCallbackHandler(),
    ):
        """
        Run the STORM pipeline.

        Args:
            opportunity: The investment opportunity to research.
            ground_truth_url: A ground truth URL including a curated article about the investment opportunity. The URL will be excluded.
            do_research: If True, research the investment opportunity through information-seeking conversation;
             if False, expect conversation_log.json and raw_search_results.json to exist in the output directory.
            do_generate_outline: If True, generate an outline for the investment opportunity;
             if False, expect storm_gen_outline.txt to exist in the output directory.
            do_generate_article: If True, generate a curated article for the investment opportunity;
             if False, expect storm_gen_article.txt to exist in the output directory.
            do_polish_article: If True, polish the article by adding a summarization section and (optionally) removing
             duplicated content.
            remove_duplicate: If True, remove duplicated content.
            callback_handler: A callback handler to handle the intermediate results.
        """
        assert (
            do_research
            or do_generate_outline
            or do_generate_article
            or do_polish_article
        ), makeStringRed(
            "No action is specified. Please set at least one of --do-research, --do-generate-outline, --do-generate-article, --do-polish-article"
        )

        self.opportunity = opportunity
        self.opportunity_id = opportunity_id
        self.db = database(self.args.database_path)
        self.article_dir_name = truncate_filename(
            opportunity.replace(" ", "_").replace("/", "_")
        )
        self.article_output_dir = os.path.join(
            self.args.output_dir, self.article_dir_name
        )

        # research module
        information_table: StormInformationTable = None
        if do_research:
            information_table = self.run_knowledge_curation_module(
                ground_truth_url=ground_truth_url, callback_handler=callback_handler
            )
        # outline generation module
        outline: StormArticle = None
        if do_generate_outline:
            # load information table if it's not initialized
            if information_table is None:
                information_table = self._load_information_table_from_db(opportunity_id)
            outline = self.run_outline_generation_module(
                information_table=information_table, callback_handler=callback_handler
            )

        # article generation module
        draft_article: StormArticle = None
        if do_generate_article:
            if information_table is None:
                information_table = self._load_information_table_from_db(opportunity_id)
            if outline is None:
                outline = self._load_outline_from_db(opportunity, opportunity_id)
            draft_article = self.run_article_generation_module(
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler,
            )

        # article polishing module
        if do_polish_article:
            if draft_article is None:
                draft_article = self._load_draft_article_from_db(opportunity_id=opportunity_id)
            self.run_article_polishing_module(
                draft_article=draft_article, remove_duplicate=remove_duplicate
            )
