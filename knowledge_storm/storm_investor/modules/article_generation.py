import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union
import threading
import multiprocessing
import sys
import time
import os

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing

class StormArticleGenerationModule(ArticleGenerationModule):
    """
    The interface for article generation stage. Given investment opportunity, collected information from
    knowledge curation stage, generated outline from outline generation stage,
    """

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)

    def generate_section(
        self, opportunity, section_name, information_table, section_outline, section_query
    ):
        collected_info: List[Information] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )
        output = self.section_gen(
            opportunity=opportunity,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": collected_info,
        }

    def generate_section_with_debug(self, opportunity, section_title, information_table, section_outline, section_query):
        thread_id = threading.current_thread().ident
        start = time.time()
        result = self.generate_section(opportunity, section_title, information_table, section_outline, section_query)
        duration = time.time() - start
        print(f"Section {section_title} in thread {thread_id} took {duration:.2f}s")
        return result

    def generate_article(
        self,
        opportunity: str,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Generate investment report for the investment opportunity based on the information table and article outline.

        Args:
            opportunity (str): The investment opportunity of the report.
            information_table (StormInformationTable): The information table containing the collected information.
            article_with_outline (StormArticle): The investment report with specified outline.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the article generation process. Defaults to None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(opportunity_name=opportunity)

        sections_to_write = article_with_outline.get_first_level_section_names()

        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(
                f"No outline for {opportunity}. Will directly search with the investment opportunity."
            )
            section_output_dict = self.generate_section(
                opportunity=opportunity,
                section_name=opportunity,
                information_table=information_table,
                section_outline="",
                section_query=[opportunity],
            )
            section_output_dict_collection = [section_output_dict]
        else:

            def print_system_info():
                print("\n=== System Information ===")
                print(f"Python version: {sys.version}")
                print(f"CPU count: {multiprocessing.cpu_count()}")
                print(f"Initial thread count: {threading.active_count()}")
                print(f"Main thread: {threading.current_thread().name}")
                print("\n=== Environment Variables ===")
                thread_related = {k:v for k,v in os.environ.items() if 'THREAD' in k.upper()}
                print(f"Thread-related env vars: {thread_related}")
                print("\n=== All Environment Variables ===")
                print(dict(os.environ))
                print("\n=== Active Threads ===")
                for thread in threading.enumerate():
                    print(f"Thread: {thread.name} ({thread.ident})")
                print("========================\n")

            # Add before your ThreadPoolExecutor code
            print_system_info()

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                print(f"\nExecutor created with max_workers={self.max_thread_num}")
                print(f"Thread count after executor creation: {threading.active_count()}")
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    # We don't want to write a separate introduction section.
                    if section_title.lower().strip() == "introduction":
                        continue
                        # We don't want to write a separate conclusion section.
                    if section_title.lower().strip().startswith(
                        "conclusion"
                    ) or section_title.lower().strip().startswith("summary"):
                        continue
                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True
                    )
                    section_outline = "\n".join(queries_with_hashtags)
                    future_to_sec_title[
                        executor.submit(
                            self.generate_section_with_debug,
                            opportunity,
                            section_title,
                            information_table,
                            section_outline,
                            section_query,
                        )
                    ] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                parent_section_name=opportunity,
                current_section_content=section_output_dict["section_content"],
                current_section_info_list=section_output_dict["collected_info"],
            )
        article.post_processing()
        return article


class ConvToSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(
        self, opportunity: str, outline: str, section: str, collected_info: List[Information]
    ):
        info = ""
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets)
            info += "\n\n"

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(opportunity=opportunity, info=info, section=section).output
            )

        return dspy.Prediction(section=section)


class WriteSection(dspy.Signature):
    """Write the section of an investment report based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    """

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    opportunity = dspy.InputField(prefix="The investment opportunity of the report: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str,
    )
