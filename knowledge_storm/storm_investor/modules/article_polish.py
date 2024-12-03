import copy
from typing import Union

import dspy

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing


class StormArticlePolishingModule(ArticlePolishingModule):
    """
    The interface for investment report generation stage. Given investment opportunity, collected information from
    knowledge curation stage, generated outline from outline generation stage.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm, polish_engine=self.article_polish_lm
        )

    def polish_article(
        self, opportunity: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Polish article.

        Args:
            opportunity (str): The investment opportunity of the report.
            draft_article (StormArticle): The draft report.
            remove_duplicate (bool): Whether to use one additional LM call to remove duplicates from the report.
        """

        article_text = draft_article.to_string()
        polish_result = self.polish_page(
            opportunity=opportunity, draft_page=article_text, polish_whole_page=remove_duplicate
        )
        lead_section = f"# summary\n{polish_result.lead_section}"
        polished_article = "\n\n".join([lead_section, polish_result.page])
        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article
        )
        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        polished_article.post_processing()
        return polished_article


class WriteLeadSection(dspy.Signature):
    """Write a lead section for the given investment report with the following guidelines:
    1. The lead should stand on its own as a concise overview of the report's investment opportunity. It should identify the investment opportunity, establish context, explain why the opportunity is notable, and summarize the most important points, including any prominent controversies.
    2. The lead section should be concise and contain no more than four well-composed paragraphs.
    3. The lead section should be carefully sourced as appropriate. Add inline citations (e.g., "Washington, D.C., is the capital of the United States.[1][3].") where necessary.
    """

    opportunity = dspy.InputField(prefix="The investment opportunity of the report: ", format=str)
    draft_page = dspy.InputField(prefix="The draft report:\n", format=str)
    lead_section = dspy.OutputField(prefix="Write the lead section:\n", format=str)


class PolishPage(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the report and deleting them to make sure there is no repetition in the report. You won't delete any non-repeated part in the report. You will keep the inline citations and report structure (indicated by "#", "##", etc.) appropriately. Do your job for the following report."""

    draft_page = dspy.InputField(prefix="The draft report:\n", format=str)
    page = dspy.OutputField(prefix="Your revised report:\n", format=str)


class PolishPageModule(dspy.Module):
    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)

    def forward(self, opportunity: str, draft_page: str, polish_whole_page: bool = True):
        # NOTE: Change show_guidelines to false to make the generation more robust to different LM families.
        with dspy.settings.context(lm=self.write_lead_engine, show_guidelines=False):
            lead_section = self.write_lead(
                opportunity=opportunity, draft_page=draft_page
            ).lead_section
            if "The lead section:" in lead_section:
                lead_section = lead_section.split("The lead section:")[1].strip()
        if polish_whole_page:
            # NOTE: Change show_guidelines to false to make the generation more robust to different LM families.
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                page = self.polish_page(draft_page=draft_page).page
        else:
            page = draft_page

        return dspy.Prediction(lead_section=lead_section, page=page)
