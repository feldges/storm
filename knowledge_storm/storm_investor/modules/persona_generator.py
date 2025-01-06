import logging
import re
from typing import Union, List

import dspy
import requests
from bs4 import BeautifulSoup


def get_wiki_page_title_and_toc(url):
    """Get the main title and table of contents from an url of a Wikipedia page."""

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Get the main title from the first h1 tag
    main_title = soup.find("h1").text.replace("[edit]", "").strip().replace("\xa0", " ")

    toc = ""
    levels = []
    excluded_sections = {
        "Contents",
        "See also",
        "Notes",
        "References",
        "External links",
    }

    # Start processing from h2 to exclude the main title from TOC
    for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        level = int(
            header.name[1]
        )  # Extract the numeric part of the header tag (e.g., '2' from 'h2')
        section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")
        if section_title in excluded_sections:
            continue

        while levels and level <= levels[-1]:
            levels.pop()
        levels.append(level)

        indentation = "  " * (len(levels) - 1)
        toc += f"{indentation}{section_title}\n"

    return main_title, toc.strip()


class FindRelatedOpportunity(dspy.Signature):
    """I'm writing an investment report for the investment opportunity mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this investment opportunity, or examples that help me understand the typical content and structure included in investment reports for similar opportunities.
    Please list the urls in separate lines."""

    opportunity = dspy.InputField(prefix="Investment opportunity:", format=str)
    related_opportunities = dspy.OutputField(format=str)


class GenPersona(dspy.Signature):
    """You need to select a group of experts or investment professionals (the persona) who will work together to create a comprehensive report on the potential investment opportunity, typically a company. Each of them represents a different perspective, role, or affiliation related to this investment opportunity. You can use other Wikipedia pages of related investment opportunities for inspiration. For each editor, add a description of what they will focus on.
    Give your answer in the following format: 1. short summary of persona 1: description\n2. short summary of persona 2: description\n...
    """

    opportunity = dspy.InputField(prefix="Investment opportunity:", format=str)
    examples = dspy.InputField(
        prefix="Wiki page outlines of related opportunities for inspiration:\n", format=str
    )
    personas = dspy.OutputField(format=str)


class CreateWriterWithPersona(dspy.Module):
    """Discover different perspectives of researching the investment opportunity by reading Wikipedia pages of related opportunities."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.find_related_opportunity = dspy.ChainOfThought(FindRelatedOpportunity)
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.engine = engine

    def forward(self, opportunity: str, draft=None):
        with dspy.settings.context(lm=self.engine):
            # Get section names from wiki pages of relevant investment opportunities for inspiration.
            related_opportunities = self.find_related_opportunity(opportunity=opportunity).related_opportunities
            urls = []
            for s in related_opportunities.split("\n"):
                if "http" in s:
                    urls.append(s[s.find("http") :])
            examples = []
            for url in urls:
                try:
                    title, toc = get_wiki_page_title_and_toc(url)
                    examples.append(f"Title: {title}\nTable of Contents: {toc}")
                except Exception as e:
                    logging.error(f"Error occurs when processing {url}: {e}")
                    continue
            if len(examples) == 0:
                examples.append("N/A")
            gen_persona_output = self.gen_persona(
                opportunity=opportunity, examples="\n----------\n".join(examples)
            ).personas

        personas = []
        for s in gen_persona_output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                personas.append(match.group(1))

        sorted_personas = personas

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=sorted_personas,
            related_opportunities=related_opportunities,
        )


class StormPersonaGenerator:
    """
    A generator class for creating personas based on a given investment opportunity.

    This class uses an underlying engine to generate personas tailored to the specified investment opportunity.
    The generator integrates with a `CreateWriterWithPersona` instance to create diverse personas,
    including a default 'Basic Fact Writer' persona.

    Attributes:
        create_writer_with_persona (CreateWriterWithPersona): An instance responsible for
            generating personas based on the provided engine and investment opportunity.

    Args:
        engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The underlying engine used for generating
            personas. It must be an instance of either `dspy.dsp.LM` or `dspy.dsp.HFModel`.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.create_writer_with_persona = CreateWriterWithPersona(engine=engine)

    def generate_persona(self, opportunity: str, max_num_persona: int = 3) -> List[str]:
        """
        Generates a list of personas based on the provided investment opportunity, up to a maximum number specified.

        This method first creates personas using the underlying `create_writer_with_persona` instance
        and then prepends a default 'Basic Fact Writer' persona to the list before returning it.
        The number of personas returned is limited to `max_num_persona`, excluding the default persona.

        Args:
            opportunity (str): The investment opportunity for which personas are to be generated, typically a company name or a company name with a short description (as a one-liner).
            max_num_persona (int): The maximum number of personas to generate, excluding the
                default 'Basic Fact Writer' persona.

        Returns:
            List[str]: A list of persona descriptions, including the default 'Basic Fact Writer' persona
                and up to `max_num_persona` additional personas generated based on the investment opportunity.
        """
        personas = self.create_writer_with_persona(opportunity=opportunity)
        default_persona = "Basic Fact Writer: Basic Fact Writer focusing on broadly covering the basic facts about the investment opportunity."
        considered_personas = [default_persona] + personas.personas[:max_num_persona]
        return considered_personas
