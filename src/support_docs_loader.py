import json
import requests
import re
import uuid
from pathlib import Path
import os
import pandas as pd
from enum import Enum
from pandas import DataFrame
from bs4 import BeautifulSoup, Tag, NavigableString
from typing import List, Iterator, Dict, Any, Sequence, Tuple, Set
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.document_loaders import BaseLoader


class SupportingDocumentsLoader(BaseLoader):
    """
    A class for loading Cisco supporting documents.

    At the moment, it supports loading of Admin Guide and CLI Guide documents.

    The main class entry method is `from_url`, which expects a url like this:

    https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-350-cli-.html

    It then parses the Book Table of Contents structure to get the paths to the individual documents.

    The `load` method fetches the documents from the paths and returns a list of Document objects.

    Example:

        ```python
        loader = SupportingDocumentsLoader.from_url("https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-350-cli-.html")
        documents = loader.load()

        ```

    You could also pass a list of URLs to the constructor but it does expect a certain format.
    """

    def __init__(self, paths: List[str]) -> None:
        self.paths = paths
        self.documents: List[Document] = []

    def save_to_json(self, path: str) -> None:
        with open(path, "w") as json_file:
            json_docs = [doc.dict() for doc in self.documents]
            json.dump(json_docs, json_file, indent=4)

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        for path in self.paths:
            yield from self._fetch(path)

    def _fetch(self, path: str):
        response = requests.get(path)
        response.raise_for_status()
        page_content = response.content
        soup = BeautifulSoup(page_content, "html.parser")
        topic_data = self._parse_content(soup)
        metadatas = [
            self._build_metadata(soup, path, topic=data["topic"]) for data in topic_data
        ]
        for data, meta in zip(topic_data, metadatas):
            if data["text"] == "This chapter contains the following sections:":
                continue
            yield Document(page_content=data["text"], metadata=meta)

    def load_schema(self):
        return list(self.lazy_load_cli_schema())

    def lazy_load_cli_schema(self):
        for path in self.paths:
            yield from self._fetch_cli(path)

    def _fetch_cli(self, path: str):
        response = requests.get(path)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        topic_data = self._parse_cli_guide(soup)
        for data in topic_data:
            yield data

    @classmethod
    def from_url(cls, url: str) -> "SupportingDocumentsLoader":
        """
        Create a SupportingDocumentsLoader instance from a given URL.

        Args:
            url (str): The URL to fetch the supporting documents from.

        Returns:
            SupportingDocumentsLoader: An instance of the SupportingDocumentsLoader class.

        Raises:
            requests.HTTPError: If there is an HTTP error while fetching the URL.
        """
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        toc = soup.select("ul#bookToc > li > a")
        links = [f"https://www.cisco.com{link.get('href')}" for link in toc]
        return cls(paths=links)

    @staticmethod
    def sanitize_text(text: str) -> str:
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        cleaned_text = cleaned_text.replace("\\", "")
        cleaned_text = cleaned_text.replace("#", " ")
        cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)
        return cleaned_text

    @staticmethod
    def _build_metadata(soup: BeautifulSoup, url: str, **kwargs) -> Dict[str, str]:
        """Build metadata from BeautifulSoup output.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML.
            url (str): The URL of the source.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, str]: The metadata dictionary containing the extracted information.
        """
        metadata = {"source": url}
        if title := soup.find("meta", attrs={"name": "description"}):
            metadata["title"] = title.get("content", "Chapter not found.")
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")
        if concept := soup.find("meta", attrs={"name": "concept"}):
            metadata["concept"] = concept.get("content", "No concept found.")
        if topic := kwargs.get("topic"):
            metadata["topic"] = topic
        metadata["doc_id"] = str(uuid.uuid4())
        return metadata

    def _parse_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        This method is useful for parsing the content of the Admin Guide or CLI Guide into a large chunk of text and topics suitable for Langchain Documents.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object representing the parsed HTML content.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the parsed topics and text.

        """
        topic_sections = soup.find_all("section", class_=("body"))

        return [
            {
                "topic": (
                    section.find_previous("h2", class_="title").get_text(strip=True)
                    + " "
                    + section.find_previous_sibling("h3", class_="title").get_text(
                        strip=True
                    )
                    if section.find_previous_sibling("h3", class_="title")
                    else section.find_previous(class_="title").get_text(strip=True)
                ),
                "text": self.sanitize_text(section.get_text()),
            }
            for section in topic_sections
        ]

    def load_and_split(
        self, text_splitter: TextSplitter | None = None
    ) -> List[Document]:
        """
        Loads the documents and splits them into smaller chunks using the specified text splitter.

        Args:
            text_splitter (TextSplitter | None): The text splitter to use for splitting the documents.
                If None, a default RecursiveCharacterTextSplitter will be used.

        Returns:
            List[Document]: A list of split documents.

        """
        docs = self.load()
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, chunk_overlap=50, add_start_index=True
            )
        return text_splitter.split_documents(docs)

    def _parse_cli_guide(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Parses the CLI Guide only into a suitable schema suitable for Database storage.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object containing the HTML content of the CLI Guide.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the parsed CLI Guide data.

        """
        article_bodies = soup.find_all(
            "article", class_=("topic", "reference", "nested1")
        )
        topic: str = soup.find("meta", attrs={"name": "description"}).get(
            "content", None
        )
        cli_sections = []
        for article in article_bodies:
            section_body: Tag | NavigableString = article.find(
                "section", class_=("body")
            )
            command_name = section_body.find_previous(class_=("title",)).get_text(
                strip=True
            )
            if topic == "Introduction":
                sections = []
                content = {"description": [], "command_name": command_name}
                for child in article.children:
                    if not isinstance(child, Tag):
                        continue
                    if child.name == "p":
                        sections.append(child.get_text())
                    elif child.name == "ul":
                        sec = [li.get_text() for li in child.find_all("li")]
                        sections.extend(sec)
                    elif child.name == "pre":
                        lines = child.get_text().split("\n")
                        sections.extend(lines)
                    else:
                        sections.append(child.get_text())
                content["description"] = list(map(self.sanitize_text, sections))
                content["description"] = list(
                    filter(lambda x: x != "", content["description"])
                )
                content["topic"] = topic
                print(content)
                cli_sections.append(content)

            else:
                body_sections = section_body.find_all("section")
                (
                    description,
                    syntax,
                    parameters,
                    default_config,
                    command_mode,
                    user_guidelines,
                    examples,
                ) = (None, None, [], None, None, None, None)
                seen_parameters = set()
                for i, sec in enumerate(body_sections):
                    if i == 0:
                        description = sec.get_text()
                    elif sec.find(string=re.compile(r"^Syntax", flags=re.I)):
                        paragraphs = sec.find_all("p")
                        syntax = [p.get_text() for p in paragraphs]
                    elif sec.find(string=re.compile(r"^Parameters", flags=re.I)):
                        p = sec.find("p")
                        if p:
                            normalized_text = p.get_text().strip()
                            if normalized_text not in seen_parameters:
                                seen_parameters.add(normalized_text)
                                parameters = [normalized_text]
                        ul = sec.find("ul")
                        if ul:
                            li = ul.find_all("li")
                            for l in li:
                                normalized_text = l.get_text().strip()
                                if normalized_text not in seen_parameters:
                                    seen_parameters.add(normalized_text)
                                    parameters.append(normalized_text)

                    elif sec.find(string=re.compile(r"^Default Configuration")):
                        p = sec.find("p")
                        default_config = p.get_text() if p else sec.get_text()
                    elif sec.find(string=re.compile(r"^Command Mode")):
                        command_mode_p = sec.find("p")
                        command_mode = (
                            command_mode_p.get_text()
                            if command_mode_p
                            else sec.get_text()
                        )
                    elif sec.find(string=re.compile(r"^User Guidelines")):
                        p = sec.find_all("p")
                        if p:
                            user_guidelines = [para.get_text() for para in p]
                            user_guidelines = " ".join(user_guidelines)
                        else:
                            user_guidelines = sec.get_text()
                    elif sec.find(string=re.compile(r"^Examples?")):
                        examples = self._get_examples(sec)
                        print(f"Examples: {examples}")

                cli_sections.append(
                    {
                        "topic": topic,
                        "command_name": command_name,
                        "description": (
                            self.sanitize_text(description) if description else None
                        ),
                        "syntax": (
                            list(map(self.sanitize_text, syntax)) if syntax else None
                        ),
                        "parameters": list(map(self.sanitize_text, parameters)),
                        "default_configuration": (
                            self.sanitize_text(default_config)
                            if default_config
                            else None
                        ),
                        "command_mode": (
                            self.sanitize_text(command_mode) if command_mode else None
                        ),
                        "user_guidelines": (
                            self.sanitize_text(user_guidelines)
                            if user_guidelines
                            else None
                        ),
                        "examples": examples,
                    }
                )
        return cli_sections

    def _get_examples(self, section: Tag) -> List[Dict[str, Any]]:
        """
        Gets the examples of each section from the CLI Guide.

        Args:
            section (Tag): The section tag containing the examples.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the examples.
        """
        examples = []
        dic = {}
        desc = section.find("p")
        if desc:
            dic["description"] = self.sanitize_text(desc.get_text())
        ul = section.find("ul")
        if ul:
            li: Tag | NavigableString = ul.find_all("li")
            dic["commands"] = [" ".join(l.stripped_strings) for l in li]
            # dic["commands"] = [l.get_text() for l in li]
        pre = section.find("pre")
        if pre:
            lines = pre.get_text().split("\n")
            if "commands" in dic:
                dic["commands"].extend(lines)
            else:
                dic["commands"] = lines
        if "commands" in dic:
            dic["commands"] = list(filter(lambda x: x != "", dic["commands"]))
        examples.append(dic)
        return examples


class ReportType(str, Enum):
    CSV = "csv"
    HTML = "html"
    XML = "xml"
    EXCEL = "excel"
    JSON = "json"
    MARKDOWN = "markdown"
    DICT = "dict"
    PICKLE = "pickle"
    STRING = "string"


class CLIReports:
    def __init__(
        self,
        product_urls: Dict[str, str],
        schema_path: str = f"{Path.cwd()}/data/schemas",
        report_path: str = f"{Path.cwd()}/data/reports",
    ) -> None:
        self.product_urls = product_urls
        self.schema_path = schema_path
        self.report_path = report_path
        self.make_dir()
        self.products_schemas = self.load_schemas()
        self.product_commands: Dict[str, Set[str]] = {}
        self.commands_to_topics: Dict[str, List[Tuple[str, str]]] = {}

    def make_dir(self):
        os.makedirs(self.schema_path, exist_ok=True)
        os.makedirs(self.report_path, exist_ok=True)

    def load_schemas(self):
        products_schemas = {}
        for product, url in self.product_urls.items():
            if os.path.exists(f"{self.schema_path}/{product}_cli_schema.json"):
                print(f"Pulling from {self.schema_path}/{product}_cli_schema.json")
                products_schemas[product] = json.load(
                    open(f"{self.schema_path}/{product}_cli_schema.json", "r")
                )
            else:
                print("Scraping from url. Please standby....")
                cli_guide = SupportingDocumentsLoader.from_url(url)
                schema = cli_guide.load_schema()
                products_schemas[product] = schema
                with open(f"{self.schema_path}/{product}_cli_schema.json", "w") as file:
                    json.dump(schema, file, indent=2)
        return products_schemas

    def process_schemas(self):
        for product, schema in self.products_schemas.items():
            commands = set(
                doc["command_name"] for doc in schema if doc["topic"] != "Introduction"
            )
            self.commands_to_topics[product] = [
                (doc["command_name"], doc["topic"])
                for doc in schema
                if doc["topic"] != "Introduction"
            ]
            self.product_commands[product] = commands

    def sort_dataframe(self, df: DataFrame, column_name: str, ascending: bool = True):
        """
        Sorts the dataframe based on the given column in ascending or descending

        Args:
            df (DataFrame): The DataFrame to sort
            column_name (str): The column to sort on
            ascending (bool, optional): Sort ascending. Defaults to True.
        """
        return df.sort_values(by=column_name, ascending=ascending)

    def generate_type_report(
        self, df: DataFrame, path: str, type_name: ReportType = ReportType.CSV
    ):
        """
        Generate a specific type of report. CSV, HTML, Excel, etc.

        Args:
            df (DataFrame): The DataFrame
            path (str): The file path to write to.
            type_name (ReportType): The type of report to generate.
        """
        try:
            if type_name == ReportType.CSV:
                df.to_csv(path, index=False)
            elif type_name == ReportType.HTML:
                df.to_html(path, index=False, justify="center", classes=("table"))
            elif type_name == ReportType.EXCEL:
                df.to_excel(path, index=False)
            elif type_name == ReportType.JSON:
                df.to_json(path, orient="records")
            elif type_name == ReportType.MARKDOWN:
                df.to_markdown(path, index=False)
            elif type_name == ReportType.DICT:
                df.to_dict(path, index=False)
            elif type_name == ReportType.PICKLE:
                df.to_pickle(path)
            elif type_name == ReportType.STRING:
                return df.to_string(index=False)
            else:
                raise ValueError(
                    f"Unsupported report type: Got: {type_name}. Supported types are: {ReportType.__members__}"
                )
        except Exception as e:
            print(f"Error generating report: {e}")

    def generate_reports(self):
        common_commands = set.intersection(*self.product_commands.values())
        common_commands_count = len(common_commands)
        common_commands_df = pd.DataFrame(common_commands, columns=["common_commands"])
        common_commands_df.to_csv("./data/reports/common_commands.csv", index=False)

        unique_commands = set.union(*self.product_commands.values())
        unique_commands_count = len(unique_commands)

        product_diffs = {
            product: commands.difference(common_commands)
            for product, commands in self.product_commands.items()
        }
        product_unique_counts = {
            product: len(commands) for product, commands in product_diffs.items()
        }
        product_total_counts = {
            product: len(commands)
            for product, commands in self.product_commands.items()
        }

        counts_df = pd.DataFrame(
            {
                "product": list(product_total_counts.keys()),
                "total_commands": list(product_total_counts.values()),
                "unique_commands": list(product_unique_counts.values()),
            }
        )
        counts_df["common_commands"] = common_commands_count
        counts_df["unique_commands_all"] = unique_commands_count
        self.generate_type_report(
            counts_df,
            f"{self.report_path}/product_counts.csv",
            ReportType.CSV,
        )
        self.generate_type_report(
            counts_df,
            f"{self.report_path}/product_counts.html",
            ReportType.HTML,
        )

        # Create a dictionary where the keys are the command names
        # The values are list of product families that have the command

        command_to_products = {command: [] for command in unique_commands}
        for product, commands in self.product_commands.items():
            for command in commands:
                command_to_products[command].append(product)
        # Create a DataFrame from the dictionary
        command_to_products_df = pd.DataFrame(
            list(command_to_products.items()), columns=["command", "products"]
        )

        command_to_products_df = self.sort_dataframe(command_to_products_df, "command")

        self.generate_type_report(
            command_to_products_df,
            f"{self.report_path}/command_to_products.csv",
            ReportType.CSV,
        )

        command_to_products_plus_topics = {
            command: {"products": [], "topic": None} for command in unique_commands
        }
        for product, command_list in self.commands_to_topics.items():
            for command, topic in command_list:
                command_to_products_plus_topics[command]["products"].append(product)
                command_to_products_plus_topics[command]["topic"] = topic

        command_to_products_plus_topics_df = pd.DataFrame(
            [
                (command, data["products"], data["topic"])
                for command, data in command_to_products_plus_topics.items()
            ],
            columns=["command", "products", "topic"],
        )
        command_to_products_plus_topics_df = self.sort_dataframe(
            command_to_products_plus_topics_df, "command"
        )
        self.generate_type_report(
            command_to_products_plus_topics_df,
            f"{self.report_path}/command_to_products_plus_topics.csv",
            ReportType.CSV,
        )
        self.generate_type_report(
            command_to_products_plus_topics_df,
            f"{self.report_path}/command_to_products_plus_topics.html",
            ReportType.HTML,
        )

        command_description_df = pd.DataFrame(
            [
                (doc["command_name"], doc["description"])
                for schema in self.products_schemas.values()
                for doc in schema
            ],
            columns=["command", "description"],
        )
        self.generate_type_report(
            command_description_df,
            f"{self.report_path}/command_descriptions.csv",
            ReportType.CSV,
        )
        self.generate_type_report(
            command_description_df,
            f"{self.report_path}/command_descriptions.html",
            ReportType.HTML,
        )

        command_syntax_df = pd.DataFrame(
            [
                (doc["command_name"], doc["syntax"])
                for schema in self.products_schemas.values()
                for doc in schema
                if "syntax" in doc and "command_name" in doc
            ],
            columns=["command", "syntax"],
        )
        self.generate_type_report(
            command_syntax_df,
            f"{self.report_path}/command_syntax.csv",
            ReportType.CSV,
        )

        command_syntax_diffs = []
        for command, products in command_to_products.items():
            syntaxes = set()
            for product in products:
                for doc in self.products_schemas[product]:
                    if doc["command_name"] == command and "syntax" in doc:
                        if isinstance(doc["syntax"], list):
                            syntaxes.update(doc["syntax"])
                        else:
                            syntaxes.add(doc["syntax"])
            if len(syntaxes) > 1:
                command_syntax_diffs.append((command, syntaxes))

        command_syntax_diffs_df = pd.DataFrame(
            command_syntax_diffs, columns=["command", "syntax"]
        )
        command_syntax_diffs_df = self.sort_dataframe(
            command_syntax_diffs_df, "command"
        )
        self.generate_type_report(
            command_syntax_diffs_df,
            f"{self.report_path}/command_syntax_diffs.csv",
        )
        self.generate_type_report(
            command_syntax_diffs_df,
            f"{self.report_path}/command_syntax_diffs.html",
            ReportType.HTML,
        )

        command_syntax_diffs_with_products = []
        for command, products in command_to_products.items():
            syntaxes = {}
            for product in products:
                for doc in self.products_schemas[product]:
                    if doc["command_name"] == command and "syntax" in doc:
                        if isinstance(doc["syntax"], list):
                            for syntax in doc["syntax"]:
                                if syntax not in syntaxes:
                                    syntaxes[syntax] = [product]
                                else:
                                    syntaxes[syntax].append(product)
                        else:
                            if doc["syntax"] not in syntaxes:
                                syntaxes[doc["syntax"]] = [product]
                            else:
                                syntaxes[doc["syntax"]].append(product)
            if len(syntaxes) > 1:
                syntaxes_products = [
                    (syntax, ", ".join(products))
                    for syntax, products in syntaxes.items()
                ]
                command_syntax_diffs_with_products.append((command, syntaxes_products))

        command_syntax_diffs_with_products_df = pd.DataFrame(
            command_syntax_diffs_with_products, columns=["command", "syntax_products"]
        )
        command_syntax_diffs_with_products_df = self.sort_dataframe(
            command_syntax_diffs_with_products_df, "command"
        )
        self.generate_type_report(
            command_syntax_diffs_with_products_df,
            f"{self.report_path}/command_syntax_diffs_with_products.csv",
            ReportType.CSV,
        )

    def run(self):
        self.process_schemas()
        self.generate_reports()


if __name__ == "__main__":
    # Just add the URLs for the products you want to load
    # Ensure the URL is the Book Table of Contents page
    product_urls = {
        "cbs_220": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbss/CBS220/CLI-Guide/b_220CLI.html",
        "cbs_250": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-250-cli.html",
        "cbs_350": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-350-cli-.html",
        "cat_1300": "https://www.cisco.com/c/en/us/td/docs/switches/campus-lan-switches-access/Catalyst-1200-and-1300-Switches/cli/C1300-cli.html",
    }

    cli_reports = CLIReports(product_urls=product_urls)
    cli_reports.run()
