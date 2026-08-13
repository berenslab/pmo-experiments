"""Module for detecting dataset contamination in PubMed-Ophtha against test datasets."""

import asyncio
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import unicodedata
from typing import TypedDict
from xml.etree import ElementTree as ET

import pyarrow.parquet as pq
from rapidfuzz import fuzz
from tqdm.auto import tqdm

from pmo_experiments.util.pmc_article_download import save_article_bodies

logger = logging.getLogger(__name__)

DATASET_KEYWORD_MAP = {
    "aptos": ["aptos", "aptos2019"],
    "refuge2": ["refuge", "refuge2"],
    "rfmid": ["rfmid", "rfmid2", "rfmidv2", "rfmid 2.0", "rfmid 2", "rfmid v2"],
    "brset": ["brset"],
    "eddfs": ["eddfs"],
    "kaggle": ["kaggle", "eyepacs"],
    "idrid": ["idrid"],
}


TEST_DATASET_CITATION_MAP = [
    # IDRiD — IEEE Dataport 2018
    {
        "dataset": "idrid",
        "title": "Indian Diabetic Retinopathy Image Dataset (IDRiD)",
        "url": "https://dx.doi.org/10.21227/H25W98",
        "authors": [
            {"surname": "Porwal", "given_names": "Prasanna"},
            {"surname": "Pachade", "given_names": "Samiksha"},
            {"surname": "Kamble", "given_names": "Ravi"},
            {"surname": "Kokare", "given_names": "Manesh"},
            {"surname": "Deshmukh", "given_names": "Girish"},
            {"surname": "Sahasrabuddhe", "given_names": "Vivek"},
            {"surname": "Meriaudeau", "given_names": "Fabrice"},
        ],
    },
    # EDDFS — MMSP 2022
    {
        "dataset": "eddfs",
        "title": "Eye Disease Diagnosis and Fundus Synthesis: A Large-Scale Dataset and Benchmark",  # noqa: E501
        "url": "https://doi.org/10.1109/MMSP55362.2022.9949547",
        "authors": [
            {"surname": "Xia", "given_names": "Xue"},
            {"surname": "Zhan", "given_names": "Kun"},
            {"surname": "Li", "given_names": "Ying"},
            {"surname": "Xiao", "given_names": "Guobei"},
            {"surname": "Yan", "given_names": "Jinhua"},
            {"surname": "Huang", "given_names": "Zhuxiang"},
            {"surname": "Huang", "given_names": "Guofu"},
            {"surname": "Fang", "given_names": "Yuming"},
        ],
    },
    # BRSET — PhysioNet 2023
    {
        "dataset": "brset",
        "title": "A Brazilian Multilabel Ophthalmological Dataset (BRSET)",
        "url": "https://doi.org/10.13026/xcxw-8198",
        "authors": [
            {"surname": "Nakayama", "given_names": "Luis Filipe"},
            {"surname": "Goncalves", "given_names": "Mariana"},
            {"surname": "Zago Ribeiro", "given_names": "Lucas"},
            {"surname": "Santos", "given_names": "Helen"},
            {"surname": "Ferraz", "given_names": "Daniel"},
            {"surname": "Malerbi", "given_names": "Fernando"},
            {"surname": "Celi", "given_names": "Leo Anthony"},
            {"surname": "Regatieri", "given_names": "Caio"},
        ],
    },
    # BRSET — PLOS Digital Health 2024
    {
        "dataset": "brset",
        "title": "BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos",  # noqa: E501
        "url": "https://doi.org/10.1371/journal.pdig.0000454",
        "authors": [
            {"surname": "Nakayama", "given_names": "Luis Filipe"},
            {"surname": "Restrepo", "given_names": "David"},
            {"surname": "Matos", "given_names": "João"},
            {"surname": "Ribeiro", "given_names": "Lucas Zago"},
            {"surname": "Malerbi", "given_names": "Fernando Korn"},
            {"surname": "Celi", "given_names": "Leo Anthony"},
            {"surname": "Regatieri", "given_names": "Caio Saito"},
        ],
    },
    # REFUGE — arXiv 2019
    {
        "dataset": "refuge",
        "title": "REFUGE Challenge: A Unified Framework for Evaluating Automated Methods for Glaucoma Assessment from Fundus Photographs",  # noqa: E501
        "url": "https://arxiv.org/abs/1910.03667",
        "authors": [
            {"surname": "Orlando", "given_names": "José Ignacio"},
            {"surname": "Fu", "given_names": "Huazhu"},
            {"surname": "Barbosa Breda", "given_names": "João"},
            {"surname": "van Keer", "given_names": "Karel"},
            {"surname": "Bathula", "given_names": "Deepti R."},
            {"surname": "Diaz-Pinto", "given_names": "Andrés"},
            {"surname": "Fang", "given_names": "Ruogu"},
            {"surname": "Heng", "given_names": "Pheng-Ann"},
            {"surname": "Kim", "given_names": "Jeyoung"},
            {"surname": "Lee", "given_names": "Joonho"},
            {"surname": "Lee", "given_names": "Joonseok"},
            {"surname": "Li", "given_names": "Xiaoxiao"},
            {"surname": "Liu", "given_names": "Peng"},
            {"surname": "Lu", "given_names": "Shuai"},
            {"surname": "Murugesan", "given_names": "Balamurali"},
            {"surname": "Naranjo", "given_names": "Valery"},
            {"surname": "Phaye", "given_names": "Sai Samarth R."},
            {"surname": "Shankaranarayana", "given_names": "Sharath M."},
            {"surname": "Sikka", "given_names": "Apoorva"},
            {"surname": "Son", "given_names": "Jaemin"},
            {"surname": "van den Hengel", "given_names": "Anton"},
            {"surname": "Wang", "given_names": "Shujun"},
            {"surname": "Wu", "given_names": "Junyan"},
            {"surname": "Wu", "given_names": "Zifeng"},
            {"surname": "Xu", "given_names": "Guanghui"},
            {"surname": "Xu", "given_names": "Yong-Li"},
            {"surname": "Yin", "given_names": "Pengshuai"},
            {"surname": "Li", "given_names": "Fei"},
            {"surname": "Zhang", "given_names": "Xiulan"},
            {"surname": "Xu", "given_names": "Yanwu"},
            {"surname": "Bogunovic", "given_names": "Hrvoje"},
        ],
    },
    # REFUGE
    {
        "dataset": "refuge",
        "title": "REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs",  # noqa: E501
        "url": "https://doi.org/10.1016/j.media.2019.101570",
        "authors": [
            {"surname": "Orlando", "given_names": "José Ignacio"},
            {"surname": "Fu", "given_names": "Huazhu"},
            {"surname": "Barbosa Breda", "given_names": "João"},
            {"surname": "van Keer", "given_names": "Karel"},
            {"surname": "Bathula", "given_names": "Deepti R."},
            {"surname": "Diaz-Pinto", "given_names": "Andrés"},
            {"surname": "Fang", "given_names": "Ruogu"},
            {"surname": "Heng", "given_names": "Pheng-Ann"},
            {"surname": "Kim", "given_names": "Jeyoung"},
            {"surname": "Lee", "given_names": "Joonho"},
            {"surname": "Lee", "given_names": "Joonseok"},
            {"surname": "Li", "given_names": "Xiaoxiao"},
            {"surname": "Liu", "given_names": "Peng"},
            {"surname": "Lu", "given_names": "Shuai"},
            {"surname": "Murugesan", "given_names": "Balamurali"},
            {"surname": "Naranjo", "given_names": "Valery"},
            {"surname": "Phaye", "given_names": "Sai Samarth R."},
            {"surname": "Shankaranarayana", "given_names": "Sharath M."},
            {"surname": "Sikka", "given_names": "Apoorva"},
            {"surname": "Son", "given_names": "Jaemin"},
            {"surname": "van den Hengel", "given_names": "Anton"},
            {"surname": "Wang", "given_names": "Shujun"},
            {"surname": "Wu", "given_names": "Junyan"},
            {"surname": "Wu", "given_names": "Zifeng"},
            {"surname": "Xu", "given_names": "Guanghui"},
            {"surname": "Xu", "given_names": "Yong-Li"},
            {"surname": "Yin", "given_names": "Pengshuai"},
            {"surname": "Li", "given_names": "Fei"},
            {"surname": "Zhang", "given_names": "Xiulan"},
            {"surname": "Xu", "given_names": "Yanwu"},
            {"surname": "Bogunović", "given_names": "Hrvoje"},
        ],
    },
    # RFMiD v1 — Data 2021
    {
        "dataset": "rfmid",
        "title": "Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research",  # noqa: E501
        "url": "https://www.mdpi.com/2306-5729/6/2/14",
        "authors": [
            {"surname": "Pachade", "given_names": "Samiksha"},
            {"surname": "Porwal", "given_names": "Prasanna"},
            {"surname": "Thulkar", "given_names": "Dhanshree"},
            {"surname": "Kokare", "given_names": "Manesh"},
            {"surname": "Deshmukh", "given_names": "Girish"},
            {"surname": "Sahasrabuddhe", "given_names": "Vivek"},
            {"surname": "Giancardo", "given_names": "Luca"},
            {"surname": "Quellec", "given_names": "Gwenolé"},
            {"surname": "Mériaudeau", "given_names": "Fabrice"},
        ],
    },
    # RFMiD 2.0 — Data 2023
    {
        "dataset": "rfmid",
        "title": "Retinal Fundus Multi-Disease Image Dataset (RFMiD) 2.0: A Dataset of Frequently and Rarely Identified Diseases",  # noqa: E501
        "url": "https://www.mdpi.com/2306-5729/8/2/29",
        "authors": [
            {"surname": "Panchal", "given_names": "Sachin"},
            {"surname": "Naik", "given_names": "Ankita"},
            {"surname": "Kokare", "given_names": "Manesh"},
            {"surname": "Pachade", "given_names": "Samiksha"},
            {"surname": "Naigaonkar", "given_names": "Rushikesh"},
            {"surname": "Phadnis", "given_names": "Prerana"},
            {"surname": "Bhange", "given_names": "Archana"},
        ],
    },
    # REFUGE2 — arXiv 2022
    {
        "dataset": "refuge2",
        "title": "REFUGE2 Challenge: A Treasure Trove for Multi-Dimension Analysis and Evaluation in Glaucoma Screening",  # noqa: E501
        "url": "https://arxiv.org/abs/2202.08994",
        "authors": [
            {"surname": "Fang", "given_names": "Huihui"},
            {"surname": "Li", "given_names": "Fei"},
            {"surname": "Wu", "given_names": "Junde"},
            {"surname": "Fu", "given_names": "Huazhu"},
            {"surname": "Sun", "given_names": "Xu"},
            {"surname": "Son", "given_names": "Jaemin"},
            {"surname": "Yu", "given_names": "Shuang"},
            {"surname": "Zhang", "given_names": "Menglu"},
            {"surname": "Yuan", "given_names": "Chenglang"},
            {"surname": "Bian", "given_names": "Cheng"},
            {"surname": "Lei", "given_names": "Baiying"},
            {"surname": "Zhao", "given_names": "Benjian"},
            {"surname": "Xu", "given_names": "Xinxing"},
            {"surname": "Li", "given_names": "Shaohua"},
            {"surname": "Fumero", "given_names": "Francisco"},
            {"surname": "Sigut", "given_names": "José"},
            {"surname": "Almubarak", "given_names": "Haidar"},
            {"surname": "Bazi", "given_names": "Yakoub"},
            {"surname": "Guo", "given_names": "Yuanhao"},
            {"surname": "Zhou", "given_names": "Yating"},
            {"surname": "Baid", "given_names": "Ujjwal"},
            {"surname": "Innani", "given_names": "Shubham"},
            {"surname": "Guo", "given_names": "Tianjiao"},
            {"surname": "Yang", "given_names": "Jie"},
            {"surname": "Orlando", "given_names": "José Ignacio"},
            {"surname": "Bogunović", "given_names": "Hrvoje"},
            {"surname": "Zhang", "given_names": "Xiulan"},
            {"surname": "Xu", "given_names": "Yanwu"},
        ],
    },
    # REFUGE2 — IEEE Dataport 2026
    {
        "dataset": "refuge2",
        "title": "REFUGE2 Challenge: A Treasure Trove for Multi-Dimension Analysis and Evaluation in Glaucoma Screening",  # noqa: E501
        "url": "https://dx.doi.org/10.21227/7d23-j419",
        "authors": [
            {"surname": "Fang", "given_names": "Huihui"},
        ],
    },
    # APTOS 2019 — Kaggle 2019
    {
        "dataset": "aptos",
        "title": "APTOS 2019 Blindness Detection",
        "url": "https://kaggle.com/competitions/aptos2019-blindness-detection",
        "authors": [
            {"surname": "", "given_names": "Karthik"},
            {"surname": "", "given_names": "Maggie"},
            {"surname": "Dane", "given_names": "Sohier"},
        ],
    },
    # Kaggle EyePACS — Kaggle 2015
    {
        "dataset": "kaggle",
        "title": "Diabetic Retinopathy Detection",
        "url": "https://kaggle.com/competitions/diabetic-retinopathy-detection",
        "authors": [
            {"surname": "Dugas", "given_names": "Emma"},
            {"surname": "", "given_names": "Jared"},
            {"surname": "", "given_names": "Jorge"},
            {"surname": "Cukierski", "given_names": "Will"},
        ],
    },
]

# TODO move into function
KEYWORD_PATTERNS = {
    dataset_name: [
        re.compile(rf"\b{re.escape(kw.lower())}(?![a-z])") for kw in keywords
    ]
    for dataset_name, keywords in DATASET_KEYWORD_MAP.items()
}

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")

FUZZY_TITLE_THRESHOLD = 90


def parse_reference(ref: ET.Element) -> dict[str, str | list[dict[str, str]] | None]:
    """
    Parse a reference XML element into a dictionary.

    Args:
        ref (ET.Element): XML element of the reference.

    Returns:
        dict: Dictionary containing the parsed reference information.

    """
    # If mixed citation use other parsing
    mixed_citation = ref.find(".//mixed-citation")
    if mixed_citation is not None:
        mixed_citation_text = (
            mixed_citation.text.strip() if mixed_citation.text is not None else ""
        )
        pub_id = mixed_citation.find(".//pub-id[@pub-id-type='pmid']")
        pmid = pub_id.text if pub_id is not None else None
    else:
        # Get the pub-id of the reference
        pub_id = ref.find(".//pub-id[@pub-id-type='pmid']")
        pmid = pub_id.text if pub_id is not None else None

        mixed_citation_text = None

    # Get authors of the reference
    authors = ref.findall(".//person-group[@person-group-type='author']/name")

    # Convert to json
    authors_json = []
    for author in authors:
        # Check if style is western
        name_style = author.get("name-style")
        if name_style is not None and name_style != "western":
            continue

        _surname_el = author.find("surname")
        surname = _surname_el.text if _surname_el is not None else ""
        _given_el = author.find("given-names")
        given_names = _given_el.text if _given_el is not None else ""
        authors_json.append({"surname": surname, "given_names": given_names})

    # Get the name of the reference
    _title_el = ref.find(".//article-title")
    ref_title = _title_el.text if _title_el is not None else ""

    if ref_title is not None:
        ref_title = ref_title.strip()
    else:
        ref_title = ""

    # if ref_title == "" and mixed_citation is None:
    #     raise ValueError(
    #         f"Reference title is empty for ref: {ET.tostring(ref, encoding='unicode')}"  # noqa: E501
    #     )

    # Get the journal of the reference
    _source_el = ref.find(".//source")
    ref_journal = _source_el.text if _source_el is not None else ""

    if ref_journal is not None:
        ref_journal = ref_journal.strip()
    else:
        ref_journal = ""

    if ref_journal == "":
        # Extract text from the tags in the source element
        source_element = ref.find(".//source")
        if source_element is not None:
            ref_journal = "".join(
                source_element.itertext()
            ).strip()  # Get all text including child elements

    # Get the year of the reference
    _year_el = ref.find(".//year")
    ref_year = _year_el.text if _year_el is not None else ""

    if ref_year is not None:
        ref_year = ref_year.strip()
    else:
        ref_year = ""

    ref_json = {
        "title": ref_title,
        "journal": ref_journal,
        "year": ref_year,
        "authors": authors_json,
        "pmid": pmid,
        "mixed_citation": mixed_citation_text,
    }

    return ref_json


def normalize_title(s: str | None) -> str:
    """
    Normalize the title of a reference.

    Args:
        s (str | None): String to normalize.

    Returns:
        str: Normalized title or empty string if s is None.

    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# TODO move into function
NORMALIZED_CITATIONS = [
    (citation["dataset"], normalize_title(citation["title"]))
    for citation in TEST_DATASET_CITATION_MAP
]


def extract_doi(url: str | None) -> str | None:
    """
    Extract DOI from a URL.

    Args:
        url (str): URL to extract DOI from.

    Returns:
        str | None: Extracted DOI or None if not found.

    """
    if url is None:
        return None
    match = DOI_RE.search(url.lower())
    if match is None:
        return None
    return match.group(0).rstrip(".,;)")


# TODO move into function
CITATION_DOIS = {}
for citation in TEST_DATASET_CITATION_MAP:
    doi = extract_doi(citation.get("url"))
    if doi is not None:
        CITATION_DOIS.setdefault(citation["dataset"], set()).add(doi)


class ContaminationResult(TypedDict):
    """Result of a contamination check for a single article."""

    file_path: str
    article_id: int
    string_matches: list[str]
    ref_string_matches: list[str]
    matched_citations: list[str]
    matched_urls: list[str]
    matched_dois: list[str]


def check_contamination(file_path: str) -> ContaminationResult | None:
    """
    Check file for test contamination.

    Uses provides keywords and references.
    Matches the references using fuzzy search.

    Args:
        file_path (str): Path to XML of PMC article.

    Returns:
        ContaminationResult | None: Found contamination or None.

    """
    with open(file_path) as f:
        full_text_xml = f.read()
        root = ET.fromstring(full_text_xml)

    # Get the body of the article
    body = root.find(".//body")
    if body is None:
        logger.warning(f"No body found in file {file_path}")
    body_text = (
        ET.tostring(body, encoding="unicode").lower() if body is not None else ""
    )

    full_article_text = ET.tostring(root, encoding="unicode").lower()

    string_matches = set()
    for dataset_name, patterns in KEYWORD_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(body_text) is not None:
                string_matches.add(dataset_name)
                break
            elif pattern.search(full_article_text) is not None:
                string_matches.add(dataset_name)
                break

    references = root.findall(".//ref")

    parsed_references = []

    for ref in references:
        try:
            parsed_ref = parse_reference(ref)
            parsed_references.append(parsed_ref)
        except ValueError as e:
            logger.error(f"Error parsing reference in file {file_path}: {e}")
            continue

    ref_string_matches = set()
    for ref in references:
        ref_text = ET.tostring(ref, encoding="unicode").lower()
        for dataset_name, patterns in KEYWORD_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(ref_text) is not None:
                    ref_string_matches.add(dataset_name)
                    break

    matched_citations = set()
    for ref in parsed_references:
        ref_title_norm = normalize_title(ref["title"])
        if ref_title_norm == "":
            continue
        for dataset_name, citation_title_norm in NORMALIZED_CITATIONS:
            if citation_title_norm == "":
                continue
            if ref_title_norm == citation_title_norm:
                matched_citations.add(dataset_name)
                break
            if (
                fuzz.token_set_ratio(ref_title_norm, citation_title_norm)
                >= FUZZY_TITLE_THRESHOLD
            ):
                matched_citations.add(dataset_name)
                break

    matched_urls = set()
    for ref in references:
        for citation in TEST_DATASET_CITATION_MAP:
            if (
                citation.get("url") is not None
                and citation["url"].lower()
                in ET.tostring(ref, encoding="unicode").lower()
            ):
                matched_urls.add(citation["dataset"])
                break  # Stop searching after the first match

    for citation in TEST_DATASET_CITATION_MAP:
        if citation.get("url") is not None and citation["url"].lower() in body_text:
            matched_urls.add(citation["dataset"])
        elif (
            citation.get("url") is not None
            and citation["url"].lower() in full_article_text
        ):
            matched_urls.add(citation["dataset"])

        if citation["dataset"] != "kaggle" and citation["title"].lower() in body_text:
            matched_citations.add(citation["dataset"])
        elif (
            citation["dataset"] != "kaggle"
            and citation["title"].lower() in full_article_text
        ):
            matched_citations.add(citation["dataset"])

    matched_dois = set()
    article_dois = {doi.rstrip(".,;)") for doi in DOI_RE.findall(full_article_text)}
    for dataset_name, dois in CITATION_DOIS.items():
        if len(article_dois & dois) > 0:
            matched_dois.add(dataset_name)

    if (
        len(string_matches) > 0
        or len(matched_citations) > 0
        or len(ref_string_matches) > 0
        or len(matched_urls) > 0
        or len(matched_dois) > 0
    ):
        return {
            "file_path": file_path,
            "article_id": int(os.path.basename(file_path).removesuffix(".xml")),
            "string_matches": list(string_matches),
            "ref_string_matches": list(ref_string_matches),
            "matched_citations": list(matched_citations),
            "matched_urls": list(matched_urls),
            "matched_dois": list(matched_dois),
        }

    return None


def detect_dataset_contamination(
    dataset_path: str,
    article_body_folder: str,
    delete_body_folder: bool = False,
    output_file: str = "contaminated_panels.json",
    relevant_datasets: list[str] | None = None,
    num_concurrent_downloads: int = 3,
    num_article_ids_per_request: int = 100,
    num_workers: int = 6,
):
    """
    Detect dataset contamination for all files in PubMed-Ophtha against test datasets.

    Args:
        dataset_path (str): Path to PubMed-Ophtha parquet file.
        article_body_folder (str): Folder to save the article XML files to.
        delete_body_folder (bool, optional): If True the folder containing the article
            bodies is deleted after processing. Defaults to False.
        output_file (str, optional): File to save the contaminated_panels to. Defaults
            to "contaminated_panels.json".
        relevant_datasets (list[str] | None, optional): The test datasets to check
            against. If None checks against:
                - aptos
                - refuge
                - refuge2
                - rfmid
                - brset
                - eddfs

            Defaults to None.
        num_concurrent_downloads (int, optional): Number of concurrent article
            downloads. Defaults to 3.
        num_article_ids_per_request (int, optional): Number of articles to fetch per
            request. Defaults to 100.
        num_workers (int, optional): Number of worker processes when checking for
            contamination. Defaults to 6.

    """
    assert output_file.endswith(".json"), "Output file must be a JSON file."
    if relevant_datasets is None:
        relevant_datasets = ["aptos", "refuge2", "rfmid", "brset", "eddfs", "refuge"]

    all_datasets = set(DATASET_KEYWORD_MAP.keys()).union(
        {citation["dataset"] for citation in TEST_DATASET_CITATION_MAP}
    )

    ignored_datasets = set(all_datasets) - set(relevant_datasets)

    # Download the articles
    asyncio.run(
        save_article_bodies(
            dataset_path,
            article_body_folder,
            num_concurrent_requests=num_concurrent_downloads,
            num_article_ids_per_request=num_article_ids_per_request,
        )
    )

    all_files = glob.glob(os.path.join(article_body_folder, "*.xml"))

    parquet_file = pq.ParquetFile(dataset_path)
    schema = parquet_file.schema
    table = parquet_file.read(
        columns=[field for field in schema.names if field != "panel_image_bytes"]
    )
    dataset_df = table.to_pandas()

    contaminated_panels = []
    with mp.Pool(processes=num_workers) as pool:
        for entry in tqdm(
            pool.imap(check_contamination, all_files), total=len(all_files)
        ):
            if entry is not None:
                all_matches = set(
                    entry["string_matches"]
                    + entry["ref_string_matches"]
                    + entry["matched_citations"]
                    + entry["matched_urls"]
                    + entry["matched_dois"]
                )

                if len(all_matches - set(ignored_datasets)) > 0:
                    contaminated_panels.append(
                        {
                            "article_id": entry["article_id"],
                            "panel_ids": dataset_df[
                                dataset_df["article_id"] == entry["article_id"]
                            ]["panel_id"].tolist(),
                        }
                    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(contaminated_panels, f, indent=4, ensure_ascii=False)

    if delete_body_folder:
        for file in all_files:
            os.remove(file)

        try:
            os.rmdir(article_body_folder)
        except OSError:
            logger.warning(
                f"Could not remove folder {article_body_folder}. It may not be empty."
            )
