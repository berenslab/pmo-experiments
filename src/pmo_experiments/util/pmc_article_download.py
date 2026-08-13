"""
Module for downloading PMC articles in XML format and saving them to a specified folder.

Uses the Entrez E-utilities API to fetch articles by their PMC IDs.
The articles are saved in a pretty-printed XML format.
"""

import asyncio
import glob
import logging
import os
import time
import xml.dom.minidom
from xml.etree import ElementTree as ET

import pyarrow.parquet as pq
import requests
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


async def get_body_text(
    article_ids: list[str], output_folder: str, sleep_time: int = 2
):
    """
    Download and save the given article ids.

    Args:
        article_ids (list[str]): List of PMC article ids.
        output_folder (str): Folder to save to (needs to exist).
        sleep_time (int, optional): Time to sleep between requests (in seconds).
            Defaults to 2.

    Raises:
        ValueError: For missing PMC ID element in fetched article.

    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": ",".join(article_ids),
        "rettype": "full",
        "retmode": "xml",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    parsing_start_time = time.time()
    root = ET.fromstring(response.text)

    articles = root.findall(".//article")

    # Find the PMC ID in the XML for each article
    output_article_ids = []
    for article in articles:
        pmc_id_elem = article.find(".//article-id[@pub-id-type='pmcaid']")
        if pmc_id_elem is not None:
            output_article_ids.append(pmc_id_elem.text)
        else:
            raise ValueError("PMC ID not found in the XML for an article.")

    for article_id, article in zip(output_article_ids, articles):
        dom = xml.dom.minidom.parseString(ET.tostring(article, encoding="unicode"))
        pretty_xml_as_string = dom.toprettyxml()

        with open(os.path.join(output_folder, f"{article_id}.xml"), "w") as f:
            f.write(pretty_xml_as_string)

    time_diff = time.time() - parsing_start_time

    if time_diff < sleep_time:
        await asyncio.sleep(sleep_time - time_diff)


async def get_body_text_safe(
    semaphore: asyncio.Semaphore,
    article_ids: list[str],
    output_folder: str,
    sleep_time: int = 2,
):
    """
    Download the given PMC articles (with semaphore).

    Args:
        semaphore (asyncio.Semaphore): Semaphore managing concurrent requests.
        article_ids (list[str]): List of PMC article ids.
        output_folder (str): Folder to save the XML files to.
        sleep_time (int, optional): Time to sleep between requests (in seconds).
            Defaults to 2.

    """
    async with semaphore:
        try:
            await get_body_text(article_ids, output_folder, sleep_time)
        except Exception as e:
            logger.error(f"Error fetching article IDs {article_ids}: {e}")


async def save_article_bodies(
    dataset_path: str,
    output_dir: str,
    num_concurrent_requests: int = 5,
    num_article_ids_per_request: int = 100,
):
    """
    Download the article bodies from PubMed-Ophtha and save the XML files.

    Args:
        dataset_path (str): Path to PubMed-Ophtha.
        output_dir (str): Folder to save the XML files to.
        num_concurrent_requests (int, optional): Number of concurrent downloads.
            Defaults to 5.
        num_article_ids_per_request (int, optional): Number of articles to fetch per
            concurrent request. Defaults to 100.

    """
    assert dataset_path.endswith(".parquet"), "Dataset path must be a Parquet file."
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."
    parquet_file = pq.ParquetFile(dataset_path)
    schema = parquet_file.schema
    table = parquet_file.read(
        columns=[field for field in schema.names if field != "panel_image_bytes"]
    )
    df = table.to_pandas()
    article_ids = df["article_id"].astype(str).unique().tolist()

    semaphore = asyncio.Semaphore(num_concurrent_requests)
    os.makedirs(output_dir, exist_ok=True)

    # Filter existing files to avoid re-downloading
    existing_files = glob.glob(os.path.join(output_dir, "*.xml"))
    existing_article_ids = {
        os.path.basename(f).removesuffix(".xml") for f in existing_files
    }
    article_ids = [aid for aid in article_ids if aid not in existing_article_ids]

    tasks = []
    for i in range(0, len(article_ids), num_article_ids_per_request):
        batch_article_ids = article_ids[
            i : min(i + num_article_ids_per_request, len(article_ids))
        ]
        tasks.append(get_body_text_safe(semaphore, batch_article_ids, output_dir))

    # Create progress bar
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f
