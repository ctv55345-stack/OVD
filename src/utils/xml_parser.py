from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List


def parse_flickr30k_entities(xml_path: str | Path) -> Dict[str, List[List[float]]]:
    """Parse a Flickr30k Entities-style XML into a mapping phrase -> list of boxes.

    Each box is [xmin, ymin, xmax, ymax] in pixel coordinates.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    phrase_to_boxes: Dict[str, List[List[float]]] = {}
    for obj in root.iter("object"):
        name_node = obj.find("name")
        bbox_node = obj.find("bndbox")
        if name_node is None or bbox_node is None:
            continue
        phrase = (name_node.text or "").strip()
        try:
            xmin = float(bbox_node.find("xmin").text)
            ymin = float(bbox_node.find("ymin").text)
            xmax = float(bbox_node.find("xmax").text)
            ymax = float(bbox_node.find("ymax").text)
        except Exception:
            continue

        phrase_to_boxes.setdefault(phrase, []).append([xmin, ymin, xmax, ymax])

    return phrase_to_boxes


