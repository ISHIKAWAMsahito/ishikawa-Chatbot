import logging
import uuid
import json
from collections import defaultdict  # ⬅️ 必要な import
from typing import List, Dict

chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
MAX_HISTORY_LENGTH = 20 