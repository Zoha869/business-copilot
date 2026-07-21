import networkx as nx
from groq import Groq
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))
graph=nx.DiGraph()