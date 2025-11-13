"""
    This script is used to test the token count of the JSON data and the Toon encoded data , TOON stands for Token-Oriented Object Notation which cuts down the token count by 50%.
"""

from toon import encode 
from dotenv import load_dotenv
import os,io,csv,sys
import json
import yaml
import ast 

from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-20b", temperature=0.7)

logger.remove()
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
logger.add(sys.stderr, level="INFO", format=log_format, colorize=True)

def json_to_ast(data):
    """Recursively converts a Python object (from JSON) to an AST node."""
    if isinstance(data, dict):
        keys = [ast.Constant(value=k) for k in data.keys()]
        values = [json_to_ast(v) for v in data.values()]
        return ast.Dict(keys=keys, values=values)
    elif isinstance(data, list):
        return ast.List(elts=[json_to_ast(item) for item in data])
    elif isinstance(data, (str, int, float, bool, type(None))):
        return ast.Constant(value=data)
    else:
        # Fallback for any other types
        return ast.Constant(value=str(data))

python_source_code = """ 
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import re

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

@dataclass
class TreeNode:
    name: str
    value: Optional[int] = None
    children: List["TreeNode"] = field(default_factory=list)
    
    def add(self, node: "TreeNode") -> None:
        self.children.append(node)
    
    def find(self, name: str) -> Optional["TreeNode"]:
        if self.name == name:
            return self
        for child in self.children:
            found = child.find(name)
            if found:
                return found
        return None

def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def flatten_tree(root: TreeNode) -> List[tuple]:
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append((node.name, node.value))
        stack.extend(reversed(node.children))
    return result

def group_by(items: List[tuple], key: Callable) -> Dict[str, List[tuple]]:
    result = {}
    for item in items:
        k = key(item)
        result.setdefault(k, []).append(item)
    return result

def make_pipeline(*funcs: Callable) -> Callable:
    def pipeline(x):
        for f in funcs:
            x = f(x)
        return x
    return pipeline

def load_json_safe(path: str) -> Dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return {"error": str(e)}

def find_words_in_file(path: str, pattern: str) -> List[str]:
    with open(path, "r") as f:
        text = f.read()
    return re.findall(pattern, text)

if __name__ == "__main__":
    root = TreeNode("root", 0)
    root.add(TreeNode("child1", 1))
    root.add(TreeNode("child2", 2))
    flat = flatten_tree(root)
    print("Tree flattened:", flat)
    print("GCD of 270 and 192:", gcd(270, 192))
 """

standard_json_data = [
    {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com",
        "phone": "1234567890",
        "address": "123 Main St Anytown, USA",
        "city": "Anytown",
        "state": "CA",
        "zipcode": "12345",
        "country": "USA",
        "is_active": True,
        "created_at": "2021-01-01"
    },
    {
        "name":"Christina Miller",
        "age": 25,
        "email": "christina.miller@example.com",
        "phone": "1234567890",
        "address": "123 Main St Anytown, USA",
        "city": "Anytown",
        "state": "CA",
        "zipcode": "12345",
        "country": "USA",
        "is_active": True,
        "created_at": "2021-01-01"
    },
    {
        "name":"Michael Brown",
        "age": 35,
        "email": "michael.brown@example.com",
        "phone": "1234567890",
        "address": "123 Main St Anytown, USA",
        "city": "Anytown",
        "state": "CA",
        "zipcode": "12345",
        "country": "USA",
        "is_active": True,
        "created_at": "2021-01-01"
    },
    {
        "name":"Emily Davis",
        "age": 28,
        "email": "emily.davis@example.com",
        "phone": "1234567890",
        "address": "123 Main St Anytown, USA",
        "city": "Anytown",
        "state": "CA",
        "zipcode": "12345", 
        "country": "USA",
        "is_active": True,
        "created_at": "2021-01-01"
    },
    {
        "name":"David Wilson",
        "age": 32,
        "email": "david.wilson@example.com",
        "phone": "1234567890",
            "address": "123 Main St Anytown, USA",
        "city": "Anytown",
        "state": "CA",
        "zipcode": "12345",
        "country": "USA",
        "is_active": True,
        "created_at": "2021-01-01"
    },
    {
        "name":"James Anderson",
        "age": 31,
        "email": "james.anderson@example.com",
        "phone": "1234567890",
        "address": "123 Main St Anytown, USA",
        "city": "Anytown",
        "state": "CA",
        "zipcode": "12345",
        "country": "USA",
        "is_active": True,
        "created_at": "2021-01-01"
    },
    {
        "name":"Sophia Martinez",
        "age": 27,
        "email": "sophia.martinez@example.com",
        "phone": "1234567890",
        "address": "123 Main St Anytown, USA",
        "city": "Anytown",
        "state": "CA",
        "zipcode": "12345",
        "country": "USA",
        "is_active": True,
        "created_at": "2021-01-01"
    }
]

nested_json_data = [
    {
  "orderId": "ORD-20230512-001",
  "orderDate": "2023-05-12T14:35:10Z",
  "customer": {
    "customerId": "CUST-A8B3",
    "firstName": "Eleanor",
    "lastName": "Vance",
    "email": "eleanor.v@example.com",
    "phone": "555-0101",
    "loyaltyStatus": "Gold",
    "preferences": {
      "contactlessDelivery": True,
      "notificationChannels": ["email", "sms"]
    }
  },
  "items": [
    {
      "productId": "PROD-LPT-01",
      "productName": "QuantumBook Pro 15-inch",
      "quantity": 1,
      "unitPrice": 2499.99,
      "isTaxable": True,
      "features": ["16GB RAM", "1TB SSD", "M3 Max Chip"],
      "supplier": {
        "supplierId": "SUP-TECH-45",
        "supplierName": "Core Technologies Inc.",
        "country": "USA"
      }
    },
    {
      "productId": "PROD-ACC-05",
      "productName": "USB-C HyperDock",
      "quantity": 1,
      "unitPrice": 129.50,
      "isTaxable": True,
      "features": None,
      "supplier": {
        "supplierId": "SUP-ACC-12",
        "supplierName": "ConnectAll Gadgets",
        "country": "Taiwan"
      }
    },
    {
      "productId": "PROD-SFT-11",
      "productName": "PixelForge Pro - 1 Year Subscription",
      "quantity": 1,
      "unitPrice": 99.99,
      "isTaxable": False,
      "features": ["Digital Delivery", "Cross-platform"],
      "supplier": {
        "supplierId": "SUP-SFT-01",
        "supplierName": "Creative Software Co.",
        "country": "Canada"
      }
    }
  ],
  "shippingDetails": {
    "shippingMethod": "Express",
    "trackingNumber": "1Z999AA10123456789",
    "estimatedDelivery": "2023-05-15T18:00:00Z",
    "address": {
      "street": "123 Hill House Lane",
      "city": "Shirley",
      "state": "MA",
      "zipCode": "01464",
      "country": "USA",
      "isResidential": True
    }
  },
  "paymentHistory": [
    {
      "transactionId": "TRN-A1B2C3D4",
      "paymentMethod": "CreditCard",
      "amount": 2729.48,
      "currency": "USD",
      "status": "Completed",
      "timestamp": "2023-05-12T14:36:00Z",
      "metadata": {
        "cardBrand": "Visa",
        "lastFourDigits": "4242",
        "authCode": "A-5823B"
      }
    }
  ],
  "orderStatus": "Shipped",
  "isActive": True
}
]

toon_encoded_data = encode(standard_json_data)

yaml_string = yaml.dump(standard_json_data, sort_keys=False)

json_compact_string = json.dumps(standard_json_data, separators=(',', ':'))

output = io.StringIO()
writer = csv.DictWriter(output, fieldnames=standard_json_data[0].keys())
writer.writeheader()
writer.writerows(standard_json_data)
csv_string = output.getvalue()

json_data_string = json.dumps(standard_json_data)

ast_tree = ast.parse(python_source_code)
ast_string = ast.dump(ast_tree)


nested_toon_encoded_data = encode(nested_json_data)

nested_yaml_string = yaml.dump(nested_json_data, sort_keys=False)

nested_json_compact_string = json.dumps(nested_json_data, separators=(',', ':'))

nested_csv_string = io.StringIO()
writer = csv.DictWriter(nested_csv_string, fieldnames=nested_json_data[0].keys())
writer.writeheader()
writer.writerows(nested_json_data)
nested_csv_string = nested_csv_string.getvalue()

nested_json_data_string = json.dumps(nested_json_data)

nested_ast_tree = ast.parse(python_source_code)
nested_ast_string = ast.dump(nested_ast_tree)

if not ast_string or len(ast_string) == 0:
    logger.warning("Warning: AST string is empty")
elif len(ast_string) > 10000:
    logger.warning(f"Warning: AST string is very large ({len(ast_string)} characters)")

def get_token_count(text_data: str) -> int:
    """Invokes the LLM to get the token count for the given text."""
    try:
        response = llm.invoke(text_data)
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if isinstance(usage, dict):
                return usage.get("input_tokens", 0)
            if hasattr(usage, 'input_tokens'):
                return usage.input_tokens
        return 0
    except Exception as e:
        print(f"Could not get token count: {e}")
        return 0

token_counts = {
    "Standard JSON": get_token_count(json_data_string),
    "TOON": get_token_count(toon_encoded_data),
    "Compact JSON": get_token_count(json_compact_string),
    "YAML": get_token_count(yaml_string),
    "CSV": get_token_count(csv_string),
    "AST": get_token_count(ast_string),
    "standard python code": get_token_count(python_source_code)
}

nested_token_counts = {
    "Nested JSON": get_token_count(nested_json_data_string),
    "TOON": get_token_count(nested_toon_encoded_data),
    "Compact JSON": get_token_count(nested_json_compact_string),
    "YAML": get_token_count(nested_yaml_string),
    "CSV": get_token_count(nested_csv_string),
    "AST": get_token_count(nested_ast_string),
    "standard python code": get_token_count(python_source_code)
}
print("--- Token Count Comparison ---")

sorted_counts = sorted(token_counts.items(), key=lambda item: item[1])
nested_sorted_counts = sorted(nested_token_counts.items(), key=lambda item: item[1])

logger.info("General JSON Data")

for format_name, count in sorted_counts:
    print(f"{format_name:<15}: {count} tokens")

logger.info("Nested JSON Data")

for format_name, count in nested_sorted_counts:
    print(f"{format_name:<15}: {count} tokens")
