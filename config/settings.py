# Get API key in environment
import os
from dotenv import load_dotenv
load_dotenv()

apikey = str(os.environ.get("ALPHAVANTAGE_API_KEY"))

print(apikey)

