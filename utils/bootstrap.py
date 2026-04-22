import os
import sys
import warnings
import logging

# Suppress annoying warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Set stdout to utf-8 to prevent Windows charmap errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the root project directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Apply DNS patch for Tailscale/IPv6 conflicts
try:
    import utils.dns_patch
except Exception:
    pass
