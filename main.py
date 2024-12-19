from fasthtml.common import *
from frontend.fasthtml.storm_fasthtml import app
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    serve(host='0.0.0.0', port=port, reload=True)