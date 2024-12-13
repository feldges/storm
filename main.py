from frontend.fasthtml.storm_fasthtml import app
from fasthtml.common import *

if __name__ == "__main__":
    serve(host="0.0.0.0", port=8001, reload=True)