from waitress import serve
from app_hw4 import *

serve(app.server, port=80)