from waitress import serve
from app_hw4 import app

serve(app.server, port=80)