from waitress import serve
from app import app

serve(app.server, port=80)