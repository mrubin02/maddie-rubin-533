from waitress import serve
import app

serve(app.server, port=80)