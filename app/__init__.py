"""
__init__.py - Application factory for SmartRain Flask app.
Explicitly binds absolute paths for templates and static assets for serverless hosts (Vercel).
"""

import os
from flask import Flask
from app.database import init_db

APP_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(APP_DIR, 'templates'),
        static_folder=os.path.join(APP_DIR, 'static'),
        static_url_path='/static'
    )
    app.config['SECRET_KEY'] = 'smartrain-secret-key-2026'

    # Initialize SQLite database
    init_db()

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
