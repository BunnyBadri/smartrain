"""
__init__.py - Application factory for SmartRain Flask app.
"""

from flask import Flask
from app.database import init_db

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'smartrain-secret-key-2026'

    # Initialize SQLite database
    init_db()

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app

