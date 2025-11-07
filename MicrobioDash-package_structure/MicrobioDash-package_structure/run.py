
import webbrowser
from microbiome_analysis_tool.app import app
from microbiome_analysis_tool.config import logger

if __name__ == '__main__':
    """
    This is the main entry point to run the Dash application.
    """
    try:
        port = 8050
        host = '127.0.0.1'
        url = f"http://{host}:{port}"

        logger.info(f"Starting Dash server on {url}")

        # Automatically open the web browser to the app's URL
        webbrowser.open(url)

        # Run the app. Setting debug=True is useful for development.
        # For production, consider using a production-ready WSGI server like gunicorn.
        app.run(host=host, port=port, debug=True)

    except Exception as e:
        logger.error(f"Failed to start the Dash server: {e}", exc_info=True)
