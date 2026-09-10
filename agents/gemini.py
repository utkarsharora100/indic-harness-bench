"""Placeholder for the Gemini API adapter.

The adapter is intentionally disabled in Phase 1. Keep the connection code here so
that enabling Gemini later does not require changing the experiment runner.
"""

# import os
# from google import genai
#
#
# def create_gemini_client() -> genai.Client:
#     """Create a Gemini client from the GEMINI_API_KEY environment variable."""
#     api_key = os.environ["GEMINI_API_KEY"]
#     return genai.Client(api_key=api_key)
#
#
# Example usage once the Gemini condition is enabled:
#
# client = create_gemini_client()
# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Run the benchmark task.",
# )
# print(response.text)
