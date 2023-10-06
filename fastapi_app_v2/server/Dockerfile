FROM python:3.10

# Copy the code
COPY src /app
COPY travana-trip-planner-2c50910a2da4.json /app/keyfile.json
WORKDIR /app
ADD . /app

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/keyfile.json

# Install dependencies
# RUN pip install openai python-dotenv streamlit
RUN pip install --no-cache-dir -r src/requirements.txt

# Network port
EXPOSE 8080

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
