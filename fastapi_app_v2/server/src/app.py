from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from itinerary_generator import ItineraryGenerator
from fastapi.middleware.cors import CORSMiddleware
import os

port = int(os.getenv("PORT", 8080))
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://travana.io",
    "https://travana.io/planner.html",
    "https://travana-trip-planner.firebaseapp.com",
    "https://travana-trip-planner.web.app",
    "https://travana-trip-planning-service-fastapi-v2-h53sgn7blq-uc.a.run.app/"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = ItineraryGenerator()

class ItineraryRequest(BaseModel):
    llm: str
    destination: str
    budget: str
    arrival_date: str
    departure_date: str
    start_time: str
    end_time: str
    additional_info: str

class FeedbackRequest(BaseModel):
    rating: int
    feedback: str

@app.post("/generate_itinerary")
def generate_itinerary_endpoint(request: ItineraryRequest):
    result = generator.generate_itinerary(
        request.llm, request.destination, request.budget, 
        request.arrival_date, request.departure_date,
        request.start_time, request.end_time, request.additional_info
    )
    return {"itinerary": result}

@app.post("/user_feedback")
def user_feedback_endpoint(request: FeedbackRequest):
    generator.user_feedback(request.rating, request.feedback)
    return {"status": "Feedback received successfully!"}
